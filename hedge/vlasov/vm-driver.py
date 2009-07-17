from __future__ import division, with_statement
import numpy




def main():
    from hedge.timestep import RK4TimeStepper
    from hedge.tools import mem_checkpoint
    from math import sin, cos, pi, sqrt, exp
    from math import floor

    from hedge.backends import guess_run_context
    rcon = guess_run_context()

    from vm_interface import VlasovMaxwellCPyUserInterface
    setup = VlasovMaxwellCPyUserInterface().gather()

    discr = rcon.make_discretization(setup.x_mesh, order=setup.x_dg_order,
            debug=setup.discr_debug_flags+[ "jit_dont_optimize_large_exprs", ])

    from hedge.visualization import SiloVisualizer
    vis = SiloVisualizer(discr, rcon)

    # operator setup ----------------------------------------------------------
    from hedge.models.em import TE1DMaxwellOperator
    max_op = TE1DMaxwellOperator(setup.epsilon, setup.mu, 
            flux_type=1, dimensions=setup.x_mesh.dimensions)

    from vlasov import VlasovMaxwellOperator
    vlas_op = VlasovMaxwellOperator(
            x_dim=setup.x_mesh.dimensions, v_dim=setup.v_dim,
            maxwell_op=max_op, units=setup.units,
            species_mass=setup.species_mass, 
            species_charge=setup.species_charge,
            grid_size=setup.p_grid_size, **setup.p_discr_args)

    print "v grid:", [setup.units.v_from_p(vlas_op.species_mass, p)
            for p in vlas_op.p_discr.quad_points_1d]

    densities = setup.get_densities(discr, vlas_op)

    def get_max_fields():
        e, h = setup.get_eh(discr, vlas_op, densities)
        return max_op.assemble_eh(e=e, h=h, discr=discr)

    from hedge.tools import make_obj_array, join_fields
    fields = join_fields(get_max_fields(), densities)

    # timestep setup ----------------------------------------------------------
    stepper = RK4TimeStepper()

    dt = discr.dt_factor(vlas_op.max_eigenvalue()) * setup.dt_scale
    nsteps = int(setup.final_time/dt)

    print "%d elements, dt=%g, nsteps=%d" % (
            len(discr.mesh.elements), dt, nsteps)

    # diagnostics setup -------------------------------------------------------
    from pytools.log import LogManager, \
            add_general_quantities, \
            add_simulation_quantities, \
            add_run_info

    logmgr = LogManager("vmax.dat", "w", rcon.communicator)
    add_run_info(logmgr)
    add_general_quantities(logmgr)
    add_simulation_quantities(logmgr, dt)
    discr.add_instrumentation(logmgr)

    stepper.add_instrumentation(logmgr)

    logmgr.add_watches(["step.max", "t_sim.max", "t_step.max"])

    # timestep loop -----------------------------------------------------------
    rhs = vlas_op.bind(discr)
    j_op = vlas_op.bind(discr, vlas_op.j(
        vlas_op.make_densities_placeholder()))

    forces_ops = [vlas_op.bind(discr, force_op)
            for force_op in vlas_op.forces_T(
                vlas_op.make_densities_placeholder(),
                *vlas_op.make_maxwell_eh_placeholders())]

    def real_part(fld):
        from hedge.tools import with_object_array_or_scalar
        return with_object_array_or_scalar(lambda x: x.real, fld)

    try:
        for step in xrange(nsteps):
            logmgr.tick()

            t = step*dt

            if step % setup.vis_interval == 0:
                with vis.make_file("vmax-%04d" % step) as visf:
                    e, h, densities = vlas_op.split_e_h_densities(fields)
                    from vlasov import add_xv_to_silo

                    AXES = ["x", "y", "z"]

                    add_xv_to_silo(visf, vlas_op, discr, [
                        ("f", densities),
                        ] + [
                        ("forces"+AXES[i], force_op(t, fields))
                        for i, force_op in enumerate(forces_ops)
                        ])
                    vis.add_data(visf, [
                        ("e", real_part(e)),
                        ("h", real_part(h)),
                        ("j", real_part(j_op(t, fields))),
                        ], time=t, step=step)

            fields = stepper(fields, t, dt, rhs)
    finally:
        logmgr.close()
        discr.close()




if __name__ == "__main__":
    main()
