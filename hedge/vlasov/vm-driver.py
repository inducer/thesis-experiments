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
    vlasov_op = VlasovMaxwellOperator(
            x_discr=discr,
            p_discrs=setup.p_discrs,
            maxwell_op=max_op, units=setup.units,
            species_mass=setup.species_mass, 
            species_charge=setup.species_charge,
            use_fft=setup.use_fft)

    print "v grids:"
    for i, p_discr in enumerate(vlasov_op.p_discrs):
        print i, [setup.units.v_from_p(vlasov_op.species_mass, p)
            for p in p_discr.quad_points_1d]

    densities = setup.get_densities(discr, vlasov_op)

    def get_max_fields():
        e, h = setup.get_eh(discr, vlasov_op, densities)
        return max_op.assemble_eh(e=e, h=h, discr=discr)

    from hedge.tools import make_obj_array, join_fields
    fields = join_fields(get_max_fields(), densities)

    # timestep setup ----------------------------------------------------------
    stepper = RK4TimeStepper()

    dt = discr.dt_factor(vlasov_op.max_eigenvalue()) * setup.dt_scale
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

    from hedge.log import add_em_quantities
    from log import VlasovMaxwellFGetter, add_density_quantities
    field_getter = VlasovMaxwellFGetter(discr, max_op, vlasov_op, lambda: fields)
    add_em_quantities(logmgr, max_op, field_getter)
    add_density_quantities(logmgr, vlasov_op, field_getter)

    # timestep loop -----------------------------------------------------------
    def real_part(fld):
        from hedge.tools import with_object_array_or_scalar
        return with_object_array_or_scalar(lambda x: x.real, fld)

    try:
        for step in xrange(nsteps):
            logmgr.tick()

            t = step*dt

            if step % setup.vis_interval == 0:
                with vis.make_file("vmax-%04d" % step) as visf:
                    e, h, densities = vlasov_op.split_e_h_densities(fields)
                    from vlasov import add_xp_to_silo

                    AXES = ["x", "y", "z"]

                    add_xp_to_silo(visf, vlasov_op, discr, [
                        ("f", densities),
                        ] + [
                        ("forces"+AXES[i], axis_forces)
                        for i, axis_forces in enumerate(
                            vlasov_op.forces_T(densities, e, h))
                        ])
                    vis.add_data(visf, [
                        ("e", real_part(e)),
                        ("h", real_part(h)),
                        ("j", real_part(vlasov_op.j(fields))),
                        ], time=t, step=step)

            fields = stepper(fields, t, dt, vlasov_op)
    finally:
        logmgr.close()
        discr.close()




if __name__ == "__main__":
    main()
