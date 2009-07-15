from __future__ import division, with_statement
import numpy




def main():
    from hedge.timestep import RK4TimeStepper
    from hedge.tools import mem_checkpoint
    from math import sin, cos, pi, sqrt, exp
    from math import floor

    from hedge.backends import guess_run_context
    rcon = guess_run_context()

    from hedge.mesh import make_uniform_1d_mesh
    mesh = make_uniform_1d_mesh(-pi, pi, 20, periodic=True)

    discr = rcon.make_discretization(mesh, order=4,
            debug=[
                #"print_op_code",
                "jit_wait_on_compile_error",
                "jit_dont_optimize_large_exprs",
                ])

    from hedge.visualization import SiloVisualizer
    vis = SiloVisualizer(discr, rcon)

    # operator setup ----------------------------------------------------------
    from pyrticle.units import SI
    units = SI()

    from hedge.models.em import TE1DMaxwellOperator
    max_op = TE1DMaxwellOperator(1, 1, flux_type=1,
            dimensions=1)

    from vlasov import VlasovMaxwellOperator
    vlas_op = VlasovMaxwellOperator(
            x_dim=1, v_dim=2,
            maxwell_op=max_op, units=units,
            species_mass=1, species_charge=1,
            grid_size=16, filter_type="exponential",
            hard_scale=5, bounded_fraction=0.8,
            filter_parameters=dict(eta_cutoff=0.3))

    base_vec = discr.interpolate_volume_function(lambda x, el: cos(0.5*x[0]))
    #base_vec = discr.interpolate_volume_function(lambda x, el: 1)
    from hedge.tools import make_obj_array, join_fields

    densities = make_obj_array([
        base_vec*exp(-(0.5*numpy.dot(v, v)))#*v[0]
        for v in vlas_op.v_points])

    fields = join_fields(
            max_op.assemble_eh(discr=discr),
            densities)

    # timestep setup ----------------------------------------------------------
    stepper = RK4TimeStepper()

    dt = discr.dt_factor(vlas_op.max_eigenvalue()) / 10
    nsteps = int(10/dt)

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

            if step % 100 == 0:
                with vis.make_file("vlasov-%04d" % step) as visf:
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

