from __future__ import division, with_statement




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
    vlas_op = VlasovMaxwellOperator(max_op, units,
            species_mass=1, species_charge=1,
            grid_size=32, filter_type="exponential",
            hard_scale=5, bounded_fraction=0.9,
            filter_parameters=dict(eta_cutoff=0.3))

    sine_vec = discr.interpolate_volume_function(lambda x, el: cos(0.5*x[0]))
    from hedge.tools import make_obj_array, join_fields

    densities = make_obj_array([
        sine_vec.copy()*exp(-(0.5*v[0]**2))*v[0]
        for v in vlas_op.p_discr.quad_points])

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
    forces_op = vlas_op.bind(discr, vlas_op.forces(
        vlas_op.make_densities_placeholder(),
        *vlas_op.make_maxwell_eh_placeholders()
        ))

    def real_part(fld):
        from hedge.tools import with_object_array_or_scalar
        return with_object_array_or_scalar(lambda x: x.real, fld)

    try:
        for step in xrange(nsteps):
            logmgr.tick()

            t = step*dt

            if step % 20 == 0:
                with vis.make_file("vlasov-%04d" % step) as visf:
                    e, h, densities = vlas_op.split_e_h_densities(fields)
                    forces = [force_i[0] 
                            for force_i in forces_op(t, fields)]

                    from vlasov import add_xv_to_silo

                    add_xv_to_silo(visf, vlas_op, discr, [
                        ("f", densities),
                        ("forces", forces),
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

