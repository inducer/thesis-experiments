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

    def forces(t):
        return force_const

    from vlasov import VlasovOperator
    op = VlasovOperator(units, 1, forces,
            grid_size=16, filter_type="exponential",
            hard_scale=5, bounded_fraction=0.8,
            filter_parameters=dict(preservation_ratio=0.3))

    x_vec = discr.interpolate_volume_function(lambda x, el: x[0])
    force_const = [[1*x_vec] for v in op.velocity_points]

    sine_vec = discr.interpolate_volume_function(lambda x, el: cos(0.5*x[0]))
    from hedge.tools import make_obj_array

    densities = make_obj_array([
        sine_vec.copy()*exp(-(0.5*v[0]**2))*v[0]
        for v in op.p_discr.quad_points])

    # timestep setup ----------------------------------------------------------
    stepper = RK4TimeStepper()

    dt = discr.dt_factor(op.max_eigenvalue())
    nsteps = int(10/dt)

    print "%d elements, dt=%g, nsteps=%d" % (
            len(discr.mesh.elements), dt, nsteps)

    # diagnostics setup -------------------------------------------------------
    from pytools.log import LogManager, \
            add_general_quantities, \
            add_simulation_quantities, \
            add_run_info

    logmgr = LogManager("vlasov.dat", "w", rcon.communicator)
    add_run_info(logmgr)
    add_general_quantities(logmgr)
    add_simulation_quantities(logmgr, dt)
    discr.add_instrumentation(logmgr)

    stepper.add_instrumentation(logmgr)

    logmgr.add_watches(["step.max", "t_sim.max", "t_step.max"])

    # timestep loop -----------------------------------------------------------
    rhs = op.bind(discr)

    from vlasov import add_densities_to_silo

    try:
        for step in xrange(nsteps):
            logmgr.tick()

            t = step*dt

            if step % 20 == 0:
                with vis.make_file("vlasov-%04d.silo" % step) as visf:
                    #op.visualize_densities_with_matplotlib(discr,
                            #"vlasov-%04d.png" % step, densities)
                    add_densities_to_silo(visf, op, discr, densities)

            densities = stepper(densities, t, dt, rhs)
    finally:
        logmgr.close()
        discr.close()




if __name__ == "__main__":
    main()

