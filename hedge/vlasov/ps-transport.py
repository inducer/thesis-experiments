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
    mesh = make_uniform_1d_mesh(-pi/8, pi/8, 10, periodic=True)

    discr = rcon.make_discretization(mesh, order=3,
            debug=[
                #"print_op_code",
                "jit_wait_on_compile_error",
                "jit_dont_optimize_large_exprs",
                ])

    from hedge.visualization import SiloVisualizer
    vis = SiloVisualizer(discr, rcon)

    # operator setup ----------------------------------------------------------
    from pyrticle.units import SIUnitsWithUnityConstants
    units = SIUnitsWithUnityConstants()

    def forces_T(t):
        return force_T_const

    from vlasov import PhaseSpaceTransportOperator
    from p_discr import MomentumDiscretization
    op = PhaseSpaceTransportOperator(x_discr=discr, 
            p_discrs=[
                MomentumDiscretization(32,
                    filter_type="exponential",
                    hard_scale=0.6, 
                    bounded_fraction=0.8,
                    use_fft=False,
                    filter_parameters=dict(preservation_ratio=0.3))],
            units=units, species_mass=1, forces_T_func=forces_T)

    from hedge.tools import make_obj_array

    x_vec = discr.interpolate_volume_function(lambda x, el: x[0])
    force_T_const = [
            make_obj_array([-1*x_vec for p in op.p_grid]),
            make_obj_array([0*x_vec for p in op.p_grid]),
            ]

    sine_vec = discr.interpolate_volume_function(
            lambda x, el: cos(4*x[0]))

    densities = make_obj_array([
        sine_vec.copy()*exp(-(32*numpy.dot(v, v)))*v[0]
        for v in op.v_points])

    # timestep setup ----------------------------------------------------------
    stepper = RK4TimeStepper()

    dt = discr.dt_factor(op.max_eigenvalue()) * 0.5
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
    from vlasov import add_xp_to_silo

    try:
        for step in xrange(nsteps):
            logmgr.tick()

            t = step*dt

            if step % 20 == 0:
                with vis.make_file("vlasov-%04d" % step) as visf:
                    #op.visualize_densities_with_matplotlib(discr,
                            #"vlasov-%04d.png" % step, densities)
                    add_xp_to_silo(visf, op, discr, [
                        ("f", densities)
                        ])

            densities = stepper(densities, t, dt, op)

            if step % 20 == 0:
                densities = vlasov_op.apply_filter(densities)

    finally:
        logmgr.close()
        discr.close()




if __name__ == "__main__":
    main()

