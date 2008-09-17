# Hedge - the Hybrid'n'Easy DG Environment
# Copyright (C) 2007 Andreas Kloeckner
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.




from __future__ import division
import numpy
import numpy.linalg as la




def main():
    from hedge.element import TetrahedralElement
    from hedge.timestep import RK4TimeStepper
    from hedge.mesh import make_ball_mesh, make_cylinder_mesh, make_box_mesh
    from hedge.visualization import \
            VtkVisualizer, \
            SiloVisualizer, \
            get_rank_partition
    from hedge.tools import EOCRecorder, to_obj_array
    from math import sqrt, pi
    from analytic_solutions import \
            check_time_harmonic_solution, \
            RealPartAdapter, \
            SplitComplexAdapter, \
            CartesianAdapter, \
            CylindricalCavityMode, \
            RectangularWaveguideMode, \
            RectangularCavityMode
    from hedge.pde import MaxwellOperator
    from hedge.parallel import guess_parallelization_context

    pcon = guess_parallelization_context()

    epsilon0 = 8.8541878176e-12 # C**2 / (N m**2)
    mu0 = 4*pi*1e-7 # N/A**2.
    epsilon = 1*epsilon0
    mu = 1*mu0

    eoc_rec = EOCRecorder()

    cylindrical = False
    periodic = False

    if cylindrical:
        R = 1
        d = 2
        mode = CylindricalCavityMode(m=1, n=1, p=1,
                radius=R, height=d, 
                epsilon=epsilon, mu=mu)
        r_sol = CartesianAdapter(RealPartAdapter(mode))
        c_sol = SplitComplexAdapter(CartesianAdapter(mode))

        if pcon.is_head_rank:
            mesh = make_cylinder_mesh(radius=R, height=d, max_volume=0.01)
    else:
        if periodic:
            mode = RectangularWaveguideMode(epsilon, mu, (3,2,1))
            periodicity = (False, False, True)
        else:
            periodicity = None
        mode = RectangularCavityMode(epsilon, mu, (1,2,2))

        if pcon.is_head_rank:
            #mesh = make_box_mesh(max_volume=0.001, periodicity=periodicity)
            mesh = make_box_mesh(max_volume=0.0001, periodicity=periodicity)

    if pcon.is_head_rank:
        mesh_data = pcon.distribute_mesh(mesh)
    else:
        mesh_data = pcon.receive_mesh()

    order = 4
    #from hedge.discr_precompiled import Discretization
    from hedge.cuda import Discretization
    discr = Discretization(mesh_data, order=order, 
            #debug=["cuda_flux", "cuda_debugbuf"]
            )

    #vis = VtkVisualizer(discr, pcon, "em-%d" % order)
    vis = SiloVisualizer(discr, pcon)

    mode.set_time(0)
    boxed_fields = [discr.volume_to_gpu(to_obj_array(mode(discr)
        .real.astype(discr.default_scalar_type)))]
    op = MaxwellOperator(discr, epsilon, mu, upwind_alpha=1)

    dt = discr.dt_factor(op.max_eigenvalue())
    final_time = 4e-10
    nsteps = int(final_time/dt)+1
    dt = final_time/nsteps

    boxed_t = [0]

    #check_time_harmonic_solution(discr, mode, c_sol)
    #continue

    stepper = RK4TimeStepper()
    
    # diagnostics setup ---------------------------------------------------
    from pytools.log import LogManager, add_general_quantities, \
            add_simulation_quantities, add_run_info

    logmgr = LogManager("maxwell-%d.dat" % order, "w", pcon.communicator)
    add_run_info(logmgr)
    add_general_quantities(logmgr)
    add_simulation_quantities(logmgr, dt)
    discr.add_instrumentation(logmgr)
    stepper.add_instrumentation(logmgr)

    logmgr.add_watches(["step.max", "t_sim.max", "t_step.max", 
        "t_diff_op+t_inner_flux",
        "n_flops/(t_diff_op+t_inner_flux)"
        ])
    # timestep loop -------------------------------------------------------

    def timestep_loop():
        for step in range(nsteps):
            logmgr.tick()

            if step % 100 == 0:
                e, h = op.split_eh(boxed_fields[0])
                visf = vis.make_file("em-%d-%04d" % (order, step))
                vis.add_data(visf,
                        [ ("e", 
                            #e
                            discr.volume_from_gpu(e)
                            ), 
                            ("h", 
                                #h
                                discr.volume_from_gpu(h)
                                ), 
                            ],
                        time=boxed_t[0], step=step
                        )
                visf.close()

            boxed_fields[0] = stepper(boxed_fields[0], boxed_t[0], dt, op.rhs)
            boxed_t[0] += dt

    if False:
        from cProfile import Profile
        from lsprofcalltree import KCacheGrind
        prof = Profile()
        try:
            prof.runcall(timestep_loop)
            fields = boxed_fields[0]
        finally:
            kg = KCacheGrind(prof)
            import sys
            kg.output(file(sys.argv[0]+".log", 'w'))
    else:
        timestep_loop()
        fields = boxed_fields[0]

    numpy.seterr('raise')
    mode.set_time(boxed_t[0])
    true_fields = to_obj_array(mode(discr).real)

    for i, (f, cpu_tf) in enumerate(zip(fields, true_fields)):
        cpu_f = discr.volume_from_gpu(f).astype(numpy.float64)

        from hedge.tools import relative_error
        print "comp %i: rel error l2 (not L2) norm: %g" % (
                i, relative_error(la.norm(cpu_f-cpu_tf), la.norm(cpu_tf)))

if __name__ == "__main__":
    import cProfile as profile
    #profile.run("main()", "wave2d.prof")
    main()
