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
    from hedge.mesh import make_ball_mesh, make_cylinder_mesh, make_box_mesh
    from hedge.visualization import \
            VtkVisualizer, \
            SiloVisualizer, \
            get_rank_partition
    from hedge.tools import EOCRecorder, to_obj_array
    from math import sqrt, pi

    import sys
    sys.path.append("../../../hedge/examples/maxwell")
    from analytic_solutions import \
            check_time_harmonic_solution, \
            RealPartAdapter, \
            SplitComplexAdapter, \
            RectangularWaveguideMode, \
            RectangularCavityMode

    from hedge.backends import guess_run_context

    rcon = guess_run_context()
    cpu_rcon = guess_run_context(disable=set(["cuda"]))

    epsilon0 = 8.8541878176e-12 # C**2 / (N m**2)
    mu0 = 4*pi*1e-7 # N/A**2.
    epsilon = 1*epsilon0
    mu = 1*mu0

    eoc_rec = EOCRecorder()

    periodic = False

    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("--single", action="store_true")
    parser.add_option("--order", default=4, type="int")
    parser.add_option("--h", default=0.08, type="float")
    parser.add_option("--final-time", default=4e-10, type="float")
    parser.add_option("--vis-interval", default=0, type="int")
    parser.add_option("--steps", type="int")
    parser.add_option("--cpu", action="store_true")
    parser.add_option("-d", "--debug-flags", metavar="DEBUG_FLAG,DEBUG_FLAG")
    parser.add_option("--log-file", default="maxwell-%(order)s.dat")
    options, args = parser.parse_args()
    assert not args

    if rcon.is_head_rank:
        print "----------------------------------------------------------------"
        print "ORDER %d" % options.order
        print "----------------------------------------------------------------"

    if periodic:
        mode = RectangularWaveguideMode(epsilon, mu, (5,4,3))
        periodicity = (False, False, True)
    else:
        periodicity = None
    mode = RectangularCavityMode(epsilon, mu, (4,3,3))

    if rcon.is_head_rank:
        mesh = make_box_mesh(max_volume=options.h**3/6, periodicity=periodicity)
        #mesh = make_box_mesh(max_volume=0.0007, periodicity=periodicity)
                #return_meshpy_mesh=True
        #meshpy_mesh.write_neu(open("box.neu", "w"), 
                #bc={frozenset(range(1,7)): ("PEC", 1)})

    if rcon.is_head_rank:
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    from hedge.pde import MaxwellOperator
    op = MaxwellOperator(epsilon, mu, flux_type=1)

    debug_flags = [
        #"cuda_flux", 
        #"cuda_debugbuf"
        #"cuda_lift_plan",
        #"cuda_diff_plan",
        #"cuda_gather_plan",
        #"cuda_dumpkernels",
        ]
    if options.debug_flags:
        for f in options.debug_flags.split(","):
            debug_flags.append(f)

    from hedge.backends.jit import Discretization as CPUDiscretization
    from hedge.backends.cuda import Discretization as GPUDiscretization
    if options.cpu:
        discr = cpu_rcon.make_discretization(mesh_data, order=options.order, debug=debug_flags,
                default_scalar_type=numpy.float32 if options.single else numpy.float64)

        cpu_discr = discr
    else:
        from pycuda.driver import device_attribute
        discr = rcon.make_discretization(mesh_data, order=options.order, debug=debug_flags,
                tune_for=op.op_template(),
                mpi_cuda_dev_filter=lambda dev: 
                dev.get_attribute(device_attribute.MULTIPROCESSOR_COUNT) > 2)

        cpu_discr = cpu_rcon.make_discretization(mesh_data, order=options.order)

    if options.vis_interval:
        #vis = VtkVisualizer(discr, rcon, "em-%d" % options.order)
        vis = SiloVisualizer(discr, rcon)

    mode.set_time(0)
    boxed_fields = [discr.convert_volume(to_obj_array(mode(discr)
        .real.astype(discr.default_scalar_type)), kind=discr.compute_kind)]

    dt = discr.dt_factor(op.max_eigenvalue())
    if options.steps is None:
        nsteps = int(options.final_time/dt)+1
        dt = options.final_time/nsteps
    else:
        nsteps = options.steps

    boxed_t = [0]

    #check_time_harmonic_solution(discr, mode, c_sol)
    #continue
    
    # diagnostics setup ---------------------------------------------------
    from pytools.log import LogManager, add_general_quantities, \
            add_simulation_quantities, add_run_info

    logmgr = LogManager(options.log_file % {"order": options.order}, 
            "w", rcon.communicator)
    add_run_info(logmgr)
    add_general_quantities(logmgr)
    add_simulation_quantities(logmgr, dt)
    discr.add_instrumentation(logmgr)

    if isinstance(discr, CPUDiscretization):
        from hedge.timestep import RK4TimeStepper
    else:
        from hedge.backends.cuda.tools import RK4TimeStepper

    stepper = RK4TimeStepper()
    stepper.add_instrumentation(logmgr)

    if isinstance(discr, CPUDiscretization):
        logmgr.add_watches(["step.max", "t_sim.max", "t_step.max",
            ("flops/s", "(n_flops_gather+n_flops_lift+n_flops_mass+n_flops_diff)"
            "/(t_gather+t_lift+t_mass+t_diff)")
            ])
    else:
        logmgr.add_watches(["step.max", "t_sim.max", "t_step.max", 
            ("t_compute", "t_diff.max+t_gather.max+t_lift.max+t_rk4.max+t_vector_math.max"),
            ("flops/s", "(n_flops_gather.sum+n_flops_lift.sum+n_flops_mass.sum+n_flops_diff.sum+n_flops_vector_math.sum+n_flops_rk4.sum)"
            "/(t_gather.max+t_lift.max+t_mass.max+t_diff.max+t_vector_math.max+t_rk4.max)")
            ])

    logmgr.set_constant("h", options.h)
    logmgr.set_constant("is_cpu", options.cpu)

    # timestep loop -------------------------------------------------------

    rhs = op.bind(discr)

    import gc

    def timestep_loop():
        for step in range(nsteps):
            logmgr.tick()

            if options.vis_interval and step % options.vis_interval == 0:
                e, h = op.split_eh(boxed_fields[0])
                visf = vis.make_file("em-%d-%04d" % (options.order, step))
                vis.add_data(visf, [ 
                    ("e", discr.convert_volume(e, kind="numpy")), 
                    ("h", discr.convert_volume(h, kind="numpy")), 
                    ],
                    time=boxed_t[0], step=step)
                visf.close()

            boxed_fields[0] = stepper(boxed_fields[0], boxed_t[0], dt, rhs)
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

    from hedge.tools import relative_error
    true_fields = to_obj_array(mode(cpu_discr).real.copy())

    total_diff = 0
    total_true = 0
    for i, (f, cpu_tf) in enumerate(zip(fields, true_fields)):
        cpu_f = discr.convert_volume(f, kind="numpy")

        l2_diff = cpu_discr.norm(cpu_f-cpu_tf) 
        l2_true = cpu_discr.norm(cpu_tf)

        total_diff += l2_diff**2
        total_true += l2_true**2

    relerr = relative_error(total_diff**0.5, total_true**0.5)
    logmgr.set_constant("relerr", relerr)
    logmgr.close()

    if rcon.is_head_rank:
        print "rel L2 error: %g" % relerr

    discr.close()

if __name__ == "__main__":
    main()
