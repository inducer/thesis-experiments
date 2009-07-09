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
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("--single", action="store_true")
    parser.add_option("--order", default=4, type="int")
    parser.add_option("--h", default=None, type="float")
    parser.add_option("--mesh-size", default=1, type="float")
    parser.add_option("--final-time", default=4e-10, type="float")
    parser.add_option("--vis-interval", default=0, type="int")
    parser.add_option("--profile", action="store_true")
    parser.add_option("--profile-cuda", action="store_true")
    parser.add_option("--extra-features")
    parser.add_option("--steps", type="int")
    parser.add_option("--cpu", action="store_true")
    parser.add_option("--local-watches", action="store_true")
    parser.add_option("-d", "--debug-flags", metavar="DEBUG_FLAG,DEBUG_FLAG")
    parser.add_option("--log-file", default="maxwell-%(order)s.dat")
    parser.add_option("--no-log-file", action="store_true")
    options, args = parser.parse_args()
    assert not args

    from hedge.element import TetrahedralElement
    from hedge.mesh import make_ball_mesh, make_cylinder_mesh, make_box_mesh
    from hedge.visualization import \
            VtkVisualizer, \
            SiloVisualizer, \
            get_rank_partition
    from hedge.tools import EOCRecorder, to_obj_array
    from math import sqrt, pi

    from os.path import dirname, join
    import sys
    sys.path.append(join(dirname(sys.argv[0]), "../../../hedge/examples/maxwell"))
    from analytic_solutions import \
            RealPartAdapter, \
            SplitComplexAdapter, \
            RectangularWaveguideMode, \
            RectangularCavityMode

    from hedge.backends import guess_run_context

    rcon = guess_run_context(["cuda", "mpi"])
    cpu_rcon = guess_run_context(["mpi"])

    if options.profile_cuda:
        import os
        import boostmpi as mpi
        os.environ["CUDA_PROFILE"] = "1"
        os.environ["CUDA_PROFILE_LOG"] = "cuda_profile_%d.log" % mpi.rank

    epsilon0 = 8.8541878176e-12 # C**2 / (N m**2)
    mu0 = 4*pi*1e-7 # N/A**2.
    epsilon = 1*epsilon0
    mu = 1*mu0

    eoc_rec = EOCRecorder()

    periodic = False

    if rcon.is_head_rank:
        print "----------------------------------------------------------------"
        print "ORDER %d" % options.order
        print "----------------------------------------------------------------"

    if periodic:
        mode = RectangularWaveguideMode(epsilon, mu, (5,4,3))
        periodicity = (False, False, True)
    else:
        periodicity = None
    mode = RectangularCavityMode(epsilon, mu, (1,1,1))

    if rcon.is_head_rank:
        if options.h:
            max_volume = options.h**3/6
        else:
            max_volume = 8e-5 / options.mesh_size

        mesh = make_box_mesh(max_volume=max_volume, periodicity=periodicity)
        #mesh = make_box_mesh(max_volume=0.0007, periodicity=periodicity)
                #return_meshpy_mesh=True
        #meshpy_mesh.write_neu(open("box.neu", "w"), 
                #bc={frozenset(range(1,7)): ("PEC", 1)})
        print "%d elements in entire mesh" % len(mesh.elements)

    if rcon.is_head_rank:
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()
    if hasattr(mesh_data, "mesh"):
        print "%d elements on rank %d" % (len(mesh_data.mesh.elements), rcon.rank)

    from hedge.models.em import MaxwellOperator
    op = MaxwellOperator(epsilon, mu, flux_type=1)

    debug_flags = [ ]
    if not options.cpu:
        debug_flags.append("cuda_no_plan_el_local")

    if options.debug_flags:
        debug_flags.extend(options.debug_flags.split(","))

    from hedge.backends.jit import Discretization as CPUDiscretization
    from hedge.backends.cuda import Discretization as GPUDiscretization
    if options.cpu:
        discr = cpu_rcon.make_discretization(mesh_data, order=options.order, debug=debug_flags,
                default_scalar_type=numpy.float32 if options.single else numpy.float64)
    else:
        from pycuda.driver import device_attribute
        discr = rcon.make_discretization(mesh_data, order=options.order, debug=debug_flags,
                default_scalar_type=numpy.float32 if options.single else numpy.float64,
                tune_for=op.op_template(),
                mpi_cuda_dev_filter=lambda dev: 
                dev.get_attribute(device_attribute.MULTIPROCESSOR_COUNT) > 2)

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

    # diagnostics setup ---------------------------------------------------
    from pytools.log import LogManager, add_general_quantities, \
            add_simulation_quantities, add_run_info

    if options.no_log_file:
        log_file_name = None
    else:
        log_file_name = options.log_file % {"order": options.order}

    logmgr = LogManager(log_file_name, "w", rcon.communicator)

    if options.extra_features:
        for feat in options.extra_features.split(","):
            colon_index = feat.find(":")
            if colon_index != -1:
                logmgr.set_constant(
                        feat[:colon_index],
                        eval(feat[colon_index+1:]))

    add_run_info(logmgr)
    add_general_quantities(logmgr)
    add_simulation_quantities(logmgr, dt)
    discr.add_instrumentation(logmgr)

    if options.cpu:
        from hedge.timestep import RK4TimeStepper
    else:
        from hedge.backends.cuda.tools import RK4TimeStepper

    stepper = RK4TimeStepper()
    stepper.add_instrumentation(logmgr)

    logmgr.add_watches(["step.loc", "t_sim.loc", "t_step.loc"])
    if options.cpu:
        if not options.local_watches:
            logmgr.add_watches([
                ("flops/s", "(n_flops_gather.sum+n_flops_lift.sum+n_flops_mass.sum+n_flops_diff.sum)"
                #"/(t_gather+t_lift+t_diff)"
                "/t_step.max"
                )
                ])
    else:
        if options.local_watches:
            logmgr.add_watches([
                ("t_compute", 
                    "t_diff.loc+t_gather.loc+t_el_local.loc+t_rk4.loc+t_vector_math.loc"),
                ])
        else:
            logmgr.add_watches([
            ("t_compute", "t_diff.max+t_gather.max+t_el_local.max+t_rk4.max+t_vector_math.max"),
            ("flops/s", "(n_flops_gather.sum+n_flops_lift.sum+n_flops_mass.sum+n_flops_diff.sum+n_flops_vector_math.sum+n_flops_rk4.sum)"
            #"/(t_gather.max+t_el_local.max+t_diff.max+t_vector_math.max+t_rk4.max)"
            "/t_step.max"
            )
            ])

    logmgr.set_constant("h", options.h)
    logmgr.set_constant("mesh_size", options.mesh_size)
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

    if options.profile:
        from cProfile import Profile
        from lsprofcalltree import KCacheGrind
        prof = Profile()

        rhs(0, boxed_fields[0]) # keep init traffic out of profile

        try:
            prof.runcall(timestep_loop)
            fields = boxed_fields[0]
        finally:
            kg = KCacheGrind(prof)
            import sys
            from hedge.tools import get_rank
            kg.output(open(
                "profile-%s-rank-%d.log" % (sys.argv[0], get_rank(discr)),
                "w"))
    else:
        timestep_loop()
        fields = boxed_fields[0]

    numpy.seterr('raise')
    mode.set_time(boxed_t[0])

    from hedge.tools import relative_error
    true_fields = discr.convert_volume(to_obj_array(mode(discr)
        .real.astype(discr.default_scalar_type)), kind=discr.compute_kind)

    l2_diff = discr.norm(fields-true_fields) 
    l2_true = discr.norm(true_fields)
    relerr = relative_error(l2_diff, l2_true)

    logmgr.set_constant("relerr", relerr)
    logmgr.close()

    if rcon.is_head_rank:
        print "rel L2 error: %g" % relerr

    discr.close()

if __name__ == "__main__":
    main()
