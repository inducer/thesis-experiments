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




def main() :
    from hedge.visualization import SiloVisualizer, VtkVisualizer
    from pytools.stopwatch import Job
    from math import sin, cos, pi, exp, sqrt
    from hedge.backends import guess_run_context

    rcon = guess_run_context()
    cpu_rcon = guess_run_context(disable=set(["cuda"]))

    dim = 2

    if dim == 1:
        if rcon.is_head_rank:
            from hedge.mesh import make_uniform_1d_mesh
            mesh = make_uniform_1d_mesh(-10, 10, 500)

    elif dim == 2:
        from hedge.mesh import \
                make_disk_mesh, \
                make_regular_square_mesh, \
                make_square_mesh, \
                make_rect_mesh
        if rcon.is_head_rank:
            #mesh = make_disk_mesh(max_area=5e-3)
            #mesh = make_regular_square_mesh(
                    #n=9, periodicity=(True,True))
            mesh = make_rect_mesh(a=(-0.5,-0.5),b=(3.5,0.5),max_area=0.008)
            #mesh.transform(Rotation(pi/8))
    elif dim == 3:
        from hedge.mesh import make_ball_mesh
        if rcon.is_head_rank:
            mesh = make_ball_mesh(max_volume=0.005)
    else:
        raise RuntimeError, "bad number of dimensions"

    if rcon.is_head_rank:
        print "%d elements" % len(mesh.elements)
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    #from hedge.discr_precompiled import Discretization
    discr = rcon.make_discretization(mesh_data, order=4,
            debug=[
                "cuda_no_plan",
                "cuda_dumpkernels",
                ]
            )
    cpu_discr = cpu_rcon.make_discretization(mesh_data, order=4,
            default_scalar_type=numpy.float32)

    from hedge.timestep import RK4TimeStepper
    stepper = RK4TimeStepper()

    #stepper = AdamsBashforthTimeStepper(1)
    #vis = VtkVisualizer(discr, rcon, "fld")
    vis = SiloVisualizer(discr, rcon)

    def source_u(x, el):
        return exp(-numpy.dot(x, x)*128)

    source_u_vec = discr.interpolate_volume_function(source_u)

    def source_vec_getter(t):
        from math import sin
        return source_u_vec*sin(10*t)

    from hedge.pde import StrongWaveOperator
    from hedge.mesh import TAG_ALL, TAG_NONE
    op = StrongWaveOperator(-1, discr.dimensions, 
            source_vec_getter,
            dirichlet_tag=TAG_ALL,
            neumann_tag=TAG_NONE,
            radiation_tag=TAG_NONE,
            flux_type="upwind",
            )

    # cpu
    source_u_vec_cpu = cpu_discr.interpolate_volume_function(source_u)

    def source_vec_getter_cpu(t):
        from math import sin
        return source_u_vec_cpu*sin(10*t)

    from hedge.pde import StrongWaveOperator
    from hedge.mesh import TAG_ALL, TAG_NONE
    cpu_op = StrongWaveOperator(-1, discr.dimensions, 
            source_vec_getter_cpu,
            dirichlet_tag=TAG_ALL,
            neumann_tag=TAG_NONE,
            radiation_tag=TAG_NONE,
            flux_type="upwind",
            )
    # end cpu

    from hedge.tools import join_fields
    fields = join_fields(discr.volume_zeros(),
            [discr.volume_zeros() for i in range(discr.dimensions)])
    #fields = join_fields(
            #discr.interpolate_volume_function(lambda x: sin(x[0])),
            #[discr.volume_zeros() for i in range(discr.dimensions)]) # v

    dt = discr.dt_factor(op.max_eigenvalue())
    nsteps = int(10/dt)

    if rcon.is_head_rank:
        print "dt", dt
        print "nsteps", nsteps

    # diagnostics setup -------------------------------------------------------
    from pytools.log import LogManager, \
            add_general_quantities, \
            add_simulation_quantities, \
            add_run_info

    logmgr = LogManager("wave.dat", "w", rcon.communicator)
    add_run_info(logmgr)
    add_general_quantities(logmgr)
    add_simulation_quantities(logmgr, dt)
    discr.add_instrumentation(logmgr)

    from pytools.log import IntervalTimer
    vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
    logmgr.add_quantity(vis_timer)
    stepper.add_instrumentation(logmgr)

    #from hedge.log import Integral, LpNorm
    #u_getter = lambda: fields[0]
    #logmgr.add_quantity(LpNorm(u_getter, discr, 1, name="l1_u"))
    #logmgr.add_quantity(LpNorm(u_getter, discr, name="l2_u"))

    logmgr.add_watches(["step.max", "t_sim.max", "t_step.max", 
        ("t_compute", "t_diff.max+t_gather.max+t_lift.max+t_vector_math.max"),
        ("flops/s", "(n_flops_gather.sum+n_flops_lift.sum+n_flops_mass.sum+n_flops_diff.sum+n_flops_vector_math.sum)"
        "/(t_gather.max+t_lift.max+t_mass.max+t_diff.max+t_vector_math.max)")
        ])

    # timestep loop -----------------------------------------------------------
    rhs = op.bind(discr)
    cpu_rhs = cpu_op.bind(cpu_discr)
    from hedge.backends.cuda import make_block_visualization
    bvis = make_block_visualization(discr)

    for step in range(nsteps):
        logmgr.tick()

        t = step*dt

        if step % 3 == 0:
            visf = vis.make_file("fld-%04d" % step)

            cpu_fields = discr.convert_volume(fields, kind="numpy")
            true_rhs = cpu_rhs(t, cpu_fields)
            gpu_rhs = discr.convert_volume(rhs(t, fields), kind="numpy")

            from pylo import DB_VARTYPE_VECTOR
            vis.add_data(visf,
                    [
                        ("u", discr.convert_volume(fields[0], kind="numpy")),
                        ("v", discr.convert_volume(fields[1:], kind="numpy")), 
                        ("rhs_u", gpu_rhs[0]),
                        ("rhs_v", gpu_rhs[1:]), 
                        ("trhs_u", true_rhs[0]),
                        ("trhs_v", true_rhs[1:]), 
                        ("blockvis", bvis),
                    ],
                    expressions=[
                        ("rhsdiff_u", "rhs_u-trhs_u"),
                        ("rhsdiff_v", "rhs_v-trhs_v", DB_VARTYPE_VECTOR),
                        ],
                    time=t,
                    #scale_factor=2e1,
                    step=step)
            visf.close()

        fields = stepper(fields, t, dt, rhs)

    vis.close()

    logmgr.tick()
    logmgr.save()

    discr.close()

if __name__ == "__main__":
    #import cProfile as profile
    #profile.run("main()", "wave2d.prof")
    main()

