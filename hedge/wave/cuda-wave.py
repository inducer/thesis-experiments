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
    from pytools.stopwatch import Job
    from math import sin, cos, pi, exp, sqrt
    from hedge.backends import guess_run_context

    rcon = guess_run_context()

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
            mesh = make_disk_mesh(max_area=5e-3)
            #mesh = make_regular_square_mesh(
                    #n=9, periodicity=(True,True))
            #mesh = make_rect_mesh(a=(-0.5,-0.5),b=(3.5,0.5),max_area=1e-2)
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

    def source_vec_getter(t):
        from math import sin
        return source_u_vec*sin(10*t)

    from hedge.pde import StrongWaveOperator
    from hedge.mesh import TAG_ALL, TAG_NONE
    op = StrongWaveOperator(-1, dim, 
            source_vec_getter,
            dirichlet_tag=TAG_ALL,
            neumann_tag=TAG_NONE,
            radiation_tag=TAG_NONE,
            flux_type="upwind",
            )

    discr = rcon.make_discretization(mesh_data, order=4,
            debug=[
                "cuda_no_plan",
                "cuda_dumpkernels",
                #"cuda_flux",
                #"cuda_debugbuf",
                ],
            tune_for=op.op_template()
            )

    from hedge.timestep import RK4TimeStepper
    stepper = RK4TimeStepper()

    def source_u(x, el):
        return exp(-numpy.dot(x, x)*128)

    source_u_vec = discr.interpolate_volume_function(source_u)

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

    from hedge.visualization import SiloVisualizer, VtkVisualizer
    #vis = VtkVisualizer(discr, rcon, "fld")
    vis = SiloVisualizer(discr, rcon)

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

    for step in range(nsteps):
        logmgr.tick()

        t = step*dt

        if step % 10 == 0:
            visf = vis.make_file("fld-%04d" % step)

            my_rhs = rhs(t, fields)

            from pylo import DB_VARTYPE_VECTOR
            vis.add_data(visf,
                    [
                        ("u", discr.convert_volume(fields[0], kind="numpy")),
                        ("v", discr.convert_volume(fields[1:], kind="numpy")), 
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

