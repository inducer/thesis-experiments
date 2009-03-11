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
    from hedge.timestep import RK4TimeStepper
    from pytools.stopwatch import Job
    from math import sin, cos, pi, exp, sqrt

    from hedge.backends import guess_run_context
    rcon = guess_run_context(disable=set(["cuda"]))

    dim = 3

    if dim == 1:
        if rcon.is_head_rank:
            from hedge.mesh import make_uniform_1d_mesh
            mesh = make_uniform_1d_mesh(-10, 10, 500)
    elif dim == 2:
        from hedge.mesh import make_rect_mesh
        if rcon.is_head_rank:
            mesh = make_rect_mesh(a=(-0.5,-0.5),b=(0.5,0.5),max_area=0.008)
    elif dim == 3:
        if rcon.is_head_rank:
            from hedge.mesh import make_ball_mesh
            mesh = make_ball_mesh(max_volume=0.005)
    else:
        raise RuntimeError, "bad number of dimensions"

    if rcon.is_head_rank:
        print "%d elements" % len(mesh.elements)
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    discr = rcon.make_discretization(mesh_data, order=4)
    stepper = RK4TimeStepper()

    from hedge.visualization import SiloVisualizer, VtkVisualizer
    #vis = VtkVisualizer(discr, rcon, "fld")
    vis = SiloVisualizer(discr, rcon)

    def source_u(x, el):
        return exp(-numpy.dot(x, x)*128)

    source_u_vec = discr.interpolate_volume_function(source_u)

    def source_vec_getter(t):
        from math import sin
        if t < 1:
            return source_u_vec*sin(10*t)
        else:
            return 0*source_u_vec


    from hedge.pde import StrongWaveOperator
    from hedge.mesh import TAG_ALL, TAG_NONE
    op = StrongWaveOperator(-1, discr.dimensions, 
            source_vec_getter,
            dirichlet_tag=TAG_NONE,
            neumann_tag=TAG_NONE,
            radiation_tag=TAG_ALL,
            flux_type="upwind",
            )

    from hedge.tools import join_fields
    fields = join_fields(discr.volume_zeros(),
            [discr.volume_zeros() for i in range(discr.dimensions)])

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

    from hedge.log import Integral, LpNorm
    u_getter = lambda: fields[0]
    logmgr.add_quantity(LpNorm(u_getter, discr, 1, name="l1_u"))
    logmgr.add_quantity(LpNorm(u_getter, discr, name="l2_u"))

    logmgr.add_watches(["step.max", "t_sim.max", "l2_u", "t_step.max"])

    # timestep loop -----------------------------------------------------------
    rhs = op.bind(discr)
    for step in range(nsteps):
        logmgr.tick()

        t = step*dt

        if step % 10 == 0:
            visf = vis.make_file("fld-%04d" % step)

            vis.add_data(visf,
                    [
                        ("u", fields[0]),
                        ("v", fields[1:]), 
                    ],
                    time=t,
                    step=step)
            visf.close()

        fields = stepper(fields, t, dt, rhs)

    vis.close()

    logmgr.tick()
    logmgr.save()

if __name__ == "__main__":
    main()

