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
from hedge.mesh import TAG_ALL, TAG_NONE




def main(write_output=True, 
        dir_tag=TAG_NONE, neu_tag=TAG_NONE, rad_tag=TAG_ALL, 
        flux_type_arg="upwind", dtype=numpy.float64, debug=[]):
    from hedge.timestep import RK4TimeStepper
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
        from hedge.mesh import make_rect_mesh, make_disk_mesh
        if rcon.is_head_rank:
            #mesh = make_rect_mesh(a=(-0.5,-0.5),b=(0.5,0.5),max_area=0.002)
            mesh = make_disk_mesh(r=0.5, max_area=0.002)
    elif dim == 3:
        if rcon.is_head_rank:
            from hedge.mesh import make_ball_mesh
            mesh = make_ball_mesh(max_volume=0.0005)
    else:
        raise RuntimeError, "bad number of dimensions"

    if rcon.is_head_rank:
        print "%d elements" % len(mesh.elements)
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    discr = rcon.make_discretization(mesh_data, order=4, debug=debug)
    stepper = RK4TimeStepper()

    from hedge.visualization import VtkVisualizer, SiloVisualizer
    if write_output:
        #vis = VtkVisualizer(discr, rcon, "fld")
        vis = SiloVisualizer(discr, rcon)

    def source_u(x, el):
        return exp(-numpy.dot(x, x)*128)

    from hedge.models.wave import StrongWaveOperator
    from hedge.mesh import TAG_ALL, TAG_NONE
    from hedge.data import \
            make_tdep_given, \
            TimeHarmonicGivenFunction, \
            TimeIntervalGivenFunction

    op = StrongWaveOperator(-1, discr.dimensions, 
            source_f=TimeIntervalGivenFunction(
                TimeHarmonicGivenFunction(
                    make_tdep_given(source_u), omega=10),
                2, 3),
            dirichlet_tag=TAG_ALL,
            dirichlet_bc_f=make_tdep_given(
                lambda x, el: 0 if x[0]> 0 else 1),
            neumann_tag=TAG_NONE,
            radiation_tag=TAG_NONE,
            flux_type=flux_type_arg
            )

    from hedge.tools import join_fields
    fields = join_fields(discr.volume_zeros(dtype=dtype),
            [discr.volume_zeros(dtype=dtype) for i in range(discr.dimensions)])

    # diagnostics setup -------------------------------------------------------
    from pytools.log import LogManager, \
            add_general_quantities, \
            add_simulation_quantities, \
            add_run_info

    if write_output:
        log_file_name = "wave.dat"
    else:
        log_file_name = None

    logmgr = LogManager(log_file_name, "w", rcon.communicator)
    add_run_info(logmgr)
    add_general_quantities(logmgr)
    add_simulation_quantities(logmgr)
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
    try:
        from hedge.timestep import times_and_steps
        step_it = times_and_steps(
                final_time=20, logmgr=logmgr,
                max_dt_getter=lambda t: op.estimate_timestep(discr,
                    stepper=stepper, t=t, fields=fields))

        for step, t, dt in step_it:
            if step % 10 == 0 and write_output:
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

        assert fields[0].dtype == dtype

    finally:
        if write_output:
            vis.close()

        logmgr.close()
        discr.close()

if __name__ == "__main__":
    main()




# entry points for py.test ----------------------------------------------------
def test_wave():
    from pytools.test import mark_test
    mark_long = mark_test.long

    yield ("dirichlet wave equation with SP data", mark_long(main),
            False, TAG_ALL, TAG_NONE, TAG_NONE, "upwind", numpy.float64)
    yield ("dirichlet wave equation with SP complex data", mark_long(main),
            False, TAG_ALL, TAG_NONE, TAG_NONE, "upwind", numpy.complex64)
    yield ("dirichlet wave equation with DP complex data", mark_long(main),
            False, TAG_ALL, TAG_NONE, TAG_NONE, "upwind", numpy.complex128)
    for flux_type in ["upwind", "central"]:
        yield ("dirichlet wave equation with %s flux" % flux_type,
                mark_long(main),
                False, TAG_ALL, TAG_NONE, TAG_NONE, flux_type)
    yield ("neumann wave equation", mark_long(main),
            False, TAG_NONE, TAG_ALL, TAG_NONE)
    yield ("radiation-bc wave equation", mark_long(main),
            False, TAG_NONE, TAG_NONE, TAG_ALL)
