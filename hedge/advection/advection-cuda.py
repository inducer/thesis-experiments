#! /usr/bin/env python
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
from pytools import memoize_method




def main() :
    from math import sin, cos, pi, sqrt
    from math import floor

    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("--single", action="store_true")
    parser.add_option("--order", default=4, type="int")
    parser.add_option("--max-volume", default=0.08, type="float")
    parser.add_option("--final-time", default=700, type="float")
    parser.add_option("--vis-interval", default=0, type="int")
    #parser.add_option("--steps", type="int")
    #parser.add_option("--cpu", action="store_true")
    parser.add_option("-d", "--debug-flags", metavar="DEBUG_FLAG,DEBUG_FLAG")
    options, args = parser.parse_args()
    assert not args

    def f(x):
        return sin(pi*x)
        #if int(floor(x)) % 2 == 0:
            #return 1
        #else:
            #return 0

    def u_analytic(x, t):
        return f((-v*x/norm_v+t*norm_v))

    def boundary_tagger(vertices, el, face_nr, points):
        if numpy.dot(el.face_normals[face_nr], v) < 0:
            return ["inflow"]
        else:
            return ["outflow"]

    from hedge.backends import guess_run_context
    rcon = guess_run_context()

    dim = 3

    if dim == 1:
        v = numpy.array([1])
        if rcon.is_head_rank:
            from hedge.mesh import make_uniform_1d_mesh
            mesh = make_uniform_1d_mesh(-2, 5, 10, periodic=True)
    elif dim == 2:
        v = numpy.array([2,0])
        if rcon.is_head_rank:
            from hedge.mesh import \
                    make_disk_mesh, \
                    make_square_mesh, \
                    make_rect_mesh, \
                    make_regular_square_mesh, \
                    make_regular_rect_mesh, \
                    make_single_element_mesh
        
            #mesh = make_square_mesh(max_area=0.0003, boundary_tagger=boundary_tagger)
            #mesh = make_regular_square_mesh(a=-r, b=r, boundary_tagger=boundary_tagger, n=3)
            #mesh = make_single_element_mesh(boundary_tagger=boundary_tagger)
            #mesh = make_disk_mesh(r=pi, boundary_tagger=boundary_tagger, max_area=0.5)
            #mesh = make_disk_mesh(boundary_tagger=boundary_tagger)
            
            if False:
                mesh = make_regular_rect_mesh(
                        (-0.5, -1.5),
                        (5, 1.5),
                        n=(10,5),
                        boundary_tagger=boundary_tagger,
                        periodicity=(True, False),
                        )
            if True:
                mesh = make_rect_mesh(
                        (-1, -1.5),
                        (5, 1.5),
                        max_area=0.3,
                        boundary_tagger=boundary_tagger,
                        periodicity=(True, False),
                        subdivisions=(10,5),
                        )
    elif dim == 3:
        v = numpy.array([0,0,0.3])
        if rcon.is_head_rank:
            from hedge.mesh import make_cylinder_mesh, make_ball_mesh, make_box_mesh

            #mesh = make_cylinder_mesh(max_volume=options.h, boundary_tagger=boundary_tagger,
                    #periodic=False, radial_subdivisions=32)
            mesh = make_box_mesh(a=(-1,-1,-1), b=(1,1,1), 
                    max_volume=options.max_volume,
                    boundary_tagger=boundary_tagger)
            #mesh = make_box_mesh(max_volume=0.01, boundary_tagger=boundary_tagger)
            #mesh = make_ball_mesh(boundary_tagger=boundary_tagger)
            #mesh = make_cylinder_mesh(max_volume=0.01, boundary_tagger=boundary_tagger)
    else:
        raise RuntimeError, "bad number of dimensions"

    norm_v = la.norm(v)

    if rcon.is_head_rank:
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    from hedge.data import \
            ConstantGivenFunction, \
            TimeConstantGivenFunction, \
            TimeDependentGivenFunction
    from hedge.pde import StrongAdvectionOperator
    op = StrongAdvectionOperator(v, 
            inflow_u=TimeConstantGivenFunction(ConstantGivenFunction()),
            #inflow_u=TimeDependentGivenFunction(u_analytic)),
            flux_type="upwind")

    debug_flags = [ ]
    if options.debug_flags:
        debug_flags.extend(options.debug_flags.split(","))

    #mesh_data = mesh_data.reordered_by("cuthill")
    discr = rcon.make_discretization(mesh_data, order=options.order, 
            tune_for=op.op_template(),
            debug=debug_flags,
            default_scalar_type=numpy.float32 if options.single else numpy.float64,
            )

    from hedge.visualization import SiloVisualizer, VtkVisualizer
    vis = SiloVisualizer(discr, rcon)

    # operator setup ----------------------------------------------------------

    #from pyrticle._internal import ShapeFunction
    #sf = ShapeFunction(1, 2, alpha=1)

    def gauss_hump(x, el):
        from math import exp, sin
        rsquared = numpy.dot(x, x)/(0.3**2)
        return exp(-rsquared)-2

    def gauss2_hump(x, el):
        from math import exp
        rsquared = (x*x)/(0.1**2)
        return exp(-rsquared)-0.5*exp(-rsquared/2)

    def wild_trig(x, el):
        from math import sin, cos
        return sin(17*x[0])*cos(22*x[1])*sin(15*x[2]) + sin(el.id)

    hump_width = 2
    def c_inf_hump(x, el):
        if abs(x) > hump_width:
            return 0
        else:
            exp(-1/(x-hump_width)**2)* exp(-1/(x+hump_width)**2)

    #u = discr.interpolate_volume_function(lambda x: u_analytic(x, 0))
    u = discr.interpolate_volume_function(gauss_hump)
    #u = discr.interpolate_volume_function(wild_trig)

    # timestep setup ----------------------------------------------------------
    from hedge.backends.cuda.tools import RK4TimeStepper
    #from hedge.timestep import RK4TimeStepper
    stepper = RK4TimeStepper()

    dt = discr.dt_factor(op.max_eigenvalue())
    nsteps = int(options.final_time/dt)

    if rcon.is_head_rank:
        print "%d elements, dt=%g, nsteps=%d" % (
                len(discr.mesh.elements),
                dt,
                nsteps)

    # diagnostics setup -------------------------------------------------------
    from pytools.log import LogManager, \
            add_general_quantities, \
            add_simulation_quantities, \
            add_run_info

    logmgr = LogManager("advection.dat", "w", rcon.communicator)
    add_run_info(logmgr)
    add_general_quantities(logmgr)
    add_simulation_quantities(logmgr, dt)
    discr.add_instrumentation(logmgr)

    from pytools.log import IntervalTimer
    vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
    logmgr.add_quantity(vis_timer)
    stepper.add_instrumentation(logmgr)

    from hedge.log import Integral, LpNorm
    u_getter = lambda: u
    #logmgr.add_quantity(Integral(u_getter, discr, name="int_u"))
    #logmgr.add_quantity(LpNorm(u_getter, discr, p=1, name="l1_u"))
    #logmgr.add_quantity(LpNorm(u_getter, discr, name="l2_u"))

    logmgr.add_watches(["step.max", "t_sim.max", "t_step.max", 
        ("t_compute", "t_diff.max+t_gather.max+t_el_local.max+t_rk4.max+t_vector_math.max"),
        ("flops/s", "(n_flops_gather.sum+n_flops_lift.sum+n_flops_mass.sum+n_flops_diff.sum+n_flops_vector_math.sum+n_flops_rk4.sum)"
        "/(t_gather.max+t_el_local.max+t_diff.max+t_vector_math.max+t_rk4.max)")
        ])

    # timestep loop -----------------------------------------------------------
    rhs = op.bind(discr)

    for step in xrange(nsteps):

        logmgr.tick()
        t = step*dt

        if options.vis_interval and step % options.vis_interval == 0:
            visf = vis.make_file("fld-%04d" % step)
            vis.add_data(visf, [
                        ("u", discr.convert_volume(u, kind="numpy")), 
                        #("u", u), 
                        #("u_true", u_true), 
                        ], 
                        #expressions=[("error", "u-u_true")]
                        time=t, 
                        step=step
                        )
            visf.close()

        u = stepper(u, t, dt, rhs)

        print discr.norm(u)

    vis.close()

    discr.close()




if __name__ == "__main__":
    #import cProfile as profile
    #profile.run("main()", "advec.prof")
    main()


