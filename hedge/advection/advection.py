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
    from hedge.timestep import RK4TimeStepper
    from hedge.visualization import SiloVisualizer, VtkVisualizer
    from hedge.tools import mem_checkpoint
    from pytools.stopwatch import Job
    from math import sin, cos, pi, sqrt
    from hedge.parallel import \
            guess_parallelization_context, \
            reassemble_volume_field
    from math import floor

    def f(x):
        return sin(pi*x)
        #if int(floor(x)) % 2 == 0:
            #return 1
        #else:
            #return 0

    def u_analytic(x, t):
        return f((-v*x/norm_v+t*norm_v))

    def boundary_tagger(vertices, el, face_nr):
        if numpy.dot(el.face_normals[face_nr], v) < 0:
            return ["inflow"]
        else:
            return ["outflow"]

    pcon = guess_parallelization_context()

    dim = 2

    if pcon.is_head_rank:
        job = Job("mesh")

    if dim == 1:
        v = numpy.array([1])
        if pcon.is_head_rank:
            from hedge.mesh import make_uniform_1d_mesh
            mesh = make_uniform_1d_mesh(-2, 5, 10, periodic=True)
    elif dim == 2:
        v = numpy.array([2,0])
        if pcon.is_head_rank:
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
        if pcon.is_head_rank:
            from hedge.mesh import make_cylinder_mesh, make_ball_mesh, make_box_mesh

            mesh = make_cylinder_mesh(max_volume=0.004, boundary_tagger=boundary_tagger,
                    periodic=False, radial_subdivisions=32)
            #mesh = make_box_mesh(dimensions=(1,1,2*pi/3), max_volume=0.01,
                    #boundary_tagger=boundary_tagger)
            #mesh = make_box_mesh(max_volume=0.01, boundary_tagger=boundary_tagger)
            #mesh = make_ball_mesh(boundary_tagger=boundary_tagger)
            #mesh = make_cylinder_mesh(max_volume=0.01, boundary_tagger=boundary_tagger)
    else:
        raise RuntimeError, "bad number of dimensions"

    norm_v = la.norm(v)

    if pcon.is_head_rank:
        mesh_data = pcon.distribute_mesh(mesh)
    else:
        mesh_data = pcon.receive_mesh()
    if pcon.is_head_rank:
        job.done()

    if pcon.is_head_rank:
        job = Job("discretization")
    #mesh_data = mesh_data.reordered_by("cuthill")
    discr = pcon.make_discretization(mesh_data, order=6)
    vis_discr = discr
    if pcon.is_head_rank:
        job.done()

    vis = SiloVisualizer(vis_discr, pcon)
    #vis = VtkVisualizer(vis_discr, pcon, "fld")

    # operator setup ----------------------------------------------------------
    from hedge.data import \
            ConstantGivenFunction, \
            TimeConstantGivenFunction, \
            TimeDependentGivenFunction
    from hedge.pde import StrongAdvectionOperator
    op = StrongAdvectionOperator(discr, v, 
            inflow_u=TimeConstantGivenFunction(ConstantGivenFunction()),
            #inflow_u=TimeDependentGivenFunction(u_analytic)),
            flux_type="upwind")

    #from pyrticle._internal import ShapeFunction
    #sf = ShapeFunction(1, 2, alpha=1)

    def gauss_hump(x, el):
        from math import exp
        rsquared = numpy.dot(x, x)/(0.3**2)
        return exp(-rsquared)
    def gauss2_hump(x, el):
        from math import exp
        rsquared = (x*x)/(0.1**2)
        return exp(-rsquared)-0.5*exp(-rsquared/2)

    hump_width = 2
    def c_inf_hump(x, el):
        if abs(x) > hump_width:
            return 0
        else:
            exp(-1/(x-hump_width)**2)* exp(-1/(x+hump_width)**2)

    #u = discr.interpolate_volume_function(lambda x: u_analytic(x, 0))
    u = discr.interpolate_volume_function(gauss_hump)

    # timestep setup ----------------------------------------------------------
    stepper = RK4TimeStepper()

    dt = discr.dt_factor(op.max_eigenvalue())
    nsteps = int(700/dt)

    if pcon.is_head_rank:
        print "%d elements, dt=%g, nsteps=%d" % (
                len(discr.mesh.elements),
                dt,
                nsteps)

    # diagnostics setup -------------------------------------------------------
    from pytools.log import LogManager, \
            add_general_quantities, \
            add_simulation_quantities, \
            add_run_info

    logmgr = LogManager("advection.dat", "w", pcon.communicator)
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
    logmgr.add_quantity(Integral(u_getter, discr, name="int_u"))
    logmgr.add_quantity(LpNorm(u_getter, discr, p=1, name="l1_u"))
    logmgr.add_quantity(LpNorm(u_getter, discr, name="l2_u"))

    logmgr.add_watches(["step.max", "t_sim.max", "l2_u", "t_step.max"])

    # timestep loop -----------------------------------------------------------
    def logmap(x, low_exp=15):
        return 0.1*numpy.log10(numpy.abs(x)+1e-15)

    from hedge.discretization import Filter, ExponentialFilterResponseFunction
    filter = Filter(discr, ExponentialFilterResponseFunction(0.97, 3))

    counter = [0]

    for step in xrange(nsteps):
        logmgr.tick()

        t = step*dt

        if False: #step % 1000 == 0:
            vis_timer.start()
            visf = vis.make_file("fld-%04d" % step)
            vis.add_data(visf, [
                        ("u", u), 
                        ("logu", logmap(u)), 
                        ], 
                        #expressions=[("error", "u-u_true")],
                        time=t, 
                        step=step
                        )
            visf.close()
            vis_timer.stop()


        def rhs(t, u):
            vis_timer.start()
            visf = vis.make_file("fld-%04d" % counter[0])
            counter[0] += 1
            vis.add_data(visf, [
                        ("u", u), 
                        ("logu", logmap(u)), 
                        ], 
                        #expressions=[("error", "u-u_true")],
                        time=t, 
                        step=step
                        )
            visf.close()
            vis_timer.stop()
            return op.rhs(t, u)
        #u = filter(stepper(u, t, dt, op.rhs))
        u = stepper(u, t, dt, rhs)
        #if step % 1 == 0:
            #u = filter(u)

        #u_true = discr.interpolate_volume_function(
                #lambda x: u_analytic(t, x))

    vis.close()

    logmgr.tick()
    logmgr.save()


if __name__ == "__main__":
    #import cProfile as profile
    #profile.run("main()", "advec.prof")
    main()


