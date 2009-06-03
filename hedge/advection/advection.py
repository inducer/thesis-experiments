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
    from hedge.tools import mem_checkpoint
    from math import sin, cos, pi, sqrt
    from math import floor

    from hedge.backends import guess_run_context
    rcon = guess_run_context(disable=set(["cuda"]))

    def f(x):
        return sin(pi*x)

    def u_analytic(x, el, t):
        return f((-numpy.dot(v, x)/norm_v+t*norm_v))

    def boundary_tagger(vertices, el, face_nr, all_v):
        if numpy.dot(el.face_normals[face_nr], v) < 0:
            return ["inflow"]
        else:
            return ["outflow"]

    dim = 2

    if dim == 1:
        v = numpy.array([1])
        if rcon.is_head_rank:
            from hedge.mesh import make_uniform_1d_mesh
            mesh = make_uniform_1d_mesh(0, 2, 10, periodic=True)
    elif dim == 2:
        v = numpy.array([2,0])
        if rcon.is_head_rank:
            from hedge.mesh import make_disk_mesh
            mesh = make_disk_mesh(boundary_tagger=boundary_tagger)
    elif dim == 3:
        v = numpy.array([0,0,1])
        if rcon.is_head_rank:
            from hedge.mesh import make_cylinder_mesh, make_ball_mesh, make_box_mesh

            mesh = make_cylinder_mesh(max_volume=0.04, height=2, boundary_tagger=boundary_tagger,
                    periodic=False, radial_subdivisions=32)
    else:
        raise RuntimeError, "bad number of dimensions"

    norm_v = la.norm(v)

    if rcon.is_head_rank:
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    if dim != 1:
        mesh_data = mesh_data.reordered_by("cuthill")

    discr = rcon.make_discretization(mesh_data, order=4)
    vis_discr = discr

    from hedge.visualization import VtkVisualizer, SiloVisualizer
    #vis = VtkVisualizer(vis_discr, rcon, "fld")
    vis = SiloVisualizer(vis_discr, rcon)

    # operator setup ----------------------------------------------------------
    from hedge.data import \
            ConstantGivenFunction, \
            TimeConstantGivenFunction, \
            TimeDependentGivenFunction
    from hedge.pde import StrongAdvectionOperator, WeakAdvectionOperator
    op = WeakAdvectionOperator(v, 
            inflow_u=TimeDependentGivenFunction(u_analytic),
            flux_type="upwind")

    u = discr.interpolate_volume_function(lambda x, el: u_analytic(x, el, 0))

    # timestep setup ----------------------------------------------------------
    from hedge.timestep import RK4TimeStepper
    stepper = RK4TimeStepper()

    dt = discr.dt_factor(op.max_eigenvalue())
    nsteps = int(700/dt)

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

    stepper.add_instrumentation(logmgr)

    from hedge.log import Integral, LpNorm
    u_getter = lambda: u
    logmgr.add_quantity(Integral(u_getter, discr, name="int_u"))
    logmgr.add_quantity(LpNorm(u_getter, discr, p=1, name="l1_u"))
    logmgr.add_quantity(LpNorm(u_getter, discr, name="l2_u"))

    logmgr.add_watches(["step.max", "t_sim.max", "l2_u", "t_step.max"])

    # timestep loop -----------------------------------------------------------
    rhs = op.bind(discr)
    def timestep_loop(u):
        for step in xrange(nsteps):
            logmgr.tick()

            t = step*dt

            if step % 5 == 0:
                visf = vis.make_file("fld-%04d" % step)
                vis.add_data(visf, [ ("u", u), ], 
                            time=t, 
                            step=step
                            )
                visf.close()

            u = stepper(u, t, dt, rhs)

    if True:
        from cProfile import Profile
        from lsprofcalltree import KCacheGrind
        prof = Profile()

        rhs(0, u) # keep init traffic out of profile

        try:
            prof.runcall(lambda: timestep_loop(u))
        finally:
            kg = KCacheGrind(prof)
            import sys
            from hedge.tools import get_rank
            kg.output(open(
                "profile-%s-rank-%d.log" % (sys.argv[0], get_rank(discr)),
                "w"))
    else:
        timestep_loop(u)

    vis.close()

    logmgr.tick()
    logmgr.save()

    discr.close()


if __name__ == "__main__":
    main()
