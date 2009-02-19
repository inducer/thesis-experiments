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
    from math import sin, cos, pi, sqrt
    from math import floor

    def f(x):
        return sin(pi*x)

    def u_analytic(x, el, t):
        return f((-numpy.dot(v, x)/norm_v+t*norm_v))

    def boundary_tagger(vertices, el, face_nr):
        if numpy.dot(el.face_normals[face_nr], v) < 0:
            return ["inflow"]
        else:
            return ["outflow"]

    dim = 1

    if dim == 1:
        transition_point = 10
        v = numpy.array([1])

        def el_tagger(el, all_vertices):
            if el.centroid(all_vertices)[0] < transition_point:
                return ["small"]
            else:
                return ["large"]

        from hedge.mesh import make_1d_mesh
        eps = 1e-5

        mesh = make_1d_mesh(
                numpy.hstack((
                    numpy.arange(0, transition_point, 0.1),
                    numpy.arange(transition_point, 20+eps, 0.2),
                    )), 
                periodic=True,
                element_tagger=el_tagger)
    elif dim == 2:
        v = numpy.array([2,0])
        from hedge.mesh import make_disk_mesh
        mesh = make_disk_mesh(boundary_tagger=boundary_tagger)
    elif dim == 3:
        v = numpy.array([0,0,1])
        from hedge.mesh import make_cylinder_mesh, make_ball_mesh, make_box_mesh

        mesh = make_cylinder_mesh(max_volume=0.04, height=2, 
                boundary_tagger=boundary_tagger,
                periodic=False, radial_subdivisions=32)
    else:
        raise RuntimeError, "bad number of dimensions"

    norm_v = la.norm(v)

    from hedge.partition import partition_from_tags, partition_mesh
    small_part, large_part = partition_mesh(mesh, 
            partition_from_tags(mesh, {"large": 1}),
            part_bdry_tag_factory=lambda opp_part:
            "from_large" if opp_part == 1 else "from_small")

    from hedge.backends.jit import Discretization
    order = 4
    small_discr = Discretization(small_part.mesh, order=order, debug=["node_permutation"])
    large_discr = Discretization(large_part.mesh, order=order, debug=["node_permutation"])
    whole_discr = Discretization(mesh, order=order)

    from hedge.visualization import VtkVisualizer, SiloVisualizer
    #vis = VtkVisualizer(vis_discr, rcon, "fld")
    vis = SiloVisualizer(whole_discr)

    # operator setup ----------------------------------------------------------
    from hedge.data import \
            ConstantGivenFunction, \
            TimeConstantGivenFunction, \
            TimeDependentGivenFunction
    from hedge.pde import StrongAdvectionOperator, WeakAdvectionOperator
    op = WeakAdvectionOperator(v, 
            inflow_u=TimeDependentGivenFunction(u_analytic),
            flux_type="upwind")

    large_u = large_discr.interpolate_volume_function(lambda x, el: u_analytic(x, el, 0))
    small_u = small_discr.interpolate_volume_function(lambda x, el: u_analytic(x, el, 0))

    # timestep setup ----------------------------------------------------------

    large_dt = large_discr.dt_factor(op.max_eigenvalue())
    small_dt = small_discr.dt_factor(op.max_eigenvalue())
    nsteps = int(700/max(large_dt, small_dt))

    assert small_dt >= large_dt/2

    from hedge.timestep import TwoRateAdamsBashforthTimeStepper
    stepper = TwoRateAdamsBashforthTimeStepper(
            large_dt=large_dt*0.025, step_ratio=2, 
            order=1)
    print "large dt=%g, small_dt=%g, nsteps=%d" % (stepper.large_dt, stepper.small_dt, nsteps)

    # diagnostics setup -------------------------------------------------------
    from pytools.log import LogManager, \
            add_general_quantities, \
            add_simulation_quantities, \
            add_run_info

    logmgr = LogManager("advection.dat", "w")
    add_run_info(logmgr)
    add_general_quantities(logmgr)
    add_simulation_quantities(logmgr, stepper.large_dt)

    logmgr.add_watches(["step.max", "t_step.max"])

    # timestep loop -----------------------------------------------------------
    rhs = op.bind(whole_discr)
    small_rhs = op.bind(small_discr)
    large_rhs = op.bind(large_discr)

    large_to_small_rhs = op.bind_interdomain(
            small_discr, small_part,
            large_discr, large_part,
            )
    small_to_large_rhs = op.bind_interdomain(
            large_discr, large_part,
            small_discr, small_part,
            )

    def full_rhs_small(t, u_small, u_large): 
        return small_rhs(t, u_small)
    def full_rhs_l2s(t, u_small, u_large): 
        return large_to_small_rhs(t, u_small, u_large)
    def full_rhs_large(t, u_small, u_large): 
        return large_rhs(t, u_large)
    def full_rhs_s2l(t, u_small, u_large):
        return small_to_large_rhs(t, u_large, u_small)

    rhss = [full_rhs_small, full_rhs_l2s,
            full_rhs_s2l, full_rhs_large]

    from functools import partial
    from hedge.partition import reassemble_parts
    reassemble = partial(reassemble_parts,
            whole_discr, 
            [small_part, large_part],
            [small_discr, large_discr])

    from hedge.tools import make_obj_array
    u = [small_u, large_u]

    def reassembled_rhs(t, u):
        return reassemble([
            full_rhs_small(t, *u)+full_rhs_l2s(t, *u),
            full_rhs_s2l(t, *u)+full_rhs_large(t, *u),
            ])

    for step in xrange(nsteps):
        logmgr.tick()

        t = step*stepper.large_dt

        if step % 10 == 0:
            whole_u = reassemble(u)
            u_rhs_real = rhs(t, whole_u)

            visf = vis.make_file("fld-%04d" % step)
            u_rhs = reassembled_rhs(t, u)
            vis.add_data(visf, [ 
                ("u", whole_u), 
                ("u_rhs", u_rhs), 
                ("u_rhs_real", u_rhs_real), 
                ("rhsdiff", u_rhs_real-u_rhs), 
                ], 
                time=t, step=step)
            visf.close()

        u = stepper(u, t, rhss)

        if step % 100 == 0:
            print la.norm(u[0]), la.norm(u[1])

    vis.close()

    logmgr.tick()
    logmgr.save()


if __name__ == "__main__":
    main()
