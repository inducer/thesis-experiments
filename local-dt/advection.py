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




class LocalDtTestRig:
    def __init__(self):
        self.make_mesh_1d()
        self.make_operator()
        self.make_discretization()
        self.make_rhss()
        self.make_timestepper()

    def make_mesh_1d(self):
        transition_point = 10

        def el_tagger(el, all_vertices):
            if el.centroid(all_vertices)[0] < transition_point:
                return ["small"]
            else:
                return ["large"]

        def boundary_tagger(vertices, el, face_nr):
            if numpy.dot(el.face_normals[face_nr], v) < 0:
                return ["inflow"]
            else:
                return ["outflow"]

        eps = 1e-5

        from hedge.mesh import make_1d_mesh
        self.mesh = make_1d_mesh(
                numpy.hstack((
                    numpy.arange(0, transition_point, 0.1),
                    numpy.arange(transition_point, 20+eps, 0.2),
                    )), 
                periodic=True,
                element_tagger=el_tagger)

        from hedge.partition import partition_from_tags, partition_mesh
        self.small_part, self.large_part = partition_mesh(self.mesh, 
                partition_from_tags(self.mesh, {"large": 1}),
                part_bdry_tag_factory=lambda opp_part:
                "from_large" if opp_part == 1 else "from_small")

    def make_operator(self):
        self.v = numpy.array([1])
        self.norm_v = la.norm(self.v)

        from hedge.data import \
                ConstantGivenFunction, \
                TimeConstantGivenFunction, \
                TimeDependentGivenFunction
        from hedge.pde import StrongAdvectionOperator, WeakAdvectionOperator
        self.op = WeakAdvectionOperator(self.v, 
                flux_type="upwind")

    def make_discretization(self, order=4):
        from hedge.backends.jit import Discretization
        self.small_discr = Discretization(self.small_part.mesh, 
                order=order, debug=["node_permutation"])
        self.large_discr = Discretization(self.large_part.mesh, 
                order=order, debug=["node_permutation"])
        self.whole_discr = Discretization(self.mesh, order=order)

        from functools import partial
        from hedge.partition import reassemble_parts
        self.reassemble = partial(reassemble_parts,
                self.whole_discr, 
                [self.small_part, self.large_part],
                [self.small_discr, self.large_discr])

        from hedge.visualization import SiloVisualizer
        self.vis = SiloVisualizer(self.whole_discr)

    def make_timestepper(self):
        large_dt = self.large_discr.dt_factor(self.op.max_eigenvalue())
        small_dt = self.small_discr.dt_factor(self.op.max_eigenvalue())

        assert small_dt >= large_dt/2

        from hedge.timestep import TwoRateAdamsBashforthTimeStepper
        self.stepper = TwoRateAdamsBashforthTimeStepper(
                large_dt=large_dt*0.025, step_ratio=2, 
                order=1)

    def make_rhss(self):
        self.rhs = self.op.bind(self.whole_discr)
        small_rhs = self.op.bind(self.small_discr)
        large_rhs = self.op.bind(self.large_discr)

        large_to_small_rhs = self.op.bind_interdomain(
                self.small_discr, self.small_part,
                self.large_discr, self.large_part,
                )
        small_to_large_rhs = self.op.bind_interdomain(
                self.large_discr, self.large_part,
                self.small_discr, self.small_part,
                )

        def full_rhs_small(t, u_small, u_large): 
            return small_rhs(t, u_small)
        def full_rhs_l2s(t, u_small, u_large): 
            return large_to_small_rhs(t, u_small, u_large)
        def full_rhs_large(t, u_small, u_large): 
            return large_rhs(t, u_large)
        def full_rhs_s2l(t, u_small, u_large):
            return small_to_large_rhs(t, u_large, u_small)

        self.rhss = [full_rhs_small, full_rhs_l2s,
                full_rhs_s2l, full_rhs_large]

        def reassembled_rhs(t, u):
            return self.reassemble([
                full_rhs_small(t, *u)+full_rhs_l2s(t, *u),
                full_rhs_s2l(t, *u)+full_rhs_large(t, *u),
                ])

        self.reassembled_rhs = reassembled_rhs

    def do_timestep(self):
        dt = self.stepper.large_dt
        nsteps = int(700/dt)

        print "large dt=%g, small_dt=%g, nsteps=%d" % (
                self.stepper.large_dt, self.stepper.small_dt, nsteps)

        from math import sin, cos, pi, sqrt

        def f(x):
            return sin(pi*x)

        def u_analytic(x, el, t):
            return f((-numpy.dot(self.v, x)/self.norm_v+t*self.norm_v))

        large_u = self.large_discr.interpolate_volume_function(
                lambda x, el: u_analytic(x, el, 0))
        small_u = self.small_discr.interpolate_volume_function(
                lambda x, el: u_analytic(x, el, 0))

        u = [small_u, large_u]

        for step in xrange(nsteps):
            if step % 100 == 0:
                print step, la.norm(u[0]), la.norm(u[1])

            t = step*dt

            if step % 10 == 0:
                whole_u = self.reassemble(u)
                u_rhs_real = self.rhs(t, whole_u)

                visf = self.vis.make_file("fld-%04d" % step)
                u_rhs = self.reassembled_rhs(t, u)
                self.vis.add_data(visf, [ 
                    ("u", whole_u), 
                    ("u_rhs", u_rhs), 
                    ("u_rhs_real", u_rhs_real), 
                    ("rhsdiff", u_rhs_real-u_rhs), 
                    ], 
                    time=t, step=step)
                visf.close()

            u = self.stepper(u, t, self.rhss)




def main() :
    tester = LocalDtTestRig()
    tester.do_timestep()




if __name__ == "__main__":
    main()
