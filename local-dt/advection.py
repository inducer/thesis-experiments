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
import matplotlib.pyplot as mpl




def build_matrix(f, in_n):
    in_vec = numpy.zeros(in_n, dtype=numpy.float64)
    out_n = len(f(in_vec))
    result = numpy.empty((out_n, in_n), dtype=numpy.float64)

    for i in xrange(in_n):
        in_vec[i-1] = 0
        in_vec[i] = 1
        result[:,i] = f(in_vec)

    return result




def plot_eigenvalues(m, dt, stepper_maker):
    evalues, evectors = la.eig(m)
    mpl.scatter(evalues.real*dt, evalues.imag*dt)
    mpl.xlabel(r"$\Delta t\, \mathrm{Re}\, \lambda$")
    mpl.ylabel(r"$\Delta t\, \mathrm{Im}\, \lambda$")
    mpl.grid()
    from stability import plot_stability_region
    from hedge.timestep import RK4TimeStepper
    plot_stability_region(RK4TimeStepper, alpha=0.1)
    mpl.show()




class LocalDtTestRig:
    # setup -------------------------------------------------------------------
    def __init__(self):
        self.make_mesh_1d()
        self.make_operator()
        self.make_discretization()
        self.make_rhss()

    def make_mesh_1d(self):
        transition_point = 1

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

        self.points = numpy.hstack((
                    numpy.arange(0, transition_point, 0.1),
                    numpy.arange(transition_point, 2+eps, 0.2),
                    ))

        from hedge.mesh import make_1d_mesh
        self.mesh = make_1d_mesh(self.points, periodic=True,
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

        self.whole_dt = self.whole_discr.dt_factor(self.op.max_eigenvalue())
        self.large_dt = self.large_discr.dt_factor(self.op.max_eigenvalue())
        self.small_dt = self.small_discr.dt_factor(self.op.max_eigenvalue())

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
            return (small_rhs(t, u_small)
                    + large_to_small_rhs(t, u_small, 0))
        def full_rhs_l2s(t, u_small, u_large): 
            return large_to_small_rhs(t, 0, u_large)
        def full_rhs_large(t, u_small, u_large): 
            return (large_rhs(t, u_large)
                    + small_to_large_rhs(t, u_large, 0))
        def full_rhs_s2l(t, u_small, u_large):
            return small_to_large_rhs(t, 0, u_small)

        self.rhss = [full_rhs_small, full_rhs_l2s,
                full_rhs_s2l, full_rhs_large]
        def reassembled_rhs(t, u):
            return self.reassemble([
                full_rhs_small(t, *u)+full_rhs_l2s(t, *u),
                full_rhs_s2l(t, *u)+full_rhs_large(t, *u),
                ])

        self.reassembled_rhs = reassembled_rhs

    # operations --------------------------------------------------------------

    def do_timestep(self):
        assert self.small_dt >= self.large_dt/2

        from hedge.timestep import TwoRateAdamsBashforthTimeStepper
        stepper = TwoRateAdamsBashforthTimeStepper(
                large_dt=0.05*self.large_dt, step_ratio=2, 
                order=1)

        dt = stepper.large_dt
        nsteps = int(700/dt)

        print "large dt=%g, small_dt=%g, nsteps=%d" % (
                stepper.large_dt, stepper.small_dt, nsteps)

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

            u = stepper(u, t, self.rhss)

    def build_part_dg_matrices(self):
        small_n = len(self.small_discr)
        large_n = len(self.large_discr)

        matrices = numpy.zeros((2,2), dtype=object)

        small_0 = self.small_discr.volume_zeros()
        large_0 = self.large_discr.volume_zeros()

        matrices[0,0] = build_matrix(
                lambda y: self.rhss[0](0, y, large_0), 
                small_n)
        matrices[0,1] = build_matrix(
                lambda y: self.rhss[1](0, small_0, y), 
                large_n)
        matrices[1,0] = build_matrix(
                lambda y: self.rhss[2](0, y, large_0), 
                small_n)
        matrices[1,1] = build_matrix(
                lambda y: self.rhss[3](0, small_0, y), 
                large_n)
        return matrices

    def build_whole_dg_matrix(self):
        return build_matrix(lambda y: self.rhs(0, y), len(self.whole_discr))

    def build_part_dg_matrix(self):
        m = self.build_part_dg_matrices()
        return numpy.vstack([numpy.hstack(m[i]) for i in range(m.shape[0])])

    def vis_matrix(self, m):
        n = m.shape[0]

        def get_Np(discr):
            eg, = discr.element_groups
            return eg.local_discretization.node_count()

        Np = get_Np(self.small_discr)

        for i in range(0, m.shape[0], get_Np(self.small_discr)):
            kwargs = dict(color="black", dashes=(3,3))
            if i == len(self.small_discr):
                kwargs["linewidth"] = 2

            mpl.axhline(y=i-0.5, **kwargs)
            mpl.axvline(x=i-0.5, **kwargs)

        mpl.imshow(m, interpolation="nearest")
        mpl.colorbar()

    def visualize_part_dg_matrix(self):
        self.vis_matrix(self.build_part_dg_matrix())
        mpl.show()

    def visualize_whole_dg_matrix(self):
        self.vis_matrix(self.build_whole_dg_matrix())
        mpl.show()

    def visualize_diff_dg_matrix(self):
        mpl.subplot(131)
        mpl.title("Diff")
        self.vis_matrix(
                self.build_whole_dg_matrix()
                -self.build_part_dg_matrix())
        mpl.subplot(132)
        mpl.title("Part")
        self.vis_matrix(self.build_part_dg_matrix())
        mpl.subplot(133)
        mpl.title("Whole")
        self.vis_matrix(self.build_whole_dg_matrix())
        mpl.show()

    def plot_whole_dg_eigenvalues(self):
        from hedge.timestep import RK4TimeStepper
        plot_eigenvalues(
                self.build_whole_dg_matrix(),
                self.whole_dt,
                RK4TimeStepper)

    def plot_part_dg_eigenvalues(self):
        from hedge.timestep import RK4TimeStepper
        plot_eigenvalues(
                self.build_part_dg_matrix(),
                self.small_dt,
                RK4TimeStepper)

    def plot_eigenmodes(self):
        m = self.build_part_dg_matrix()

        evalues, evectors = la.eig(m)

        evalue_and_index = list(enumerate(evalues))
        evalue_and_index.sort(key=lambda (i, ev): -ev.real)

        mpl.scatter(self.points, 0*self.points, label="Element Boundaries")

        for i, evalue in evalue_and_index[:3]:
            mpl.plot(
                    self.whole_discr.nodes[:,0], 
                    evectors[:,i], 
                    label=str(evalue),
                    )
        mpl.grid()
        from matplotlib.font_manager import FontProperties
        mpl.legend(loc="best", prop=FontProperties(size=8))
        mpl.show()

    def plot_eigenmodes(self):
        m = self.build_part_dg_matrix()

        evalues, evectors = la.eig(m)

        evalue_and_index = list(enumerate(evalues))
        evalue_and_index.sort(key=lambda (i, ev): -ev.real)

        mpl.scatter(self.points, 0*self.points, label="Element Boundaries")

        for i, evalue in evalue_and_index[:3]:
            mpl.plot(
                    self.whole_discr.nodes[:,0], 
                    evectors[:,i], 
                    label=str(evalue),
                    )
        mpl.grid()
        from matplotlib.font_manager import FontProperties
        mpl.legend(loc="best", prop=FontProperties(size=8))
        mpl.show()






def main() :
    tester = LocalDtTestRig()
    #tester.do_timestep()
    #tester.visualize_part_dg_matrix()
    #tester.visualize_diff_dg_matrix()
    #tester.plot_part_dg_eigenvalues()
    #tester.plot_eigenmodes()
    tester.do_timestep()




if __name__ == "__main__":
    main()
