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




# tools -----------------------------------------------------------------------
def build_matrix(f, in_n):
    in_vec = numpy.zeros(in_n, dtype=numpy.float64)
    out_n = len(f(in_vec))
    result = numpy.empty((out_n, in_n), dtype=numpy.float64)

    for i in xrange(in_n):
        in_vec[i-1] = 0
        in_vec[i] = 1
        result[:,i] = f(in_vec)

    return result




def plot_eigenvalues(m, **kwargs):
    evalues, evectors = la.eig(m)
    mpl.scatter(evalues.real, evalues.imag, **kwargs)
    mpl.xlabel(r"$\mathrm{Re}\, \lambda$")
    mpl.ylabel(r"$\mathrm{Im}\, \lambda$")
    mpl.grid()




def plot_eigenvalues_and_stab(m, stepper_maker):
    plot_eigenvalues(m)
    from stability import plot_stability_region
    plot_stability_region(stepper_maker, alpha=0.1)
    mpl.show()




# setup -----------------------------------------------------------------------
class LocalDtTestRig:
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
                    numpy.arange(0, transition_point, 0.05),
                    numpy.arange(transition_point, 2+eps, 0.1),
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




# diagnostics -----------------------------------------------------------------
def do_timestep(rig):
    assert rig.small_dt >= rig.large_dt/2

    from hedge.timestep import TwoRateAdamsBashforthTimeStepper
    stepper = TwoRateAdamsBashforthTimeStepper(
            large_dt=0.05*rig.large_dt, step_ratio=2, 
            order=1)

    dt = stepper.large_dt
    nsteps = int(700/dt)

    print "large dt=%g, small_dt=%g, nsteps=%d" % (
            stepper.large_dt, stepper.small_dt, nsteps)

    from math import sin, cos, pi, sqrt

    def f(x):
        return sin(pi*x)

    def u_analytic(x, el, t):
        return f((-numpy.dot(rig.v, x)/rig.norm_v+t*rig.norm_v))

    large_u = rig.large_discr.interpolate_volume_function(
            lambda x, el: u_analytic(x, el, 0))
    small_u = rig.small_discr.interpolate_volume_function(
            lambda x, el: u_analytic(x, el, 0))

    u = [small_u, large_u]

    for step in xrange(nsteps):
        if step % 100 == 0:
            print step, la.norm(u[0]), la.norm(u[1])

        t = step*dt

        if step % 10 == 0:
            whole_u = rig.reassemble(u)
            u_rhs_real = rig.rhs(t, whole_u)

            visf = rig.vis.make_file("fld-%04d" % step)
            u_rhs = rig.reassembled_rhs(t, u)
            rig.vis.add_data(visf, [ 
                ("u", whole_u), 
                ("u_rhs", u_rhs), 
                ("u_rhs_real", u_rhs_real), 
                ("rhsdiff", u_rhs_real-u_rhs), 
                ], 
                time=t, step=step)
            visf.close()

        u = stepper(u, t, rig.rhss)




def build_part_dg_matrices(rig):
    small_n = len(rig.small_discr)
    large_n = len(rig.large_discr)

    matrices = numpy.zeros((2,2), dtype=object)

    small_0 = rig.small_discr.volume_zeros()
    large_0 = rig.large_discr.volume_zeros()

    matrices[0,0] = build_matrix(
            lambda y: rig.rhss[0](0, y, large_0), 
            small_n)
    matrices[0,1] = build_matrix(
            lambda y: rig.rhss[1](0, small_0, y), 
            large_n)
    matrices[1,0] = build_matrix(
            lambda y: rig.rhss[2](0, y, large_0), 
            small_n)
    matrices[1,1] = build_matrix(
            lambda y: rig.rhss[3](0, small_0, y), 
            large_n)
    return matrices

def build_whole_dg_matrix(rig):
    return build_matrix(lambda y: rig.rhs(0, y), len(rig.whole_discr))

def build_part_dg_matrix(rig):
    m = build_part_dg_matrices(rig)
    return numpy.vstack([numpy.hstack(m[i]) for i in range(m.shape[0])])

def vis_matrix(rig, m):
    n = m.shape[0]

    def get_Np(discr):
        eg, = discr.element_groups
        return eg.local_discretization.node_count()

    Np = get_Np(rig.small_discr)

    for i in range(0, m.shape[0], get_Np(rig.small_discr)):
        kwargs = dict(color="black", dashes=(3,3))
        if i == len(rig.small_discr):
            kwargs["linewidth"] = 2

        mpl.axhline(y=i-0.5, **kwargs)
        mpl.axvline(x=i-0.5, **kwargs)

    mpl.imshow(m, interpolation="nearest")
    mpl.colorbar()

def visualize_part_dg_matrix(rig):
    vis_matrix(rig, build_part_dg_matrix(rig))
    mpl.show()

def visualize_whole_dg_matrix(rig):
    vis_matrix(rig, build_whole_dg_matrix(rig))
    mpl.show()

def visualize_diff_dg_matrix(rig):
    mpl.subplot(131)
    mpl.title("Diff")
    vis_matrix(rig,
            build_whole_dg_matrix(rig)-build_part_dg_matrix(rig))
    mpl.subplot(132)
    mpl.title("Part")
    vis_matrix(rig, build_part_dg_matrix(rig))
    mpl.subplot(133)
    mpl.title("Whole")
    vis_matrix(rig, build_whole_dg_matrix(rig))
    mpl.show()

def plot_whole_dg_eigenvalues(rig):
    from hedge.timestep import RK4TimeStepper
    plot_eigenvalues_and_stab(
            rig.whole_dt * build_whole_dg_matrix(rig),
            RK4TimeStepper)

def plot_part_dg_eigenvalues(rig):
    from hedge.timestep import RK4TimeStepper
    plot_eigenvalues_and_stab(
            rig.small_dt * build_part_dg_matrix(rig),
            RK4TimeStepper)

def plot_eigenmodes(rig):
    m = build_part_dg_matrix(rig)

    evalues, evectors = la.eig(m)

    evalue_and_index = list(enumerate(evalues))
    evalue_and_index.sort(key=lambda (i, ev): -ev.real)

    mpl.scatter(rig.points, 0*rig.points, label="Element Boundaries")

    for i, evalue in evalue_and_index[:3]:
        mpl.plot(
                rig.whole_discr.nodes[:,0], 
                evectors[:,i], 
                label=str(evalue),
                )
    mpl.grid()
    from matplotlib.font_manager import FontProperties
    mpl.legend(loc="best", prop=FontProperties(size=8))
    mpl.show()

class TwoRateOperator:
    def __init__(self, dt, rig, split_n):
        self.dt = dt
        self.rig = rig
        self.split_n = split_n

    def __call__(self, v):
        # We make no assumption on v. Therefore, we need to regenerate
        # the rhs history (stored in the stepper) on every invocation.
        # The easiest way of achieving that is to just construct a new
        # instance.

        from hedge.timestep import TwoRateAdamsBashforthTimeStepper
        stepper = TwoRateAdamsBashforthTimeStepper(
                large_dt=self.dt, step_ratio=2, order=1)

        s_result = stepper([
            v[:self.split_n], 
            v[self.split_n:]
            ], 0, self.rig.rhss)
        return numpy.hstack(s_result) - v

def draw_multirate_spectrum(rig):
    small_n = len(rig.small_discr)
    whole_n = len(rig.whole_discr)

    markers = ['s', 'o', '^', 'd', 'x']
    colors = ['r', 'g', 'b', 'cyan', 'magenta']
    for i, dt_fac in enumerate([0.4, 0.425, 0.45, 0.5]):
        evalues, evectors = la.eig(build_matrix(
            TwoRateOperator(dt_fac*rig.large_dt, rig, small_n), whole_n))
        mpl.scatter(evalues.real, evalues.imag, 
                marker=markers[i],
                color=colors[i],
                s=9,
                label=str(dt_fac),
                )

    evalues, evectors = la.eig(rig.whole_dt*build_whole_dg_matrix(rig))
    mpl.scatter(evalues.real, evalues.imag, 
            marker="+", color="black", label="operator",
            s=5)

    mpl.xlabel(r"$\mathrm{Re}\, \lambda$")
    mpl.ylabel(r"$\mathrm{Im}\, \lambda$")
    mpl.grid()
    mpl.legend(loc="best")
    mpl.show()

def make_spectrum_animation(rig):
    small_n = len(rig.small_discr)
    whole_n = len(rig.whole_discr)

    op_evalues, op_evectors = la.eig(rig.whole_dt*build_whole_dg_matrix(rig))
    for i, dt_fac in enumerate(numpy.arange(0.3, 0.6, 0.005)):
        mpl.clf()
        mpl.title(str(dt_fac))
        evalues, evectors = la.eig(build_matrix(
            TwoRateOperator(dt_fac*rig.large_dt, rig, small_n), whole_n))
        mpl.scatter(evalues.real, evalues.imag, 
                marker='o', color='blue', s=9)

        mpl.scatter(op_evalues.real, op_evalues.imag, 
                marker="+", color="black", s=5)

        mpl.xlabel(r"$\mathrm{Re}\, \lambda$")
        mpl.ylabel(r"$\mathrm{Im}\, \lambda$")
        mpl.grid()
        mpl.xlim([-2,1])
        mpl.ylim([-2,2])
        mpl.savefig("spectrum-%04d.png" % i)
        print i



def main() :
    rig = LocalDtTestRig()
    #visualize_part_dg_matrix(rig)
    #visualize_diff_dg_matrix(rig)
    #plot_part_dg_eigenvalues(rig)
    #plot_eigenmodes(rig)
    #do_timestep(rig)
    #draw_multirate_spectrum(rig)
    make_spectrum_animation(rig)




if __name__ == "__main__":
    main()
