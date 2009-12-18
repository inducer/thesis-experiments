from __future__ import division
from hedge.discretization.local import LocalDiscretization
from pytools import memoize_method
import numpy
import numpy.linalg as la
import hedge.mesh.element





class TrigonometricIntervalDiscretization(LocalDiscretization):
    """An arbitrary-order trigonometric finite interval element.

    Coordinate systems used:
    ========================

    unit coordinates (r)::

        ---[--------0--------]--->
           -1                1
    """
    dimensions = 1
    geometry = hedge.mesh.element.Interval

    def __init__(self, order):
        self.order = order

    @property
    def has_facial_nodes(self):
        return self.order > 0

    # node wrangling ----------------------------------------------------------
    def node_count(self):
        return 2*self.order+1

    def nodes(self):
        """Generate warped nodes in unit coordinates (r,)."""

        if self.order == 0:
            return [numpy.array([0.5])]
        else:
            from hedge.quadrature import legendre_gauss_lobatto_points
            return [numpy.array([x])
                    for x in legendre_gauss_lobatto_points(2*self.order)]

    def nodes2(self):
        return [numpy.array([x])
                for x in numpy.linspace(-1, 1, 2*self.order+1)]

    equilateral_nodes = nodes
    unit_nodes = nodes

    @memoize_method
    def get_submesh_indices(self):
        """Return a list of tuples of indices into the node list that
        generate a tesselation of the reference element."""

        return [(i, i + 1) for i in range(self.order)]


    # basis functions ---------------------------------------------------------
    class _TrigBasisFunction:
        def __init__(self, n):
            self.is_cos = n % 2 == 0
            self.n = (n+1) // 2

        def __call__(self, x):
            if self.is_cos:
                return numpy.cos(numpy.pi/2*self.n*x[0])
            else:
                return numpy.sin(numpy.pi/2*self.n*x[0])

    class DiffTrigBasisFunction(_TrigBasisFunction):
        def __call__(self, x):
            if self.is_cos:
                return -numpy.pi/2*self.n*numpy.sin(numpy.pi/2*self.n*x[0])
            else:
                return numpy.pi/2*self.n*numpy.cos(numpy.pi/2*self.n*x[0])


    def generate_mode_identifiers(self):
        for i in range(2*self.order+1):
            yield (i,)

    @memoize_method
    def basis_functions(self):
        """Get a sequence of functions that form a basis of the approximation
        space.
        """

        return [self._TrigBasisFunction(idx[0]) 
                for idx in self.generate_mode_identifiers()]

    def grad_basis_functions(self):
        """Get the gradient functions of the basis_functions(), in the
        same order.
        """

        return [self._DiffTrigBasisFunction(idx[0]) 
                for idx in self.generate_mode_identifiers()]

    # matrices ----------------------------------------------------------------
    @memoize_method
    def modal_mass_matrix(self):
        pi = numpy.pi
        sin = numpy.sin
        cos = numpy.cos

        def get_entry(i, j):
            func_a = self._TrigBasisFunction(i)
            func_b = self._TrigBasisFunction(j)
            n = func_a.n
            m = func_b.n

            if func_a.is_cos != func_b.is_cos:
                return 0

            if n == m:
                if n == 0:
                    return 2
                else:
                    return 1
            else:
                if func_a.is_cos:
                    return 2/pi*(
                            1/(n+m) * sin((n*pi+m*pi)/2)
                            +
                            1/(n-m) * sin((n*pi-m*pi)/2))
                else:
                    return -2/pi*(
                            1/(n+m) * sin((n*pi+m*pi)/2)
                            -
                            1/(n-m) * sin((n*pi-m*pi)/2))

        nc = self.node_count()
        return numpy.array([[get_entry(i, j)
            for j in range(nc)]
            for i in range(nc)], dtype=numpy.float64)

    @memoize_method
    def mass_matrix(self):
        v = self.vandermonde()
        mm = self.modal_mass_matrix()

        from hedge.tools import leftsolve
        return numpy.asarray(la.solve(v.T, leftsolve(v, mm)), order="C")

    @memoize_method
    def inverse_mass_matrix(self):
        """Return the inverse of the mass matrix of the unit element
        with respect to the nodal coefficients. Divide by the Jacobian
        to obtain the global mass matrix.
        """

        return numpy.asarray(la.inv(self.mass_matrix()), order="C")

    @memoize_method
    def modal_differentiation_matrices(self):
        pi = numpy.pi

        nc = self.node_count()
        d = numpy.zeros((nc, nc), dtype=numpy.float64)

        for n in range(0, self.order):
            i = 2*n+1
            d[i+1, i] = pi/2*(n+1)
            d[i, i+1] = -pi/2*(n+1)

        return [d]

    @memoize_method
    def differentiation_matrices(self):
        """Return matrices that map the nodal values of a function
        to the nodal values of its derivative in each of the unit
        coordinate directions.
        """
        v = self.vandermonde()

        from hedge.tools import leftsolve
        return [numpy.asarray(numpy.dot(v, leftsolve(v, d)), order="C")
                for d in self.modal_differentiation_matrices()]

    # face operations ---------------------------------------------------------
    @memoize_method
    def face_mass_matrix(self):
        return numpy.array([[1]], dtype=float)

    @staticmethod
    def get_face_index_shuffle_to_match(face_1_vertices, face_2_vertices):
        if set(face_1_vertices) != set(face_2_vertices):
            from hedge.discretization.local import FaceVertexMismatch
            raise FaceVertexMismatch("face vertices do not match")

        class IntervalFaceIndexShuffle:
            def __hash__(self):
                return 0x3472477

            def __eq__(self, other):
                return True

            def __call__(self, indices):
                return indices

        return IntervalFaceIndexShuffle()

    @memoize_method
    def face_indices(self):
        return [(0,), (self.order*2,)]

    # time step scaling -------------------------------------------------------
    def dt_non_geometric_factor(self):
        if self.order == 0:
            return 1
        else:
            unodes = self.unit_nodes()
            return la.norm(unodes[0] - unodes[1]) * 0.85

    def dt_geometric_factor(self, vertices, el):
        return abs(el.map.jacobian())




def test_simp_mass_and_diff_matrices_by_monomial():
    """Verify simplicial mass and differentiation matrices using monomials"""

    from pytools import generate_nonnegative_integer_tuples_summing_to_at_most
    from hedge.tools import Monomial

    thresh = 1e-13

    from numpy import dot
    for el in [
            TrigonometricIntervalDiscretization(4),
            ]:
        for comb in generate_nonnegative_integer_tuples_summing_to_at_most(
                3, 1):
            unodes = el.unit_nodes()
            f = Monomial(comb)
            f_n = numpy.array([f(x) for x in unodes])
            int_f_n = numpy.sum(dot(el.mass_matrix(), f_n))
            int_f = f.theoretical_integral()
            err = la.norm(int_f - int_f_n)
            print "INTERR", err
            #if err > thresh:
                #print "bad", el, comb, int_f, int_f_n, err
            #assert err < thresh

            dmats = el.differentiation_matrices()
            for i in range(el.dimensions):
                df = f.diff(i)
                df = numpy.array([df(x) for x in unodes])/2
                df_n = dot(dmats[i], f_n)
                err = la.norm(df - df_n, numpy.Inf)
                print "DIFFERR", err
                #if err > thresh:
                    #print "bad-diff", comb, i, err
                #assert err < thresh




def plot_interpolants():
    from hedge.discretization.local import IntervalDiscretization

    def f1(x):
        if x > 0:
            return x
        else:
            return 1

    def f2(x):
        return numpy.sin(5*x)

    def f2p(x):
        return 5*numpy.cos(5*x)

    def f3(x):
        #return 8*x**4 -8*x**2 +1
        return 16*x**5 - 20*x**3 + 5*x

    f = f1

    def intp_f(x):
        return sum(coeff*bf([x]) 
                for coeff, bf in zip(modal_coeffs, t.basis_functions()))

    xpoints = numpy.linspace(-1, 1, 500, endpoint=True)

    from matplotlib.pyplot import plot, show, legend
    plot(xpoints, [f(x) for x in xpoints], label="true")

    n = 7
    for t in [
            TrigonometricIntervalDiscretization(n),
            IntervalDiscretization(2*n)]:
        print t.node_count()

        nodal_coeffs = numpy.array([f(x[0]) for x in t.nodes()])
        #nodal_coeffs = numpy.dot(t.differentiation_matrices()[0], nodal_coeffs)
        modal_coeffs = la.solve(t.vandermonde(), nodal_coeffs)

        plot(xpoints, [intp_f(x) for x in xpoints], label=t.__class__.__name__[:5])

    legend()
    show()




def test_advection(flux_type_arg="upwind"):
    from hedge.tools import mem_checkpoint
    from math import sin, cos, pi, sqrt
    from math import floor

    from hedge.backends import guess_run_context
    rcon = guess_run_context()

    def f(x):
        return sin(pi*x)

    def u_analytic(x, el, t):
        return f((-numpy.dot(v, x)/norm_v+t*norm_v))

    def boundary_tagger(vertices, el, face_nr, all_v):
        if numpy.dot(el.face_normals[face_nr], v) < 0:
            return ["inflow"]
        else:
            return ["outflow"]

    v = numpy.array([1])
    if rcon.is_head_rank:
        from hedge.mesh import make_uniform_1d_mesh
        mesh = make_uniform_1d_mesh(0, 2, 10, 
                boundary_tagger=boundary_tagger
                #periodic=True
                )

    norm_v = la.norm(v)

    discr = rcon.make_discretization(mesh, 
            local_discretization=TrigonometricIntervalDiscretization(5)
            #order=10
            )

    from hedge.visualization import SiloVisualizer
    vis = SiloVisualizer(discr, rcon)

    # operator setup ----------------------------------------------------------
    from hedge.data import \
            ConstantGivenFunction, \
            TimeConstantGivenFunction, \
            TimeDependentGivenFunction
    from hedge.models.advection import StrongAdvectionOperator, WeakAdvectionOperator
    op = WeakAdvectionOperator(v, 
            inflow_u=TimeDependentGivenFunction(u_analytic),
            flux_type=flux_type_arg)

    u = discr.interpolate_volume_function(lambda x, el: u_analytic(x, el, 0))

    # timestep setup ----------------------------------------------------------
    from hedge.timestep import RK4TimeStepper
    stepper = RK4TimeStepper()

    if rcon.is_head_rank:
        print "%d elements" % len(discr.mesh.elements)

    # diagnostics setup -------------------------------------------------------
    from pytools.log import LogManager, \
            add_general_quantities, \
            add_simulation_quantities, \
            add_run_info

    logmgr = LogManager("advection.dat", "w", rcon.communicator)
    add_run_info(logmgr)
    add_general_quantities(logmgr)
    add_simulation_quantities(logmgr)
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

    try:
        from hedge.timestep import times_and_steps
        step_it = times_and_steps(
                final_time=3, logmgr=logmgr,
                max_dt_getter=lambda t: 0.015)
                #max_dt_getter=lambda t: op.estimate_timestep(discr,
                    #stepper=stepper, t=t, fields=u))

        for step, t, dt in step_it:
            if step % 5 == 0:
                visf = vis.make_file("fld-%04d" % step)
                vis.add_data(visf, [ ("u", u), ],
                            time=t,
                            step=step
                            )
                visf.close()

            u = stepper(u, t, dt, rhs)

    finally:
        vis.close()
        logmgr.close()

    true_u = discr.interpolate_volume_function(lambda x, el: u_analytic(x, el, t))
    print discr.norm(u-true_u)
    #assert discr.norm(u-true_u) < 1e-2

    discr.close()



def build_mat(discr, mesh, op):
    from hedge.tools import count_subset
    n = len(discr)
    op_mat = numpy.zeros((n, n), dtype=discr.default_scalar_type)

    from pytools import ProgressBar
    pb = ProgressBar("mat build", n)

    rhs = op.bind(discr)

    from hedge.tools import unit_vector
    for i in xrange(n):
        uvec = unit_vector(n, i, dtype=discr.default_scalar_type)
        op_result = rhs(0, uvec)
        op_mat[:, i] = op_result
        pb.progress()
    pb.finished()

    return op_mat




def plot_advection_eigenvalues():
    from hedge.backends import guess_run_context
    rcon = guess_run_context()

    def boundary_tagger(vertices, el, face_nr, all_v):
        if numpy.dot(el.face_normals[face_nr], v) < 0:
            return ["inflow"]
        else:
            return ["outflow"]

    v = numpy.array([1])
    from hedge.mesh import make_uniform_1d_mesh
    mesh = make_uniform_1d_mesh(0, 2, 3, 
            boundary_tagger=boundary_tagger,
            periodic=True
            )

    from hedge.data import make_tdep_constant
    from hedge.models.advection import StrongAdvectionOperator, WeakAdvectionOperator
    for flux_type in ["central", "upwind"]:
        op = WeakAdvectionOperator(v, 
                inflow_u=make_tdep_constant(0),
                flux_type=flux_type)

        from hedge.discretization.local import IntervalDiscretization
        from matplotlib.pyplot import plot, show, legend, title
        for ldis_maker in [
                TrigonometricIntervalDiscretization,
                lambda n: IntervalDiscretization(2*n),
                ]:

            n_values = range(0, 9)
            node_count_values = []
            max_evalues = []
            for n in n_values:
                ldis = local_discretization=ldis_maker(n)
                discr = rcon.make_discretization(mesh, local_discretization=ldis)
                node_count_values.append(ldis.node_count())

                from hedge.visualization import SiloVisualizer
                vis = SiloVisualizer(discr, rcon)

                # operator setup ----------------------------------------------------------
                import scipy.linalg as sla
                mat = build_mat(discr, mesh, op)
                evalues, evectors = sla.eig(mat)

                #plot(evalues.real, evalues.imag, "o",
                        #label="%s(%d)" % (ldis.__class__.__name__[:5], n))
                max_evalues.append(max(numpy.abs(evalues.imag)))

            if True:
                name = "%s %s" % (
                            ldis.__class__.__name__[:5], flux_type)
                print "exp_estimate", name, numpy.polyfit(
                        numpy.log10(node_count_values), 
                        numpy.log10(max_evalues), 1)[-2]
                plot(node_count_values, max_evalues, 
                        label=name)
    title("Trig basis eigenvalues")
    legend()
    show()




def main():
    from hedge.discretization.local import IntervalDiscretization
    #t = TrigonometricIntervalDiscretization(2)
    t = IntervalDiscretization(2)
    print t.face_indices()

    if False:
        numpy.set_printoptions(precision=3, linewidth=120, suppress=True)
        print t.modal_differentiation_matrices()
        print t.modal_mass_matrix()
        print t.mass_matrix()
        print la.cond(t.mass_matrix())
        print t.vandermonde()
        print la.cond(t.vandermonde())

    #test_simp_mass_and_diff_matrices_by_monomial()
    plot_interpolants()
    #test_advection()
    #plot_advection_eigenvalues()



if __name__ == "__main__":
    main()
