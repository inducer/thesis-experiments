# Hedge - the Hybrid'n'Easy DG Environment
# Copyright (C) 2008 Andreas Kloeckner
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




from __future__ import division, with_statement
import numpy
import scipy.linalg as la
la.cond = numpy.linalg.cond
from matplotlib.pyplot import plot, show, legend, clf, savefig, xlim, ylim, grid, title, spy




def build_mat(discr, bound_op):
    from hedge.tools import count_subset
    n = len(discr)
    op_mat = numpy.zeros((n, n), dtype=discr.default_scalar_type)

    from pytools import ProgressBar
    pb = ProgressBar("mat build", n)

    from hedge.tools import unit_vector
    for i in xrange(n):
        uvec = unit_vector(n, i, dtype=discr.default_scalar_type)
        op_result = bound_op(uvec)
        op_mat[:, i] = op_result
        pb.progress()
    pb.finished()

    return op_mat




def make_block_jacobi_preconditioner(discr, mat):
    eg, = discr.element_groups
    ldis = eg.local_discretization

    np = ldis.node_count()

    p = numpy.zeros_like(mat)
    n = len(discr)
    k = n//np

    for i in range(k):
        p[i*np:(i+1)*np, i*np:(i+1)*np] = la.inv(mat[i*np:(i+1)*np, i*np:(i+1)*np])

    return p





def make_matrix_free_block_diagonal_preconditioner(discr, op):
    eg, = discr.element_groups
    ldis = eg.local_discretization

    np = ldis.node_count()

    n = len(discr)
    mat = numpy.zeros((n, n), dtype=discr.default_scalar_type)
    k = n//np

    rhs = op.bind(discr)

    for i in xrange(np):
        uvec = numpy.zeros(n, dtype=numpy.float64)
        for ki in range(k):
            uvec[i+ki*np] = 1
        op_result = rhs(uvec)
        mat[:, i] = op_result

    p = numpy.zeros_like(mat)
    for i in range(k):
        p[i*np:(i+1)*np, i*np:(i+1)*np] = la.inv(mat[i*np:(i+1)*np, :np])

    return p




def block_average(discr, mat, scale_method):
    eg, = discr.element_groups
    ldis = eg.local_discretization

    np = ldis.node_count()
    n = len(discr)
    k = n//np

    #jacobian_values = []
    norm_values = []
    exp_values = []

    p0 = numpy.zeros((np,np), dtype=numpy.float64)
    for i in range(k):
        eg, grp_idx = discr.group_map[i]
        el = eg.members[grp_idx]

        #exp_values.append(la.svd(el.inverse_map.matrix)[1][0].real)
        #exp_values.append(numpy.trace(el.inverse_map.matrix))
        exp_values.append(el.inverse_map.jacobian())

        submat = mat[i*np:(i+1)*np, i*np:(i+1)*np]
        norm_values.append(la.norm(submat))

        p0 = p0 + (1/k)*submat

    ave_norm = la.norm(p0)
    #from matplotlib.pyplot import plot, show
    #plot(exp_values, norm_values, "o")
    #show()

    p = numpy.zeros_like(mat)
    for i in range(k):
        if scale_method == "norm":
            p[i*np:(i+1)*np, i*np:(i+1)*np] = (norm_values[i]/ave_norm)*p0
        elif scale_method == "exp":
            p[i*np:(i+1)*np, i*np:(i+1)*np] = exp_values[i]*p0
        elif scale_method is None:
            p[i*np:(i+1)*np, i*np:(i+1)*np] = p0
        else:
            raise ValueError(scale_method)

    return p




def make_wandering_matrix_free_block_diagonal_preconditioner(discr, op):
    eg, = discr.element_groups
    ldis = eg.local_discretization

    np = ldis.node_count()

    n = len(discr)
    mat = numpy.zeros((n, n), dtype=discr.default_scalar_type)
    k = n//np

    rhs = op.bind(discr)

    p = numpy.zeros_like(mat)

    stepsizes = [2, 5]
    for stepsize in stepsizes:
        for step_offset in range(stepsize):
            for i in xrange(np):
                uvec = numpy.zeros(n, dtype=numpy.float64)
                for ki in xrange(step_offset, k, stepsize):
                    uvec[i+ki*np] = 1
                op_result = rhs(uvec)
                mat[:, i] = op_result

            for ki in xrange(step_offset, k, stepsize):
                submat = mat[ki*np:(ki+1)*np, :np]
                p[ki*np:(ki+1)*np, ki*np:(ki+1)*np] += submat

            #from matplotlib.pyplot import show, spy
            #spy(p); show()

    for ki in xrange(k):
        submat = p[ki*np:(ki+1)*np, ki*np:(ki+1)*np]
        p[ki*np:(ki+1)*np, ki*np:(ki+1)*np] = la.inv(
                1/len(stepsizes) * submat)

    return p





def make_averaged_matrix_free_block_diagonal_preconditioner(discr, bound_op):
    eg, = discr.element_groups
    ldis = eg.local_discretization

    np = ldis.node_count()

    n = len(discr)
    mat = numpy.zeros((n, n), dtype=discr.default_scalar_type)
    k = n//np

    step_size = 5
    max_ave_count = (k+step_size-1) // step_size

    inv_ave = numpy.zeros((np, np), dtype=numpy.float64)
    blocks_added = 0

    ave_blocks = numpy.zeros((max_ave_count, np, np), dtype=numpy.float64)

    for k_offset in range(step_size):
        for i in xrange(np):
            uvec = numpy.zeros(n, dtype=numpy.float64)
            for ki in xrange(k_offset, k, step_size):
                uvec[i+ki*np] = 1

            op_result = bound_op(uvec)
            op_result.shape = (k, np)

            for ki in xrange(k_offset, k, step_size):
                j = (ki-k_offset)//step_size
                ave_blocks[j, :, i] = op_result[ki]

        for j in xrange((k-k_offset)//step_size):
            inv_ave += la.inv(ave_blocks[j])
            blocks_added += 1

    inv_ave = inv_ave/blocks_added

    p = numpy.zeros((n,n), dtype=numpy.float64)
    for i in range(k):
        p[i*np:(i+1)*np, i*np:(i+1)*np] = inv_ave

    return p





def main():
    from hedge.backends import guess_run_context
    rcon = guess_run_context()

    def refine_func(vertices, area):
        barycenter = sum(numpy.array(v) for v in vertices)
        #return bool(area > 0.002 + 0.005*la.norm(barycenter)**2)
        return bool(area > 0.01 + 0.03*la.norm(barycenter)**2)

    if False:
        from hedge.mesh import make_rect_mesh
        mesh = make_rect_mesh(a=(-1,-1), b=(1,1,), refine_func=refine_func)
    elif True:
        class case:
            a = -1
            b = 1

        n_elements = 20
        from hedge.mesh.generator import make_uniform_1d_mesh
        mesh = make_uniform_1d_mesh(case.a, case.b, n_elements, periodic=True)
    else:
        class case:
            a = -1
            b = 1

        n_elements = 5

        extent_y = 0.5
        dx = (case.b-case.a)/n_elements
        subdiv = (n_elements, int(1+extent_y//dx))
        print subdiv
        from pytools import product

        from hedge.mesh.generator import make_rect_mesh
        mesh = make_rect_mesh((case.a, 0), (case.b, extent_y), 
                periodicity=(True, True), 
                subdivisions=subdiv,
                max_area=(case.b-case.a)*extent_y/(2*product(subdiv))
                )

    def rhs(x, el):
        if la.norm(x) < 0.1:
            return 1000
        else:
            return 0

    from hedge.models.poisson import PoissonOperator
    from hedge.mesh import TAG_ALL, TAG_NONE
    from hedge.tools.second_order import (
            StabilizedCentralSecondDerivative,
            LDGSecondDerivative,
            IPDGSecondDerivative)

    op_kwargs = dict(dimensions=mesh.dimensions,
            dirichlet_tag=TAG_ALL,
            neumann_tag=TAG_NONE, 
            )
    op = PoissonOperator(
            scheme=LDGSecondDerivative(),
            #scheme=IPDGSecondDerivative(),
            #scheme=StabilizedCentralSecondDerivative(),
            **op_kwargs)

    orders = range(1, 6)

    conditions = {}
    it_counts = {}

    for n in orders:
        discr = rcon.make_discretization(mesh, order=n,
                    default_scalar_type=numpy.float64,
                    debug=["dump_op_code"])
        eg, = discr.element_groups
        ldis = eg.local_discretization
        np = ldis.node_count()

        bound_op = op.bind(discr)

        op_mat = build_mat(discr, bound_op)

        eigval, eigvec = la.eig(op_mat)
        plot(eigval.real, eigval.imag, "o")
        grid()
        show()
        return

        bj_mat = make_block_jacobi_preconditioner(discr, op_mat)

        block_avg = block_average(discr, bj_mat, scale_method=None)
        mf_block_avg = make_averaged_matrix_free_block_diagonal_preconditioner(discr, bound_op)
        preconditioners = [
                ("without", None),
                ("block jacobi", bj_mat),
                ("averaged", block_avg),
                ("averaged norm-scaled", block_average(discr, bj_mat, scale_method="norm")),
                #("matrix-free", 
                    #make_wandering_matrix_free_block_diagonal_preconditioner(discr, op)),
                ("matrix-free averaged", mf_block_avg),
                ]
        #print block_avg[:np,:np]-mf_block_avg[:np,:np]
        #print block_avg[:np,:np]
        #print mf_block_avg[:np,:np]

        if False:
            op_local = PoissonOperator(
                    scheme=LocalOnlySecondDerivative(),
                    **op_kwargs)

            pmat = make_matrix_free_block_diagonal_preconditioner(discr, op_local)
            eigval, eigvec = la.eig(numpy.dot(pmat, op_mat))
            # print max(abs(eigval.real)), min(abs(eigval.real))
            condition = max(abs(eigval.real))/min(abs(eigval.real))
            mf_conditions.append(condition)

        print "built precons"

        if False:
            for name, pmat in preconditioners:
                if pmat is not None:
                    fullmat = numpy.dot(pmat, op_mat)
                else:
                    fullmat = op_mat

                conditions.setdefault(name, []).append(la.cond(fullmat))

            print "computed conditions"

        for name, pmat in preconditioners:
            if pmat is not None:
                def precon_f(x):
                    return numpy.dot(pmat, x)
            else:
                precon_f = None
            from hedge.iterative import parallel_cg
            from hedge.data import GivenFunction

            it_count = [0]
            def cg_debug_callback(what, iterations, x, residual, d, delta):
                if what == "end":
                    it_count[0] = iterations

            print name

            u = -parallel_cg(rcon, -op.bind(discr),
                    bound_op.prepare_rhs(
                        discr.interpolate_volume_function(rhs)),
                    precon=precon_f,
                    tol=5e-10, debug_callback=cg_debug_callback,
                    dot=discr.nodewise_dot_product,
                    x=discr.volume_zeros())

            it_counts.setdefault(name, []).append(it_count[0])

        if False:
            from hedge.visualization import SiloVisualizer
            vis = SiloVisualizer(discr, rcon)
            with vis.make_file("soln") as visf:
                vis.add_data(visf, [("u", u)])

    #savefig("poisson-spec-strong%s.png" % strong)

    #print strong, la.norm(op_mat-op_mat.T)

    #xlim([-10,1])
    #ylim([-2.5,2.5])
    #grid()
    #legend()
    #show()

    if False:
        clf()
        for name, data in conditions.iteritems():
            plot(orders, data, label=name)

            print name, data
            print "exp_estimate", name, numpy.polyfit(
                    numpy.log10(orders), 
                    numpy.log10(data), 1)[-2]

        title("Poisson condition numbers")
        legend()
        show()
    print "-------------------------------------------------"
    clf()
    for name, data in it_counts.iteritems():
        plot(orders, data, label=name)

        print name, data
        print "exp_estimate", name, numpy.polyfit(
                numpy.log10(orders), 
                numpy.log10(data), 1)[-2]

    title("Poisson iteration counts")
    legend()
    show()

    #spy(op_mat)
    #show()

    eigval = sorted(eigval, key=lambda x: x.real)

    if False:
        vis_data = []
        for i, value in enumerate(eigval):
            print i, value
            split_vec = eigvec[:, i].real.copy()
            vis_data.extend([
                ("ev%04d" % i, split_vec),
                ])

        from hedge.visualization import SiloVisualizer
        vis = SiloVisualizer(discr, rcon)
        visf.close()

if __name__ == "__main__":
    main()
