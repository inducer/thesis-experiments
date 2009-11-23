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




from __future__ import division
import numpy
import scipy.linalg as la




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
        op_result = rhs(uvec)
        op_mat[:, i] = op_result
        pb.progress()
    pb.finished()

    return op_mat, discr




def make_block_diagonal_preconditioner(discr, mat):
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





def make_wandering_matrix_free_block_diagonal_preconditioner(discr, op):
    print "BUILD START"
    eg, = discr.element_groups
    ldis = eg.local_discretization

    np = ldis.node_count()

    n = len(discr)
    mat = numpy.zeros((n, n), dtype=discr.default_scalar_type)
    k = n//np

    rhs = op.bind(discr)

    p = numpy.zeros_like(mat)

    stepsizes = [2, 3]
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

    print "BUILD END"
    return p





def main():
    from hedge.backends import guess_run_context
    rcon = guess_run_context()

    from hedge.mesh import make_rect_mesh
    mesh = make_rect_mesh(max_area=0.05)

    from hedge.models.poisson import PoissonOperator
    from hedge.mesh import TAG_ALL, TAG_NONE
    from matplotlib.pyplot import plot, show, legend, clf, savefig, xlim, ylim, grid, title, spy
    from hedge.models.nd_calculus import (
            StabilizedCentralSecondDerivative,
            LDGSecondDerivative,
            LocalOnlySecondDerivative)

    op_kwargs = dict(dimensions=2,
            dirichlet_tag=TAG_ALL,
            neumann_tag=TAG_NONE, 
            strong_form=False,
            )
    op = PoissonOperator(
            #scheme=LDGSecondDerivative(),
            scheme=StabilizedCentralSecondDerivative(),
            **op_kwargs)

    n_values = []
    without_conditions = []
    bd_conditions = []
    mf_conditions = []

    for n in range(1, 6):
        n_values.append(n)
        discr = rcon.make_discretization(mesh, order=n,
                    default_scalar_type=numpy.float64)

        op_mat, discr = build_mat(discr, mesh, op)

        eigval, eigvec = la.eig(op_mat)
        condition = max(abs(eigval.real))/min(abs(eigval.real))
        without_conditions.append(condition)

        pmat = make_block_diagonal_preconditioner(discr, op_mat)
        eigval, eigvec = la.eig(numpy.dot(pmat, op_mat))
        condition = max(abs(eigval.real))/min(abs(eigval.real))
        bd_conditions.append(condition)

        if False:
            op_local = PoissonOperator(
                    scheme=LocalOnlySecondDerivative(),
                    **op_kwargs)

            pmat = make_matrix_free_block_diagonal_preconditioner(discr, op_local)
            eigval, eigvec = la.eig(numpy.dot(pmat, op_mat))
            print max(abs(eigval.real)), min(abs(eigval.real))
            condition = max(abs(eigval.real))/min(abs(eigval.real))
            mf_conditions.append(condition)

        pmat = make_wandering_matrix_free_block_diagonal_preconditioner(discr, op)
        eigval, eigvec = la.eig(numpy.dot(pmat, op_mat))
        print max(abs(eigval.real)), min(abs(eigval.real))
        condition = max(abs(eigval.real))/min(abs(eigval.real))
        mf_conditions.append(condition)

    #savefig("poisson-spec-strong%s.png" % strong)

    #print strong, la.norm(op_mat-op_mat.T)

    #xlim([-10,1])
    #ylim([-2.5,2.5])
    #grid()
    #legend()
    #show()

    clf()
    for name, data in [
            ("without", without_conditions),
            ("blockdiag", bd_conditions),
            ("matrixfree", mf_conditions),
            ]:
        plot(n_values, data, label=name)

        print name, data
        print "exp_estimate", name, numpy.polyfit(
                numpy.log10(n_values), 
                numpy.log10(data), 1)[-2]

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
        visf = vis.make_file("eigenvalues")
        vis.add_data(visf, vis_data)
        visf.close()

if __name__ == "__main__":
    main()
