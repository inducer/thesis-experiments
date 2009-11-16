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




def build_mat(rcon, mesh, op):
    discr = rcon.make_discretization(mesh, order=3,
                default_scalar_type=numpy.float64,
                tune_for=op.op_template())

    from hedge.visualization import SiloVisualizer
    vis = SiloVisualizer(discr, rcon)

    def split_large_vec(v):
        from pytools.obj_array import make_obj_array
        return make_obj_array([
            v[len(discr)*i:len(discr)*(i+1)]
            for i in range(field_count)])

    from hedge.tools import count_subset
    field_count = count_subset(op.get_eh_subset())
    n = field_count*len(discr)
    op_mat = numpy.zeros((n, n), dtype=discr.default_scalar_type)

    from pytools import ProgressBar
    pb = ProgressBar("mat build", n)

    rhs = op.bind(discr)

    dt = op.estimate_timestep(discr)

    from hedge.tools import unit_vector
    for i in xrange(n):
        uvec = unit_vector(n, i, dtype=discr.default_scalar_type)
        op_result = rhs(0, split_large_vec(uvec))
        for j in range(field_count):
            op_mat[len(discr)*j:len(discr)*(j+1), i] = op_result[j]
        pb.progress()
    pb.finished()

    discr.close()

    return dt * op_mat




def main():
    from hedge.backends import guess_run_context
    rcon = guess_run_context()

    from hedge.mesh import make_rect_mesh
    mesh = make_rect_mesh(max_area=0.05)

    from hedge.models.em import TMMaxwellOperator
    from hedge.mesh import TAG_ALL
    from matplotlib.pyplot import plot, show, legend

    for penalty_factor in [1, 5, 30]:
        op = TMMaxwellOperator(1, 1, flux_type=penalty_factor,
                #absorb_tag=TAG_ALL, 
                pec_tag=TAG_ALL)

        op_mat = build_mat(rcon, mesh, op)

        eigval, eigvec = la.eig(op_mat)
        plot(eigval.real, eigval.imag, "o", label="penalty=%g" % penalty_factor)

    legend()
    show()

    eigval = sorted(eigval, key=lambda x: x.real)

    if False:
        vis_data = []
        for i, value in enumerate(eigval):
            print i, value
            split_vec = split_large_vec(eigvec[:, i].real.copy())
            vis_data.extend([
                ("ev%04d_rho" % i, discr.convert_volume(op.rho(split_vec), kind="numpy")),
                ("ev%04d_e" % i, discr.convert_volume(op.e(split_vec), kind="numpy")),
                ("ev%04d_rho_u" % i, discr.convert_volume(op.rho_u(split_vec), kind="numpy")),
                ("ev%04d_u" % i, discr.convert_volume(op.u(split_vec), kind="numpy")),
                ])

        visf = vis.make_file("eigenvalues")
        vis.add_data(visf, vis_data)
        visf.close()

if __name__ == "__main__":
    main()
