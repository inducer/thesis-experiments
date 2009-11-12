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




def main():
    from hedge.backends import guess_run_context
    rcon = guess_run_context()

    from pytools import add_python_path_relative_to_script
    add_python_path_relative_to_script("../../hedge/examples/gas_dynamics")

    from gas_dynamics_initials import UniformMachFlow
    uflow = UniformMachFlow(reynolds=1)

    flow_dir = uflow.direction_vector(2)
    def boundary_tagger(vertices, el, face_nr, all_v):
        if numpy.dot(el.face_normals[face_nr], flow_dir) <= 0:
            return ["inflow"]
        else:
            return ["outflow"]

    from hedge.mesh import make_rect_mesh
    mesh = make_rect_mesh(boundary_tagger=boundary_tagger, 
            max_area=0.05)

    for order in [3]:
        from hedge.models.gas_dynamics import GasDynamicsOperator
        print "mu", uflow.mu
        op = GasDynamicsOperator(dimensions=2,
                gamma=uflow.gamma, mu=uflow.mu,
                prandtl=uflow.prandtl, spec_gas_const=uflow.spec_gas_const,
                bc_inflow=uflow, bc_outflow=uflow, bc_noslip=uflow,
                inflow_tag="inflow", outflow_tag="outflow", noslip_tag="noslip")

        discr = rcon.make_discretization(mesh, order=order,
                    debug=[
                        "cuda_no_plan",
                        #"cuda_dump_kernels",
                        #"dump_dataflow_graph",
                        #"dump_optemplate_stages",
                        #"dump_dataflow_graph",
                        #"print_op_code"
                        #"cuda_no_plan_el_local"
                        ],
                    default_scalar_type=numpy.float64,
                    tune_for=op.op_template())

        from hedge.visualization import SiloVisualizer
        vis = SiloVisualizer(discr, rcon)

        # timestep loop -------------------------------------------------------
        free_stream = uflow.volume_interpolant(0, discr)
        field_count = len(free_stream)

        navierstokes_ex = op.bind(discr)

        max_eigval = [0]
        def rhs(t, q):
            ode_rhs, speed = navierstokes_ex(t, q)
            max_eigval[0] = speed
            return ode_rhs

        free_stream_rhs = rhs(0, free_stream) # should be zero

        def split_large_vec(v):
            from pytools.obj_array import make_obj_array
            return make_obj_array([
                v[len(discr)*i:len(discr)*(i+1)]
                for i in range(field_count)])

        import pyublasext
        class LinearizedCNSOperator(pyublasext.Operator(numpy.float64)):
            def size1(self):
                return len(discr)*field_count

            def size2(self):
                return len(discr)*field_count

            def apply(self, operand, result):
                op_result = (
                        rhs(0, free_stream + split_large_vec(operand)) 
                        - free_stream_rhs)

                for i in range(field_count):
                    result[len(discr)*i:len(discr)*(i+1)] = op_result[i]

        cns_lin_op = LinearizedCNSOperator()

        if False:
            #results = pyublasext.operator_eigenvectors(cns_lin_op, 50)
            print "DOFS:", len(discr)
            results_large_real = pyublasext.operator_eigenvectors(
                    cns_lin_op, 
                    #int(0.1*len(discr)),
                    3,
                    which=pyublasext.LARGEST_REAL_PART)
            print "FINISHED LARGE REAL FINDING"

            results_small_real = pyublasext.operator_eigenvectors(
                    cns_lin_op, 
                    #int(0.1*len(discr)),
                    3,
                    which=pyublasext.SMALLEST_REAL_PART)
            print "FINISHED SMALL REAL FINDING"

            eigenvalues = []
            vis_data = []
            print "SMALL REAL:"
            for i, (value, vector) in enumerate(results_small_real):
                print i, value
                vis_data.append(("sr_%04d" % i, vector.real.copy()))
                eigenvalues.append(value)

            print "LARGE REAL:"
            for i, (value, vector) in enumerate(results_large_real):
                print i, value
                vis_data.append(("lr_%04d" % i, vector.real.copy()))
                eigenvalues.append(value)

            visf = vis.make_file("eigenvalues")
            vis.add_data(visf, vis_data)
            visf.close()

            from matplotlib.pyplot import plot, show
            plot(
                    [v.real for v in eigenvalues],
                    [v.imag for v in eigenvalues],
                    "o")
            show()
        else:
            n = field_count*len(discr)
            op_mat = numpy.zeros((n, n), dtype=discr.default_scalar_type)

            from pytools import ProgressBar
            pb = ProgressBar("mat build", n)

            from hedge.tools import unit_vector
            for i in xrange(n):
                x = cns_lin_op(unit_vector(n, i, dtype=discr.default_scalar_type))
                op_mat[:, i] = x
                pb.progress()
            pb.finished()

            import scipy.linalg as la
            eigval, eigvec = la.eig(op_mat)

            from matplotlib.pyplot import plot, show
            plot(eigval.real, eigval.imag, "o")
            show()

            eigval = sorted(eigval, key=lambda x: x.real)

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
