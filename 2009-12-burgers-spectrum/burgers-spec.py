from __future__ import division
import numpy




def main():
    from math import sin, pi
    from hedge.backends import guess_run_context
    rcon = guess_run_context()

    class case:
        a = -1
        b = 1

    if True:
        from hedge.mesh.generator import make_uniform_1d_mesh
        mesh = make_uniform_1d_mesh(case.a, case.b, 40, periodic=True)
    else:
        extent_y = 4
        dx = (case.b-case.a)/n_elements
        subdiv = (n_elements, int(1+extent_y//dx))
        from pytools import product

        from hedge.mesh.generator import make_rect_mesh
        mesh = make_rect_mesh((case.a, 0), (case.b, extent_y), 
                periodicity=(True, True), 
                subdivisions=subdiv,
                max_area=(case.b-case.a)*extent_y/(2*product(subdiv))
                )

    for order in [5]:
        from hedge.data import \
                ConstantGivenFunction, \
                TimeConstantGivenFunction, \
                TimeDependentGivenFunction
        from hedge.tools.second_order import (
                IPDGSecondDerivative, \
                LDGSecondDerivative, \
                CentralSecondDerivative)
        from hedge.models.burgers import BurgersOperator
        op = BurgersOperator(mesh.dimensions,
                viscosity_scheme=IPDGSecondDerivative(),
                viscosity=0.001
                )

        discr = rcon.make_discretization(mesh, order=order,
                    default_scalar_type=numpy.float64)

        bound_op = op.bind(discr)

        from hedge.visualization import SiloVisualizer
        vis = SiloVisualizer(discr, rcon)

        # timestep loop -------------------------------------------------------
        def ic_sawtooth(x, el):
            return x[0] % 1 - 0.5

        def ic_sine(x, el):
            return sin(x*pi)

        lin_center = discr.interpolate_volume_function(ic_sine)
        lin_center_rhs = bound_op(0, lin_center)

        def apply_linearized_burgers_op(operand):
            return (bound_op(0, lin_center + operand)
                    - lin_center_rhs)

        n = len(discr)
        op_mat = numpy.zeros((n, n), dtype=discr.default_scalar_type)

        from pytools import ProgressBar
        pb = ProgressBar("mat build", n)

        from hedge.tools import unit_vector
        mag = 1e-2
        for i in xrange(n):
            x = (1/mag)*apply_linearized_burgers_op(
                    mag*unit_vector(n, i, dtype=discr.default_scalar_type))
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
            vec = eigvec[:, i].real.copy()
            vis_data.extend([
                ("ev%04d_u" % i, discr.convert_volume(vec, kind="numpy")),
                ])

        visf = vis.make_file("eigenvalues")
        vis.add_data(visf, vis_data)
        visf.close()

if __name__ == "__main__":
    main()
