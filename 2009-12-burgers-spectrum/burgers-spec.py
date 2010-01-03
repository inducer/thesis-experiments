from __future__ import division
import numpy




def make_bump(discr):
    el_values = {}

    from random import uniform, seed

    seed(1)

    def bumpy_f(x, el):
        try:
            return el_values[el.id]
        except KeyError:
            i = 0
            if i == 0:
                if el.id > 5:
                    result = 1
                else:
                    result = 0
            elif i == 1:
                x = uniform(0, 1)
                if x < 0.15:
                    result = 1
                else:
                    result = 0
            else:
                result = uniform(0, 1)
                if uniform(0,1) > 0.05:
                    result = 0
            el_values[el.id] = result
            return result

    from smoother import TriBlobSmoother
    smoother = TriBlobSmoother(discr)

    return smoother(discr.interpolate_volume_function(bumpy_f))




def main():
    from pytools import add_python_path_relative_to_script
    add_python_path_relative_to_script("../2009-12-artificial-viscosity")

    from math import sin, pi
    from hedge.backends import guess_run_context
    rcon = guess_run_context()

    class case:
        a = -1
        b = 1

    n_elements = 50
    if True:
        from hedge.mesh.generator import make_uniform_1d_mesh
        mesh = make_uniform_1d_mesh(case.a, case.b, n_elements, periodic=True)
    else:
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

    order = 5
    for viscosity in [0., 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
    #for viscosity in [1.]:
        from hedge.tools.second_order import (
                IPDGSecondDerivative, \
                LDGSecondDerivative, \
                CentralSecondDerivative)
        from hedge.models.burgers import BurgersOperator
        op = BurgersOperator(mesh.dimensions,
                viscosity_scheme=IPDGSecondDerivative(
                    #stab_coefficient=10,
                    ),
                #viscosity=viscosity
                )

        discr = rcon.make_discretization(mesh, order=order,
                    default_scalar_type=numpy.float64,
                    debug=[
                        "dump_op_code"
                        ])

        from hedge.visualization import SiloVisualizer
        vis = SiloVisualizer(discr, rcon)

        # timestep loop -------------------------------------------------------
        def ic_sawtooth(x, el):
            return x[0] % 1 - 0.5

        def ic_sine(x, el):
            return sin(x*pi)

        def ic_constant(x, el):
            return 1

        lin_center = discr.interpolate_volume_function(ic_constant)

        bump = make_bump(discr)
        bound_op = op.bind(discr, u0=lin_center, sensor=lambda u: viscosity*bump)

        lin_center_rhs = bound_op(0, lin_center)

        def apply_linearized_burgers_op(operand):
            return (bound_op(0, lin_center + operand)
                    - lin_center_rhs)

        def apply_op(operand):
            return bound_op(0, operand)

        n = len(discr)
        op_mat = numpy.zeros((n, n), dtype=discr.default_scalar_type)

        from pytools import ProgressBar
        pb = ProgressBar("mat build", n)

        from hedge.tools import unit_vector
        mag = 1e-6
        for i in xrange(n):
            #x = (1/mag)*apply_linearized_burgers_op(
            x = (1/mag)*apply_op(
                    mag*unit_vector(n, i, dtype=discr.default_scalar_type))
            op_mat[:, i] = x
            pb.progress()
        pb.finished()

        import scipy.linalg as la
        eigval, eigvec = la.eig(op_mat)

        from matplotlib.pyplot import clf, plot, savefig, xlim, ylim, grid, title
        clf()
        plot(eigval.real, eigval.imag, "o")
        grid()
        #xlim([-1500,400])
        #ylim([-1000,1000])
        title("Spectrum of Poorly Linearized DG Burgers N=%d, $\mu$=%s" % (order, viscosity))
        savefig("burgers-eigval-%.2e.png" % viscosity)

        if False:
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
