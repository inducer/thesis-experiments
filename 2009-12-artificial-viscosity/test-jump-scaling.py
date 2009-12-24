from __future__ import division
import numpy
import numpy.linalg as la
from math import sin, cos, pi, sqrt




def main():
    from hedge.backends import guess_run_context
    rcon = guess_run_context()

    order = 4
    n_elements = 20

    class case:
        a = -pi
        b = pi
    if rcon.is_head_rank:
        if False:
            from hedge.mesh.generator import make_uniform_1d_mesh
            mesh = make_uniform_1d_mesh(case.a, case.b, 20, periodic=True)
        else:
            def needs_refinement(vertices, area):
                x =  sum(numpy.array(v) for v in vertices)/3

                max_area_volume = 0.3e-2 + 0.07*(0.05*x[1]**2 + 0.3*min(x[0]+1,0)**2)

                return bool(area > 10*max_area_volume)

            extent_y = 4
            dx = (case.b-case.a)/n_elements
            subdiv = (n_elements, int(1+extent_y//dx))
            from pytools import product

            from hedge.mesh.generator import make_rect_mesh
            mesh = make_rect_mesh((case.a, -2), (case.b, 2), 
                    periodicity=(True, True), 
                    subdivisions=subdiv,
                    refine_func=needs_refinement
                    )

    discr = rcon.make_discretization(mesh, order=order)
    vis_discr = rcon.make_discretization(mesh, order=10)

    from hedge.discretization import Projector
    vis_proj = Projector(discr, vis_discr)

    from hedge.visualization import VtkVisualizer, SiloVisualizer
    #vis = SiloVisualizer(vis_discr, rcon)
    vis = VtkVisualizer(vis_discr, rcon, "fld")

    # function ----------------------------------------------------------------
    el_values = {}

    from random import uniform, seed

    seed(1)

    def bumpy_f(x, el):
        try:
            return el_values[el.id]
        except KeyError:
            if False:
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

    bumpy = discr.interpolate_volume_function(bumpy_f)

    # operator setup ----------------------------------------------------------
    from hedge.optemplate.operators import (FilterOperator,
            MassOperator, OnesOperator, InverseVandermondeOperator,
            InverseMassOperator)
    from hedge.optemplate.primitives import Field
    from hedge.optemplate.tools import get_flux_operator
    from hedge.tools.symbolic import make_common_subexpression as cse
    from pymbolic.primitives import Variable

    from hedge.flux import (
            FluxScalarPlaceholder, ElementOrder,
            ElementJacobian, FaceJacobian)

    u_flux = FluxScalarPlaceholder(0)

    u = Field("u")
    jump_part = InverseMassOperator()(
            get_flux_operator(
                ElementJacobian()/(ElementOrder()**2 * FaceJacobian())
                    *(u_flux.ext - u_flux.int))(u))

    bound_jump_part = discr.compile(jump_part)

    jump_bumpy = bound_jump_part(u=bumpy)

    visf = vis.make_file("bumpy")
    vis.add_data(visf, [ 
        ("bumpy", vis_proj(bumpy)), 
        ("jump_bumpy", vis_proj(jump_bumpy)), 
        ])
    visf.close()




if __name__ == "__main__":
    main()
