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
from hedge.tools import Reflection, Rotation




def main() :
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("--dim", default=3, type="int")
    parser.add_option("--single", action="store_true")
    parser.add_option("--no-vis", action="store_true")
    parser.add_option("--order", default=4, type="int")
    parser.add_option("--tol", default=1e-10, type="float")
    parser.add_option("--max-it", default=None, type="int")
    parser.add_option("--no-cg-progress", action="store_true")
    parser.add_option("--write-summary")
    parser.add_option("--no-diff-tensor", action="store_true")
    parser.add_option("--max-volume", default=0.0005, type="float")
    parser.add_option("--cpu", action="store_true")
    parser.add_option("-d", "--debug-flags", metavar="DEBUG_FLAG,DEBUG_FLAG")
    options, args = parser.parse_args()
    assert not args

    from hedge.data import GivenFunction, ConstantGivenFunction

    from hedge.backends import guess_run_context
    if options.cpu:
        rcon = guess_run_context(disable=set(["cuda"]))
    else:
        rcon = guess_run_context()

    def boundary_tagger(fvi, el, fn, points):
        from math import atan2, pi
        normal = el.face_normals[fn]
        if -10/180*pi < atan2(normal[1], normal[0]) < 10/180*pi:
            return ["neumann"]
        else:
            return ["dirichlet"]

    if options.dim == 2:
        if rcon.is_head_rank:
            from hedge.mesh import make_disk_mesh
            mesh = make_disk_mesh(r=0.5, boundary_tagger=boundary_tagger,
                    max_area=1e-2)
    elif options.dim == 3:
        if rcon.is_head_rank:
            from hedge.mesh import make_ball_mesh
            mesh = make_ball_mesh(max_volume=options.max_volume,
                    boundary_tagger=lambda fvi, el, fn, points: ["dirichlet"])
    else:
        raise RuntimeError, "bad number of dimensions"

    if rcon.is_head_rank:
        print "%d elements" % len(mesh.elements)
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    def dirichlet_bc(x, el):
        from math import sin
        return sin(10*x[0])

    def rhs_c(x, el):
        if la.norm(x) < 0.1:
            return 1000
        else:
            return 0

    def my_diff_tensor():
        result = numpy.eye(options.dim)
        result[0,0] = 0.1
        return result

    from hedge.pde import WeakPoissonOperator
    op = WeakPoissonOperator(options.dim, 
            diffusion_tensor= None if options.no_diff_tensor else 
            ConstantGivenFunction(my_diff_tensor()) ,
            dirichlet_tag="dirichlet",
            neumann_tag="neumann", 

            dirichlet_bc=GivenFunction(dirichlet_bc),
            neumann_bc=ConstantGivenFunction(-10),
            )

    kwargs = {}
    if not options.cpu:
        kwargs["tune_for"] = op.grad_op_template()

    debug_flags = [ ]
    if options.debug_flags:
        debug_flags.extend(options.debug_flags.split(","))
        
    discr = rcon.make_discretization(mesh_data, order=options.order,
            default_scalar_type=numpy.float32 if options.single else numpy.float64,
            debug=debug_flags,
            **kwargs)

    bound_op = op.bind(discr)

    conv_history = []

    def cg_debug(event, iterations, x, res, d, delta):
        if event == "it":
            conv_history.append(delta)
            if iterations % 50 == 0 and not options.no_cg_progress:
                print "it=%d delta=%g" % (iterations, delta)

    rhs_gf = GivenFunction(rhs_c)
    rhs_discr = rhs_gf.volume_interpolant(discr)
    rhs_prep = bound_op.prepare_rhs(rhs_gf)

    start_guess = discr.volume_zeros()
    # warm-up
    for i in range(3):
        resid = bound_op(start_guess) - rhs_prep
    start_resid = discr.nodewise_dot_product(resid, resid)**0.5

    from hedge.tools import parallel_cg, ConvergenceError
    from time import time
    start = time()

    max_iterations = options.max_it or bound_op.shape[0]//10
    print "max_it=%d" % max_iterations

    try:
        u = -parallel_cg(rcon, -bound_op, rhs_prep,
                debug_callback=cg_debug, tol=options.tol,
                dot=discr.nodewise_dot_product,
                x=start_guess,
                max_iterations=max_iterations)
        converged = True
    except ConvergenceError:
        print "NO CONVERGENCE!"
        converged = False
    elapsed = time()-start
    print "%g s, %d iterations, %g it/s" % (
            elapsed, len(conv_history), len(conv_history)/elapsed)

    if converged:
        resid = bound_op(u) - rhs_prep
        end_resid = discr.nodewise_dot_product(resid, resid)**0.5
    else:
        end_resid = 0

    if converged and not options.no_vis:
        from hedge.visualization import SiloVisualizer, VtkVisualizer
        vis = VtkVisualizer(discr, rcon)
        visf = vis.make_file("fld")
        vis.add_data(visf, [ 
            ("sol", discr.convert_volume(u, kind="numpy")), 
            ("rhs", discr.convert_volume(rhs_discr, kind="numpy")), 
            ("rhs_prep", discr.convert_volume(rhs_prep, kind="numpy")), 
            ])
        visf.close()

    if options.write_summary:
        txt = open(options.write_summary, "a")
        txt.write(repr({
            "order": options.order,
            "cpu": options.cpu,
            "single": options.single,
            "elapsed": elapsed,
            "element_count": len(mesh.elements),
            "start_resid": start_resid,
            "end_resid": end_resid,
            "tol": options.tol,
            "conv_history": conv_history,
            "no_diff_tensor": options.no_diff_tensor,
            "converged": converged,
            })+"\n")
        txt.close()

    discr.close()





if __name__ == "__main__":
    main()

