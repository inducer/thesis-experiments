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
from pytools import memoize_method




class WindowedPlaneWave:
    """See Jackson, 3rd ed. Section 7.1."""

    def __init__(self, amplitude, origin, epsilon, mu, v, omega, sigma, 
            dimensions=3):
        self.ctx = dict(
                amplitude=amplitude, origin=origin,
                espilon=epsilon, mu=mu, v=v,
                omega=omega, sigma=sigma)

        self.dimensions = dimensions

        self.nodes_cache = {}

    @memoize_method
    def exprs(self):
        from pymbolic import var
        from pymbolic.primitives import CommonSubexpression

        def make_vec(basename):
            from hedge.tools import make_obj_array
            return make_obj_array(
                    [var("%s%d" % (basename, i)) for i in range(self.dimensions)])

        amplitude, origin, x, v = [make_vec(n) for n in "amplitude origin x v".split()]
        epsilon, mu, omega, sigma = [var(f) for f in "epsilon mu omega sigma".split()]
        sin, cos, sqrt, exp = [var(f) for f in "sin cos sqrt exp".split()]

        c_inv = CommonSubexpression(sqrt(mu*epsilon))
        norm_v_squared = CommonSubexpression(numpy.dot(v, v))
        n = v/sqrt(norm_v_squared)
        k = v*CommonSubexpression(omega/norm_v_squared) 

        t = var("t")

        modulation = CommonSubexpression(
            sin(numpy.dot(k, x) - omega*t)
            * exp(-numpy.dot(v, x)**2/(2*sigma**2))
            )

        e = amplitude * modulation
        from hedge.tools import join_fields
        return join_fields(
                e,
                c_inv*numpy.cross(n, e),)

    def boundary_interpolant(self, t, discr, tag):
        try:
            nodes = self.nodes_cache[discr, tag]
        except KeyError:
            nodes = discr.boundary_to_gpu(
                    tag, discr.get_boundary(tag).nodes)
            self.nodes_cache[discr, tag] = nodes

        raise RuntimeError("none")




def main():
    from hedge.element import TetrahedralElement
    from hedge.mesh import make_ball_mesh, make_cylinder_mesh, make_box_mesh
    from hedge.visualization import SiloVisualizer
    from hedge.tools import to_obj_array
    from math import sqrt, pi
    from hedge.backends import guess_run_context

    rcon = guess_run_context()

    epsilon0 = 8.8541878176e-12 # C**2 / (N m**2)
    mu0 = 4*pi*1e-7 # N/A**2.
    epsilon = 1*epsilon0
    mu = 1*mu0

    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("--order", default=4, type="int")
    parser.add_option("--h", default=0.08, type="float")
    parser.add_option("--final-time", default=4e-10, type="float")
    parser.add_option("--vis-interval", default=0, type="int")
    parser.add_option("--steps", type="int")
    parser.add_option("--cpu", action="store_true")
    parser.add_option("-d", "--debug-flags", metavar="DEBUG_FLAG,DEBUG_FLAG")
    parser.add_option("--log-file", default="maxwell-%(order)s.dat")
    options, args = parser.parse_args()
    assert not args

    print "----------------------------------------------------------------"
    print "ORDER %d" % options.order
    print "----------------------------------------------------------------"

    pml_width = 0.25

    def make_mesh():
        from meshpy.geometry import GeometryBuilder, make_ball
        geob = GeometryBuilder()
        ball_points, ball_facets, ball_facet_hole_starts, _ = make_ball(0.5)
        geob.add_geometry(ball_points, ball_facets, ball_facet_hole_starts, 
                facet_markers=1)
        geob.wrap_in_box(pml_width)
        geob.wrap_in_box(pml_width)

        from meshpy.tet import MeshInfo, build
        mi = MeshInfo()
        geob.set(mi)
        mi.set_holes([geob.center()])
        built_mi = build(mi)

        fvi2fm = built_mi.face_vertex_indices_to_face_marker

        def boundary_tagger(fvi, el, fn):
            face_marker = fvi2fm[frozenset(fvi)]
            if face_marker == 1:
                return ["shell"]
            else:
                return ["outside"]

        from hedge.mesh import make_conformal_mesh
        return make_conformal_mesh(
            built_mi.points,
            built_mi.elements, 
            boundary_tagger)

    mesh = make_mesh()

    if rcon.is_head_rank:
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    ibc = WindowedPlaneWave(
        amplitude=numpy.array([1,0,0]),
        origin=numpy.array([0,0,0]),
        epsilon=epsilon,
        mu=mu,
        v=0.5/sqrt(mu*epsilon)*numpy.array([0,0,1]),
        omega=1e7,
        sigma=0.2)

    ibc.exprs()

    from hedge.pde import GedneyPMLMaxwellOperator
    op = GedneyPMLMaxwellOperator(epsilon, mu, 
            incident_tag="shell",
            pec_tag="outside",
            incident_bc=ibc,
            flux_type=1)

    debug_flags = []
    if options.debug_flags:
        debug_flags.extend(options.debug_flags.split(","))

    from hedge.backends.dynamic import Discretization as CPUDiscretization
    from hedge.backends.cuda import Discretization as GPUDiscretization
    if options.cpu:
        discr = CPUDiscretization(mesh, order=options.order, debug=debug_flags)

        def to_gpu(x):
            return x
        def from_gpu(x):
            return x
        cpu_discr = discr
    else:
        discr = GPUDiscretization(mesh, order=options.order, debug=debug_flags,
                tune_for=op.op_template())

        def to_gpu(x):
            return discr.volume_to_gpu(x)
        def from_gpu(x):
            return discr.volume_from_gpu(x)

        cpu_discr = CPUDiscretization(mesh, order=options.order)


    if options.vis_interval:
        #vis = VtkVisualizer(discr, rcon, "em-%d" % options.order)
        vis = SiloVisualizer(discr, rcon)

    dt = discr.dt_factor(op.max_eigenvalue())
    if options.steps is None:
        nsteps = int(options.final_time/dt)+1
        dt = options.final_time/nsteps
    else:
        nsteps = options.steps

    t = 0

    # diagnostics setup ---------------------------------------------------
    from pytools.log import LogManager, add_general_quantities, \
            add_simulation_quantities, add_run_info

    logmgr = LogManager(options.log_file % {"order": options.order}, 
            "w", rcon.communicator)
    add_run_info(logmgr)
    add_general_quantities(logmgr)
    add_simulation_quantities(logmgr, dt)
    discr.add_instrumentation(logmgr)

    if isinstance(discr, CPUDiscretization):
        from hedge.timestep import RK4TimeStepper
    else:
        from hedge.backends.cuda.tools import RK4TimeStepper

    stepper = RK4TimeStepper()
    stepper.add_instrumentation(logmgr)

    if isinstance(discr, CPUDiscretization):
        logmgr.add_watches(["step.max", "t_sim.max", "t_step.max",
            ("flops/s", "(n_flops_gather+n_flops_lift+n_flops_mass+n_flops_diff)"
            "/(t_gather+t_lift+t_mass+t_diff)")
            ])
    else:
        logmgr.add_watches(["step.max", "t_sim.max", "t_step.max", 
            ("t_compute", "t_diff+t_gather+t_lift+t_rk4+t_vector_math"),
            ("flops/s", "(n_flops_gather+n_flops_lift+n_flops_mass+n_flops_diff+n_flops_vector_math+n_flops_rk4)"
            "/(t_gather+t_lift+t_mass+t_diff+t_vector_math+t_rk4)")
            ])

    logmgr.set_constant("h", options.h)
    logmgr.set_constant("is_cpu", options.cpu)

    # timestep loop -------------------------------------------------------

    rhs = op.bind(discr, sigma=to_gpu(op.sigma_from_width(discr, pml_width)))

    fields = op.assemble_ehdb(discr=discr)

    for step in range(nsteps):
        logmgr.tick()

        if options.vis_interval and step % options.vis_interval == 0:
            e, h = op.split_eh(fields[0])
            visf = vis.make_file("em-%d-%04d" % (options.order, step))
            vis.add_data(visf,
                    [ ("e", 
                        #e
                        from_gpu(e)
                        ), 
                        ("h", 
                            #h
                            from_gpu(h)
                            ), 
                        ],
                    time=t, step=step
                    )
            visf.close()

        fields = stepper(fields, t, dt, rhs)
        t += dt

    logmgr.close()

if __name__ == "__main__":
    main()
