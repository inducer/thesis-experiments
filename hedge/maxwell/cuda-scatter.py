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




def get_mesquite_tet_points(points, elements):
    in_name = "_mesquite_in.vtk"
    out_name = "_mesquite_out.vtk"

    def generate_faces():
        for v1, v2, v3, v4 in elements:
            yield frozenset([v1,v2,v3])
            yield frozenset([v1,v2,v4])
            yield frozenset([v1,v3,v4])
            yield frozenset([v2,v3,v4])

    face_histogram = {}

    for f in generate_faces():
        face_histogram[f] = face_histogram.get(f, 0) + 1

    boundary_vertices = set()
    for face, count in face_histogram.iteritems():
        if count == 1:
            # a boundary face
            boundary_vertices |= set(face)

    is_boundary = [1 if i in boundary_vertices else 0
            for i in range(len(points))]

    from pyvtk import VtkData, UnstructuredGrid, PointData, Scalars
    vtkelements = VtkData(
        UnstructuredGrid(
          numpy.array(points), 
          tetra=numpy.array(elements)),
        "Mesh",
        PointData(
            Scalars(numpy.array(is_boundary), name="fixed" )))
    vtkelements.tofile(in_name)

    from os import system, unlink
    if system("mesquite %s %s" % (in_name, out_name)):
        from warnings import warn
        warn("mesquite invocation failed")
        return points

    result = VtkData(out_name).structure.points
    unlink(in_name)
    unlink(out_name)
    return result




def make_swizzle_matrix(spec):
    axes = ["x", "y", "z"]

    mapping = dict((axis, axis) for axis in axes)
    for one_spec in spec.split(","):
        import_axis, final_axis = one_spec.split(":")
        mapping[import_axis] = final_axis

    assert set(mapping.keys()) == set(axes), \
            "axis mapping not complete"
    assert set(axis.lstrip("-") for axis in mapping.itervalues()) == set(axes), \
            "Axis mapping not onto"

    n = len(axes)
    result = numpy.zeros((n, n), dtype=int)

    for imp_axis, final_axis in mapping.iteritems():
        imp_axis = axes.index(imp_axis)

        sign = 1
        while final_axis.startswith("-"):
            sign *= -1
            final_axis = final_axis[1:]
        final_axis = axes.index(final_axis)

        result[final_axis, imp_axis] = sign

    return result




class WindowedPlaneWave:
    """See Jackson, 3rd ed. Section 7.1."""

    def __init__(self, amplitude, origin, epsilon, mu, v, omega, sigma, 
            dimensions=3):

        self.amplitude = amplitude
        self.origin = origin
        self.epsilon = epsilon
        self.mu = mu
        self.v = v
        self.omega = omega
        self.sigma = sigma
        self.sigma = sigma

        self.dimensions = dimensions

        self.nodes_cache = {}

    @memoize_method
    def make_func(self, discr):
        from pymbolic import var

        def make_vec(basename):
            from hedge.tools import make_obj_array
            return make_obj_array(
                    [var("%s%d" % (basename, i)) for i in range(self.dimensions)])

        x = make_vec("x")
        t = var("t")

        epsilon = self.epsilon
        mu = self.mu
        sigma = self.sigma
        omega = self.omega

        v = self.v
        n = v/la.norm(v)
        k = v*self.omega/numpy.dot(v,v)
        amplitude = self.amplitude
        origin = self.origin
        print "k", k

        sin, cos, sqrt, exp = [var(f) for f in "sin cos sqrt exp".split()]

        c_inv = sqrt(mu*epsilon)

        from pymbolic.primitives import CommonSubexpression
        modulation = CommonSubexpression(
            cos(numpy.dot(k, x) - omega*t)
            * 
            exp(-numpy.dot(n, x-v*t-origin)**2/(2*sigma**2))
            )

        e = amplitude * modulation

        from hedge.backends.cuda.vector_expr import CompiledVectorExpression
        from hedge.tools import join_fields

        def type_getter(expr):
            from pymbolic import get_dependencies
            from pymbolic.primitives import Variable
            deps = get_dependencies(expr, composite_leaves=False)
            var, = deps
            assert isinstance(var, Variable)
            return var.name.startswith("x"), discr.default_scalar_type, 

        return CompiledVectorExpression(
                join_fields(e, c_inv*numpy.cross(n, e)),
                type_getter, discr.default_scalar_type,
                allocator=discr.pool.allocate)

    def __call__(self, t, x, discr):
        ctx = {"t": t}
        for i, xi in enumerate(x):
            ctx["x%d" % i] = xi

        from pymbolic.primitives import Variable
        def lookup_value(expr):
            assert isinstance(expr, Variable)
            return ctx[expr.name]

        return self.make_func(discr)(lookup_value)

    def volume_interpolant(self, t, discr):
        try:
            nodes = self.nodes_cache[discr]
        except KeyError:
            from hedge.tools import make_obj_array
            nodes = discr.volume_to_gpu(
                    make_obj_array(discr.nodes.T.astype(
                        discr.default_scalar_type)))
            self.nodes_cache[discr] = nodes

        return self(t, nodes, discr)


    def boundary_interpolant(self, t, discr, tag):
        try:
            nodes = self.nodes_cache[discr, tag]
        except KeyError:
            from hedge.tools import make_obj_array
            bnodes = discr.get_boundary(tag).nodes
            assert len(bnodes), "no scatterer boundary--that's kind of bad"
            nodes = discr.boundary_to_gpu(
                    tag, make_obj_array(bnodes.T.astype(discr.default_scalar_type)))
            self.nodes_cache[discr, tag] = nodes

        return self(t, nodes, discr)




def main():
    from hedge.element import TetrahedralElement
    from hedge.mesh import make_ball_mesh, make_cylinder_mesh, make_box_mesh
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
    parser.add_option("--vis-interval", default=0, type="int")
    parser.add_option("--steps", type="int")
    parser.add_option("--cpu", action="store_true")
    parser.add_option("--pml_factor", default=0.05, type="float")
    parser.add_option("--swizzle", metavar="FROM:TO,FROM:TO")
    parser.add_option("-d", "--debug-flags", metavar="DEBUG_FLAG,DEBUG_FLAG")
    parser.add_option("--log-file", default="maxwell-%(order)s.dat")
    options, args = parser.parse_args()

    print "----------------------------------------------------------------"
    print "ORDER %d" % options.order
    print "----------------------------------------------------------------"

    def make_mesh(args):
        from meshpy.geometry import GeometryBuilder, make_ball
        geob = GeometryBuilder()
        
        if args:
            from meshpy.ply import parse_ply
            data = parse_ply(args[0])
            geob.add_geometry(
                    points=[pt[:3] for pt in data["vertex"].data],
                    facets=[fd[0] for fd in data["face"].data],
                    facet_markers=1)
        else:
            ball_points, ball_facets, ball_facet_hole_starts, _ = make_ball(0.5,
                    subdivisions=25)
            geob.add_geometry(ball_points, ball_facets, ball_facet_hole_starts, 
                    facet_markers=1)

        if options.swizzle is not None:
            mtx = make_swizzle_matrix(options.swizzle)
            geob.apply_transform(lambda x: numpy.dot(mtx, numpy.array(x)))

        bbox_min, bbox_max = geob.bounding_box()
        pml_width = la.norm(bbox_max-bbox_min)*options.pml_factor
        geob.wrap_in_box(pml_width)
        geob.wrap_in_box(pml_width)

        from meshpy.tet import MeshInfo, build
        mi = MeshInfo()
        geob.set(mi)
        mi.set_holes([geob.center()])
        built_mi = build(mi, max_volume=options.h**3/6)

        print "%d elements" % len(built_mi.elements)

        fvi2fm = built_mi.face_vertex_indices_to_face_marker

        def boundary_tagger(fvi, el, fn):
            face_marker = fvi2fm[frozenset(fvi)]
            if face_marker == 1:
                return ["scatterer"]
            else:
                return ["outside"]

        from hedge.mesh import make_conformal_mesh
        return make_conformal_mesh(
                get_mesquite_tet_points(built_mi.points, built_mi.elements),
                #built_mi.points,
                built_mi.elements, 
                boundary_tagger), pml_width

    mesh, pml_width = make_mesh(args)

    mesh_min, mesh_max = mesh.bounding_box()
    mesh_size = mesh_max- mesh_min

    if rcon.is_head_rank:
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    ibc = WindowedPlaneWave(
        amplitude=numpy.array([1,0,0]),
        origin=numpy.array([0,0,mesh_min[2]]),
        epsilon=epsilon,
        mu=mu,
        v=0.99/sqrt(mu*epsilon)*numpy.array([0,0,1]),
        omega=0.7e9*2*pi,
        sigma=1)

    final_time = mesh_size[2]/la.norm(ibc.v)

    from hedge.pde import GedneyPMLMaxwellOperator
    op = GedneyPMLMaxwellOperator(epsilon, mu, 
            incident_tag="scatterer",
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
        from hedge.visualization import VtkVisualizer, SiloVisualizer
        vis = VtkVisualizer(discr, rcon, "em-%d" % options.order,
                compressor="zlib")
        #vis = SiloVisualizer(discr, rcon)

    dt = discr.dt_factor(op.max_eigenvalue())
    if options.steps is None:
        nsteps = int(final_time/dt)+1
        dt = final_time/nsteps
    else:
        nsteps = options.steps

    print "final time: %g - dt:%g - steps: %d" % (final_time, dt, nsteps)

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

    for step in xrange(nsteps):
        logmgr.tick()

        if options.vis_interval and step % options.vis_interval == 0:
            e, h = op.split_eh(fields)
            visf = vis.make_file("em-%d-%08d" % (options.order, step))
            incident_e, incident_h = op.split_eh(ibc.volume_interpolant(t, discr))
            vis.add_data(visf,
                    [ 
                        ("e", from_gpu(e)), 
                        ("h", from_gpu(h)), 
                        ("inc_e", from_gpu(incident_e)), 
                        ("inc_h", from_gpu(incident_h)), 
                        ],
                    time=t, step=step
                    )
            visf.close()

        fields = stepper(fields, t, dt, rhs)
        t += dt

    logmgr.close()

if __name__ == "__main__":
    main()
