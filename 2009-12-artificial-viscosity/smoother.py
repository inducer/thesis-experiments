from __future__ import division

import numpy
import numpy.linalg as la




class TemplatedSmoother:
    def __init__(self, discr):
        self.discr = discr

        self.core = discr.volume_zeros()

        for eg in self.discr.element_groups:
            ldis = eg.local_discretization

            self.core_element, self.core_faces = \
                    self.get_triangle_smoother_elements(ldis.order)
            for rng in eg.ranges:
                self.core[rng] = self.core_element

    def get_triangle_smoother_elements(self, order):
        #     5
        #    /3\       s
        #   3---4     ^
        #  /0\1/2\   /
        # 0---1---2  ---> r

        from math import sqrt
        r = numpy.array([1,0], dtype=numpy.float64)
        s = numpy.array([0.5,sqrt(3)/2], dtype=numpy.float64)

        from hedge.mesh.element import Triangle
        from hedge.mesh import make_conformal_mesh_ext
        points = numpy.array(
                [0*r, 1*r, 2*r, s, r+s, 2*s], dtype=numpy.float64)
        mesh = make_conformal_mesh_ext(
                points=points,
                elements=[
                    Triangle(0, [0,1,3], points),
                    Triangle(1, [3,1,4], points),
                    Triangle(2, [1,2,4], points),
                    Triangle(3, [3,4,5], points),
                    ])

        discr = self.discr.run_context.make_discretization(mesh, order=order)
        from hedge.models.poisson import PoissonOperator
        from hedge.tools.second_order import IPDGSecondDerivative
        from hedge.mesh import TAG_NONE, TAG_ALL
        op = PoissonOperator(discr.dimensions,
                dirichlet_tag=TAG_ALL, neumann_tag=TAG_NONE, 
                scheme=IPDGSecondDerivative(1000))
        bound_op = op.bind(discr)

        def rhs(x, el):
            if el.id == 1:
                return -1
            else:
                return 0

        from hedge.iterative import parallel_cg
        u = -parallel_cg(self.discr.run_context, -bound_op, 
                bound_op.prepare_rhs(discr.interpolate_volume_function(rhs)), 
                tol=5e-8, dot=discr.nodewise_dot_product)

        u = u/discr.nodewise_max(u)

        if False:
            vis_discr = rcon.make_discretization(mesh, order=30)

            from hedge.discretization import Projector
            vis_proj = Projector(discr, vis_discr)

            from hedge.visualization import SiloVisualizer, VtkVisualizer
            vis = SiloVisualizer(vis_discr, self.discr.run_context)
            visf = vis.make_file("template")
            vis.add_data(visf, [ ("sol", 
                vis_proj(discr.convert_volume(u, kind="numpy"))), ])
            visf.close()

        eg, = discr.element_groups

        return u[eg.ranges[1]], [
                u[eg.ranges[3]],
                u[eg.ranges[0]],
                u[eg.ranges[2]],
                ]

    def __call__(self, u):
        result = u*self.core

        for fg in self.discr.face_groups:
            np = fg.ldis_loc.node_count()
            for fp in fg.face_pairs:
                for fa, fb in [(fp.int_side, fp.ext_side), (fp.ext_side, fp.int_side)]:
                    ebi = fa.el_base_index
                    result[ebi:ebi+np] += \
                            u[fb.el_base_index]*self.core_faces[fa.face_id]

        from hedge.mesh import TAG_NONE, TAG_ALL
        from hedge.models.diffusion import DiffusionOperator
        op = DiffusionOperator(self.discr.dimensions,
                dirichlet_tag=TAG_NONE, neumann_tag=TAG_ALL)
        bound_op = op.bind(self.discr)

        #for i in range(5):
            #laplace_result = bound_op(0, result)
            #result += 0.1*la.norm(result)/la.norm(laplace_result)*laplace_result

        return result




# all ramps centered around 0, roughly ramping in (-1, 1)
def ramp_bruno(t):
    # http://www.acm.caltech.edu/%7Ebruno/hyde_bruno_3d_jcp.pdf
    x = (t+1)/2
    if x <= 0:
        return 0
    elif x >= 1:
        return 1
    else:
        from math import exp
        return 1-exp(2*exp(-1/x)/(x-1))

def ramp_sin(t):
    if t <= -1:
        return 0
    elif t >= 1:
        return 1
    else:
        from math import sin, pi
        return 0.5+0.5*sin(t*pi/2)




def ramp_cubic(t):
    if t <= -1:
        return 0
    elif t >= 1:
        return 1
    else:
        from math import sin, pi
        return -(t-2)*(t+1)**2/4




class TriBlobSmoother(object):
    def __init__(self, ramp=ramp_cubic, scaling=2, use_max=False):
        self.ramp = ramp
        self.scaling = scaling
        self.use_max = use_max

    def __str__(self):
        return "%s(ramp=%s, scaling=%r, use_max=%s)" % (
                type(self).__name__,
                self.ramp.__name__,
                self.scaling,
                self.use_max)

    def make_brick(self, discr):
        from hedge.discretization import ones_on_volume
        mesh_volume = discr.integral(ones_on_volume(discr))
        dx = (mesh_volume / len(discr))**(1/discr.dimensions)

        mesh = discr.mesh
        bbox_min, bbox_max = mesh.bounding_box()
        bbox_size_before = bbox_max-bbox_min

        bbox_min -= 1e-3*bbox_size_before
        bbox_max += 1e-3*bbox_size_before

        bbox_size = bbox_max-bbox_min
        dims = numpy.asarray(bbox_size/dx, dtype=numpy.int32)

        from pyrticle._internal import Brick, BoxFloat
        brick = Brick(0, 0, bbox_size/dims, bbox_min, dims)

        brick_node_num_to_el_info = {}

        for eg in discr.element_groups:
            ldis = eg.local_discretization

            for el in eg.members:
                el_bbox = BoxFloat(*el.bounding_box(discr.mesh.points))
                el_slice = discr.find_el_range(el.id)

                for node_num in range(el_slice.start, el_slice.stop):
                    try:
                        cell_number = brick.which_cell(
                                    discr.nodes[node_num])
                    except ValueError:
                        pass
                    else:
                        brick_node_num_to_el_info.setdefault(
                                brick.index(cell_number), set()).add(
                                        (el, el_slice.start))

        return brick, brick_node_num_to_el_info

    def prepare(self, discr, mod, ramp, scaling, exponent=1):
        brick, brick_node_num_to_el_info = self.make_brick(discr)

        from pyrticle._internal import BoxFloat, BrickIterator
        from pytools import product

        # map (tgt_base_idx, source_idx) to el_vector
        siv = mod.SourceInfoVec()
        indices = []

        vertices = discr.mesh.points
        for eg in discr.element_groups:
            ldis = eg.local_discretization
            node_count = ldis.node_count()

            for src_el, src_el_rng in zip(eg.members, eg.ranges):
                src_base_idx = src_el_rng.start

                bbox_min, bbox_max = src_el.bounding_box(vertices)
                bbox_size = bbox_max-bbox_min
                bbox_min = bbox_min - bbox_size
                bbox_max = bbox_max + bbox_size

                global_to_bary = ldis.unit_to_barycentric.post_compose(
                        src_el.inverse_map);

                bbox = BoxFloat(bbox_min, bbox_max).intersect(brick.bounding_box())

                si = mod.SourceInfo()

                import pyublas
                si.global_to_bary_mat = pyublas.why_not(global_to_bary.matrix, matrix=True,
                        dtype=numpy.float64)
                si.global_to_bary_vec = global_to_bary.vector
                si.source_node_number = src_el_rng.start
                si.dest_descriptor_start_index = len(indices)

                checked_els = set()

                for idx in BrickIterator(brick, brick.index_range(bbox)):
                    el_info = brick_node_num_to_el_info.get(brick.index(idx), set())
                    for tgt_el, tgt_base_idx in el_info:
                        if tgt_el in checked_els:
                            continue
                        checked_els.add(tgt_el)

                        el_vector = numpy.empty(node_count, dtype=numpy.float64)
                        for node_idx in range(node_count):
                            bary_coords = global_to_bary(
                                    discr.nodes[tgt_base_idx+node_idx])
                            el_vector[node_idx] = product(
                                    ramp(scaling*bc+1)**exponent for bc in bary_coords)

                        if la.norm(el_vector) != 0:
                            indices.append(tgt_base_idx)
                            indices.append(tgt_base_idx+node_count)

                si.dest_descriptor_end_index = len(indices)

                siv.append(si)

        tgt_indices = numpy.array(indices, dtype=numpy.uint32)
        return siv, tgt_indices

    def make_codepy_module(self, toolchain, dtype, dimensions):
        from codepy.libraries import add_codepy
        toolchain = toolchain.copy()
        add_codepy(toolchain)
        toolchain.enable_debugging()

        from codepy.cgen import (Struct, Value, Include, Statement,
                Typedef, FunctionBody, FunctionDeclaration, Block, Const,
                Reference, Line, POD, Initializer, CustomLoop, If, For)
        S = Statement

        from codepy.bpl import BoostPythonModule
        mod = BoostPythonModule()

        mod.add_to_preamble([
            Include("vector"),
            Include("hedge/base.hpp"),
            Include("boost/foreach.hpp"),
            Include("boost/numeric/ublas/io.hpp"),
            ])

        mod.add_to_module([
            S("namespace ublas = boost::numeric::ublas"),
            S("using namespace hedge"),
            S("using namespace pyublas"),
            Line(),
            Typedef(POD(dtype, "value_type")),
            Line(),
            Initializer(Const(Value("npy_uint", "dim")),
                dimensions),
            Line(),
            ])

        mod.add_struct(Struct("source_info", [
            Value("ublas::matrix<double>", "global_to_bary_mat"),
            Value("ublas::bounded_vector<double, dim+1>", "global_to_bary_vec"),
            Value("node_number_t", "source_node_number"),
            Value("npy_uint", "dest_descriptor_start_index"),
            Value("npy_uint", "dest_descriptor_end_index"),
            ]), py_name="SourceInfo", by_value_members=[
                "global_to_bary_mat",
                "global_to_bary_vec"])

        mod.add_to_module([
            Line(),
            Typedef(Value("std::vector<source_info>", "source_info_vec")),
            Line(),
            ])
        mod.expose_vector_type("source_info_vec", "SourceInfoVec")

        mod.add_function(FunctionBody(
            FunctionDeclaration(Value("value_type", "ramp_cubic"), [
                Value("value_type", "t")]),
            Block([
                If("t<=-1", S("return 0")),
                If("t>=1", S("return 1")),
                Initializer(
                    Value("value_type", "tp1"), "t+1"),
                S("return -(t-2)*tp1*tp1/4")
                ])))

        mod.add_to_module([ Line(), ])

        mod.add_function(FunctionBody(
            FunctionDeclaration(Value("void", "compose_smoother"), [
                Const(Reference(Value("source_info_vec", "siv"))),
                Const(Value("numpy_array<npy_uint>", "indices")),
                Value("numpy_vector<double>", "nodes"),
                Value("numpy_array<value_type>", "result"),
                Const(Value("numpy_array<value_type>", "unsmoothed")),
                Const(Value("value_type", "scaling")),
                ]),
            Block([
                Typedef(Value("numpy_array<value_type>::iterator", 
                    "it_type")),
                Typedef(Value("numpy_array<value_type>::const_iterator", 
                    "cit_type")),
                Typedef(Value("numpy_array<npy_uint>::const_iterator", 
                    "idx_it_type")),
                Line(),
                Initializer(Value("it_type", "result_it"), 
                    "result.begin()"),
                Initializer(Value("idx_it_type", "indices_it"), 
                    "indices.begin()"),
                Initializer(Value("cit_type", "unsmoothed_it"), 
                    "unsmoothed.begin()"),
                Line(),
                CustomLoop(
                    "BOOST_FOREACH(const source_info &si, siv)",
                    Block([
                        Initializer(
                            Value("value_type", "src_val"),
                            "unsmoothed_it[si.source_node_number]"),
                        If("src_val == 0", S("continue")),
                        Line(),
                        For(
                            "npy_uint idx = si.dest_descriptor_start_index",
                            "idx < si.dest_descriptor_end_index",
                            "idx += 2",
                            Block([
                                Initializer(Value(
                                    "node_number_t", "start_node"),
                                    "indices_it[idx]"),
                                Initializer(Value(
                                    "node_number_t", "end_node"),
                                    "indices_it[idx+1]"),
                                For(
                                    "node_number_t node_nr = start_node",
                                    "node_nr < end_node",
                                    "++node_nr",
                                    Block([
                                        Initializer(
                                            Value("bounded_vector<double, dim>", "node"),
                                                "subrange(nodes, dim*node_nr, "
                                                "dim*(node_nr+1))"),
                                        Initializer(
                                            Value("bounded_vector<double, dim+1>", "bary"),
                                            "prod(si.global_to_bary_mat, node)"
                                            "+si.global_to_bary_vec"),

                                        Initializer(Value("value_type", "nodeval"), 1),
                                        For("unsigned i = 0", "i < dim+1", "++i",
                                            S("nodeval *= ramp_cubic(scaling*bary[i]+1)")),

                                        S("result_it[node_nr] = std::max("
                                            "src_val*nodeval, result_it[node_nr])")
                                        if self.use_max else
                                        S("result_it[node_nr] += src_val*nodeval"),
                                        ])
                                    )
                                ])
                            )
                        ])
                    )
                ])))

        return mod.compile(toolchain)

    def bind(self, discr):
        mod = self.make_codepy_module(
                discr.toolchain, discr.default_scalar_type,
                discr.dimensions)
        source_info_vec, tgt_indices = self.prepare(
                discr, mod, self.ramp, self.scaling)

        def do(u):
            result = discr.volume_zeros(dtype=u.dtype)

            mod.compose_smoother(
                    source_info_vec,
                    tgt_indices,
                    discr.nodes,
                    result,
                    u,
                    self.scaling)

            return result

        return do
