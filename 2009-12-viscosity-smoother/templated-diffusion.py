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
from pytools import Record, memoize_method




class SmootherInfo(Record):
    pass




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




class TriBlobSmoother:
    def __init__(self, discr):
        self.discr = discr
        self.prepare()

    def make_brick(self):
        discr = self.discr

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

    def prepare(self):
        discr = self.discr
        brick, brick_node_num_to_el_info = self.make_brick()

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

        from pyrticle._internal import BoxFloat, BrickIterator
        from pytools import product

        # map (tgt_base_idx, source_idx) to el_vector
        el_vectors = {}
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

                for idx in BrickIterator(brick, brick.index_range(bbox)):
                    el_info = brick_node_num_to_el_info.get(brick.index(idx), set())
                    for tgt_el, tgt_base_idx in el_info:
                        if (tgt_base_idx, src_base_idx) in el_vectors:
                            continue

                        el_vector = numpy.empty(node_count, dtype=numpy.float64)
                        el_vectors[tgt_base_idx, src_base_idx] = el_vector

                        for node_idx in range(node_count):
                            bary_coords = global_to_bary(
                                    discr.nodes[tgt_base_idx+node_idx])
                            el_vector[node_idx] = product(
                                    ramp_sin(2*bc)**1.4 for bc in bary_coords)

        deletable_keys = []
        for key, vector in el_vectors.iteritems():
            if la.norm(vector) == 0:
                deletable_keys.append(key)

        for key in deletable_keys:
            del el_vectors[key]

        self.el_vectors = el_vectors

        print "overstore: %g" % (
                len(self.el_vectors) 
                / len(discr.mesh.elements))

    def __call__(self, u):
        result = self.discr.volume_zeros(dtype=u.dtype)

        for (tgt_idx, src_idx), vector in self.el_vectors.iteritems():
            result[tgt_idx:tgt_idx+len(vector)] += u[src_idx]*vector

        return result





def main():
    from hedge.backends import guess_run_context
    rcon = guess_run_context()

    if False:
        from hedge.mesh.generator import make_disk_mesh
        mesh = make_disk_mesh(r=0.5, max_area=1e-2)
    elif True:
        from hedge.mesh.generator import make_regular_rect_mesh
        mesh = make_regular_rect_mesh()
    else:
        from hedge.mesh.generator import make_uniform_1d_mesh
        mesh = make_uniform_1d_mesh(-3, 3, 20, periodic=True)

    el_values = {}

    from random import uniform

    def bumpy_f(x, el):
        try:
            return el_values[el.id]
        except KeyError:
            x = uniform(0, 1)
            if x < 1.4:
                result = 1
            else:
                result = 0
            el_values[el.id] = result
            return result

    discr = rcon.make_discretization(mesh, order=4)

    bumpy = discr.interpolate_volume_function(bumpy_f)

    if False:
        smoother = TemplatedSmoother(discr)
        smoothed = smoother(bumpy)
    elif False:
        p1_discr = rcon.make_discretization(discr.mesh, order=1)

        from hedge.discretization import Projector
        down_proj = Projector(discr, p1_discr)
        up_proj = Projector(p1_discr, discr)

        from hedge.mesh import TAG_NONE, TAG_ALL
        from hedge.models.diffusion import DiffusionOperator
        from hedge.tools.second_order import IPDGSecondDerivative
        op = DiffusionOperator(p1_discr.dimensions,
                dirichlet_tag=TAG_NONE, neumann_tag=TAG_ALL,
                scheme=IPDGSecondDerivative(10000))
        bound_op = op.bind(p1_discr)

        p1_accu = down_proj(bumpy)
        for i in range(10):
            laplace_result = bound_op(0, p1_accu)
            p1_accu += 0.3*la.norm(p1_accu)/la.norm(laplace_result)*laplace_result
        smoothed = up_proj(p1_accu)
    else:
        smoother = TriBlobSmoother(discr)
        smoothed = smoother(bumpy)

    vis_discr = rcon.make_discretization(mesh, order=30)

    from hedge.discretization import Projector
    vis_proj = Projector(discr, vis_discr)

    from hedge.visualization import SiloVisualizer
    vis = SiloVisualizer(vis_discr, rcon)
    visf = vis.make_file("bumpy")
    vis.add_data(visf, [ 
        ("bumpy", vis_proj(bumpy)),
        ("smoothed", vis_proj(smoothed)),
        ])
    visf.close()



if __name__ == "__main__":
    main()
