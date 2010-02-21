import numpy




class AirplaneProblem(object):
    def __init__(self, mesh_name=None, swizzle=None, el_volume=1):
        self.mesh_name = mesh_name
        self.swizzle = swizzle
        self.el_volume = el_volume

        from pytools import add_python_path_relative_to_script
        add_python_path_relative_to_script("../../hedge/examples/gas_dynamics")

        from gas_dynamics_initials import UniformMachFlow
        #self.flow = UniformMachFlow(mach=0.84, reynolds=numpy.inf)
        self.flow = UniformMachFlow(mach=0.1, reynolds=numpy.inf)

        self.final_time = 10

    @property
    def gamma(self):
        return self.flow.gamma

    def get_initial_data(self):
        return self.flow

    def make_mesh(self):
        from meshpy.geometry import GeometryBuilder, make_ball, Marker
        geob = GeometryBuilder()
        
        obstacle_marker = Marker.FIRST_USER_MARKER

        if self.mesh_name is not None:
            from meshpy.ply import parse_ply
            data = parse_ply(self.mesh_name)
            geob.add_geometry(
                    points=[pt[:3] for pt in data["vertex"].data],
                    facets=[fd[0] for fd in data["face"].data],
                    facet_markers=obstacle_marker)
            free_space_factor = 1.5
        else:
            ball_points, ball_facets, ball_facet_hole_starts, _ = make_ball(0.5,
                    subdivisions=15)
            geob.add_geometry(ball_points, ball_facets, ball_facet_hole_starts, 
                    facet_markers=obstacle_marker)
            free_space_factor = 1

        if self.swizzle is not None:
            mtx = make_swizzle_matrix(self.swizzle)
            geob.apply_transform(lambda x: numpy.dot(mtx, numpy.array(x)))

        bbox_min, bbox_max = geob.bounding_box()
        geob.wrap_in_box(free_space_factor)

        from meshpy.tet import MeshInfo, build
        mi = MeshInfo()
        geob.set(mi)
        mi.set_holes([geob.center()])
        built_mi = build(mi, max_volume=self.el_volume)

        #built_mi.write_vtk("airplane.vtk")

        fvi2fm = built_mi.face_vertex_indices_to_face_marker

        flow_dir = self.flow.direction_vector(3)

        def boundary_tagger(fvi, el, face_nr, points):
            face_marker = fvi2fm[frozenset(fvi)]
            if face_marker == obstacle_marker:
                return ["no_slip"]
            elif numpy.dot(el.face_normals[face_nr], flow_dir) >= 0:
                return ["outflow"]
            else:
                return ["inflow"]

        from hedge.mesh import make_conformal_mesh_ext
        vertices = numpy.asarray(built_mi.points, 
                dtype=float, order="C")
        from hedge.mesh.element import Tetrahedron
        return make_conformal_mesh_ext(
                vertices,
                [Tetrahedron(i, el_idx, vertices)
                    for i, el_idx in enumerate(built_mi.elements)],
                boundary_tagger)

    def get_operator(self, setup):
        from hedge.models.gas_dynamics import GasDynamicsOperator
        from hedge.second_order import IPDGSecondDerivative
        from hedge.mesh import TAG_ALL, TAG_NONE

        return GasDynamicsOperator(dimensions=3,
                gamma=self.flow.gamma,
                mu=self.flow.mu,

                bc_inflow=self.flow,
                bc_outflow=self.flow,
                bc_noslip=self.flow,

                second_order_scheme=IPDGSecondDerivative(
                    stab_coefficient=setup.stab_coefficient),
                #second_order_scheme=CentralSecondDerivative(),

                supersonic_inflow_tag=TAG_NONE,
                supersonic_outflow_tag=TAG_NONE,
                inflow_tag="inflow",
                outflow_tag="outflow",
                noslip_tag="no_slip")

