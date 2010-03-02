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
import numpy.linalg as la
from hedge.mesh import TAG_ALL, TAG_NONE




def read_shape():
    shapes = []
    for l in open("points.txt"):
        if l.startswith("m"):
            continue

        shape = []
        last_pt = None
        for point in l.split():
            if point == "m":
                shapes.append(numpy.array(shape))
                shape = []
                continue

            pt = numpy.array([float(s) for s in point.split(",")])
            pt *= numpy.array([1,-1])

            if last_pt is not None:
                ignore = la.norm(pt) < 4
                pt += last_pt
            else:
                ignore = False

            if not ignore:
                shape.append(pt)
            last_pt = pt

        shapes.append(numpy.array(shape))

    return shapes




def make_mesh():
    def round_trip_connect(seq):
        result = []
        for i in range(len(seq)):
            result.append((i, (i+1)%len(seq)))
        return result

    shapes = read_shape()

    #from matplotlib.pyplot import plot,show
    #plot(shapes[0][:,0], shapes[0][:,1])
    #show()

    from meshpy.geometry import GeometryBuilder, Marker
    builder = GeometryBuilder()

    for shape in shapes:
        from meshpy.geometry import make_box
        points = shape
        facets = round_trip_connect(range(len(points)))
        builder.add_geometry(points=points, facets=facets,
                facet_markers=Marker.FIRST_USER_MARKER)

    points, facets, facet_markers = make_box((-200, -600), (400, -300))
    builder.add_geometry(points=points, facets=facets,
            facet_markers=facet_markers)

    from meshpy.triangle import MeshInfo, build
    mi = MeshInfo()
    builder.set(mi)
    holes = []
    for shape, sign, frac in zip(shapes, [1, 1, -1, 1, 1, 1], [0.5, 0, 0, 0, 0, 0]):
        avg = numpy.average(shape, axis=0)
        start_idx = int(frac*shape.shape[0])
        start = shape[start_idx]
        holes.append(start + sign*0.01*(avg-start))

    mi.set_holes(holes)

    mesh = build(mi,
            allow_boundary_steiner=True,
            generate_faces=True,
            max_volume=10)

    from meshpy.triangle import write_gnuplot_mesh
    write_gnuplot_mesh("mesh.dat", mesh)

    print len(mesh.elements), "elements"

    face_marker_to_tag = {
            Marker.FIRST_USER_MARKER: "dirichlet",
            Marker.MINUS_X: "radiation",
            Marker.PLUS_X: "radiation",
            Marker.MINUS_Y: "radiation",
            Marker.PLUS_Y: "radiation"
            }

    fvi2fm = mesh.face_vertex_indices_to_face_marker

    def bdry_tagger(fvi, el, fn, all_v):
        face_marker = fvi2fm[fvi]
        return [face_marker_to_tag[face_marker]]

    from hedge.mesh import make_conformal_mesh_ext
    vertices = numpy.asarray(mesh.points, dtype=float, order="C")
    from hedge.mesh.element import Triangle
    return make_conformal_mesh_ext(
            vertices,
            [Triangle(i, el_idx, vertices)
                for i, el_idx in enumerate(mesh.elements)],
            bdry_tagger)




def main(write_output=True, 
        dir_tag=TAG_NONE, neu_tag=TAG_NONE, rad_tag=TAG_ALL, 
        flux_type_arg="upwind", dtype=numpy.float64, debug=["cuda_no_plan"]):
    from pytools.stopwatch import Job
    from math import sin, cos, pi, exp, sqrt

    from hedge.backends import guess_run_context
    rcon = guess_run_context()

    mesh = make_mesh()

    if rcon.is_head_rank:
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    discr = rcon.make_discretization(mesh_data, order=4, debug=debug,
            default_scalar_type=dtype)
    from hedge.timestep import RK4TimeStepper
    stepper = RK4TimeStepper(dtype=dtype)

    from hedge.visualization import VtkVisualizer
    if write_output:
        vis = VtkVisualizer(discr, rcon, "fld")

    gauss_center = numpy.array([-100,-525])
    def source_u(x, el):
        x = x-gauss_center
        return exp(-numpy.dot(x, x)*0.05)

    from hedge.models.wave import StrongWaveOperator
    from hedge.mesh import TAG_ALL, TAG_NONE
    from hedge.data import \
            make_tdep_given, \
            TimeHarmonicGivenFunction, \
            TimeIntervalGivenFunction

    op = StrongWaveOperator(-1, discr.dimensions, 
            source_f=TimeIntervalGivenFunction(
                TimeHarmonicGivenFunction(
                    make_tdep_given(source_u), omega=10),
                0, 1),
            dirichlet_tag="dirichlet",
            neumann_tag=TAG_NONE,
            radiation_tag="radiation",
            flux_type=flux_type_arg
            )

    from hedge.tools import join_fields
    fields = join_fields(discr.volume_zeros(dtype=dtype),
            [discr.volume_zeros(dtype=dtype) for i in range(discr.dimensions)])

    # diagnostics setup -------------------------------------------------------
    from pytools.log import LogManager, \
            add_general_quantities, \
            add_simulation_quantities, \
            add_run_info

    if write_output:
        log_file_name = "wave.dat"
    else:
        log_file_name = None

    logmgr = LogManager(log_file_name, "w", rcon.communicator)
    add_run_info(logmgr)
    add_general_quantities(logmgr)
    add_simulation_quantities(logmgr)
    discr.add_instrumentation(logmgr)

    from pytools.log import IntervalTimer
    vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
    logmgr.add_quantity(vis_timer)
    stepper.add_instrumentation(logmgr)

    logmgr.add_watches(["step.max", "t_sim.max", "t_step.max"])

    # timestep loop -----------------------------------------------------------
    rhs = op.bind(discr)
    try:
        from hedge.timestep import times_and_steps
        step_it = times_and_steps(
                final_time=1000, logmgr=logmgr,
                max_dt_getter=lambda t: op.estimate_timestep(discr,
                    stepper=stepper, t=t, fields=fields))

        for step, t, dt in step_it:
            if step % 100 == 0 and write_output:
                visf = vis.make_file("fld-%04d" % step)

                vis.add_data(visf,
                        [
                            ("u", discr.convert_volume(fields[0], kind="numpy")),
                            ("v", discr.convert_volume(fields[1:], kind="numpy")), 
                        ],
                        time=t,
                        step=step)
                visf.close()

            fields = stepper(fields, t, dt, rhs)

    finally:
        if write_output:
            vis.close()

        logmgr.close()
        discr.close()




if __name__ == "__main__":
    main()
