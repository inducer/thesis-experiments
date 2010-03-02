from __future__ import division, with_statement
import numpy
import numpy.linalg as la
from pytools.obj_array import make_obj_array




class SodProblem(object):
    final_time = 0.25
    is_periodic = False

    def __init__(self, **kwargs):
        from sod import SodProblem
        self.sod = SodProblem(**kwargs)

    def make_exact_func(self, t):
        if t == 0:
            return self.make_initial_func()
        else:
            data = self.sod.get_data_for_time(t)

            def f(x, el):
                return data(x)

            f.shape = self.sod.shape

            return f

    def make_initial_func(self):
        def e_from_p(p):
            # neglects a velocity term which is 0 for the Sod IC
            return p / (self.gamma - 1)

        s = self.sod

        def f(x, el):
            if x[0] <= (self.a+self.b)/2:
                return [s.rho_l, e_from_p(s.p_l)] + [0]*s.dim
            else:
                return [s.rho_r, e_from_p(s.p_r)] + [0]*s.dim

        f.shape = self.sod.shape

        return f

    @property
    def gamma(self):
        return self.sod.gamma

    @property
    def a(self):
        return self.sod.xl

    @property
    def b(self):
        return self.sod.xr




class LaxProblem(object):
    # http://www.crs4.it/HTML/int_book/NumericalMethods/subsection3_3_1.html
    a = 0
    b = 4
    is_periodic = False

    gamma = 1.4
    final_time = 0.17*b

    def __init__(self, dim=1):
        self.dim = dim

    def make_initial_func(self):
        def e_from_p(p, u, rho):
            return p / (self.gamma-1) + rho / 2 * u**2


        def f(x, el):
            if x[0] <= (self.a+self.b)/2:
                rho_l = .445
                p_l = 3.528
                u_l = .698

                return [rho_l, e_from_p(p_l, u_l, rho_l), rho_l*u_l] + [0]*(self.dim-1)
            else:
                rho_r = .5
                p_r = .571
                u_r = 0

                return [rho_r, e_from_p(p_r, u_r, rho_r), rho_r*u_r] + [0]*(self.dim-1)

        f.shape = (2+self.dim,)

        return f




class ShuOsherProblem(object):
    # http://www.astro.princeton.edu/~jstone/tests/shu-osher/Shu-Osher.html
    a = -5
    b = 5
    is_periodic = False

    gamma = 1.4
    final_time = 2

    def __init__(self, dim=1):
        self.dim = dim

    def make_initial_func(self):
        def e_from_p(p, u, rho):
            return p / (self.gamma-1) + rho / 2 * u**2

        from math import sin, pi

        def f(x, el):
            if x[0] <= -4:
                rho_l = 3.857143
                p_l = 10.33333
                u_l = 2.629369

                return [rho_l, e_from_p(p_l, u_l, rho_l), rho_l*u_l] + [0]*(self.dim-1)
            else:
                rho_r = 1 + 0.2*sin(5*x[0])
                p_r = 1
                u_r = 0

                return [rho_r, e_from_p(p_r, u_r, rho_r), rho_r*u_r] + [0]*(self.dim-1)

        f.shape = (2+self.dim,)

        return f




class SquareInChannelProblem(object):
    def __init__(self, el_volume=1):
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
        def round_trip_connect(seq):
            result = []
            for i in range(len(seq)):
                result.append((i, (i+1)%len(seq)))
            return result

        def needs_refinement(vertices, area):
            x =  sum(numpy.array(v) for v in vertices)/3

            max_area_volume = 0.7e-2 + 0.03*(0.05*x[1]**2 + 0.3*min(x[0]+1,0)**2)

            max_area_corners = 1e-3 + 0.001*max(
                    la.norm(x-corner)**4 for corner in obstacle_corners)

            return bool(area > 10*min(max_area_volume, max_area_corners))

        from meshpy.geometry import make_box
        points, facets, _ = make_box((-0.5,-0.5), (0.5,0.5))
        obstacle_corners = points[:]

        from meshpy.geometry import GeometryBuilder, Marker

        profile_marker = Marker.FIRST_USER_MARKER
        builder = GeometryBuilder()
        builder.add_geometry(points=points, facets=facets,
                facet_markers=profile_marker)

        points, facets, facet_markers = make_box((-16, -22), (25, 22))
        builder.add_geometry(points=points, facets=facets,
                facet_markers=facet_markers)

        from meshpy.triangle import MeshInfo, build
        mi = MeshInfo()
        builder.set(mi)
        mi.set_holes([(0,0)])

        mesh = build(mi, refinement_func=needs_refinement,
                allow_boundary_steiner=True,
                generate_faces=True)

        #from meshpy.triangle import write_gnuplot_mesh
        #write_gnuplot_mesh("mesh.dat", mesh)

        fvi2fm = mesh.face_vertex_indices_to_face_marker

        face_marker_to_tag = {
                profile_marker: "no_slip",
                Marker.MINUS_X: "inflow",
                Marker.PLUS_X: "outflow",
                Marker.MINUS_Y: "inflow",
                Marker.PLUS_Y: "inflow"
                }

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

    def get_operator(self, setup):
        from hedge.models.gas_dynamics import GasDynamicsOperator
        from hedge.second_order import IPDGSecondDerivative
        from hedge.mesh import TAG_ALL, TAG_NONE

        return GasDynamicsOperator(dimensions=2,
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




def make_stepper():
    #from hedge.timestep import RK4TimeStepper
    from hedge.timestep.dumka3 import Dumka3TimeStepper

    #return RK4TimeStepper()
    return Dumka3TimeStepper(3, rtol=1e-6)
    #return Dumka3TimeStepper(4)




def pre_smudge_ic(discr, op, bound_sensor, fields, adv_dt, visualize):
    stepper = make_stepper()

    bound_smudge_op = op.bind(discr, sensor=bound_sensor, viscosity_only=True)

    def smudge_rhs(t, q):
        ode_rhs, speed = bound_smudge_op(t, q)
        return ode_rhs

    next_dt = 0.005 * adv_dt

    from hedge.timestep import times_and_steps
    step_it = times_and_steps(
            final_time=300.5,
            max_dt_getter=lambda t: next_dt,
            taken_dt_getter=lambda: taken_dt)

    for step, t, dt in step_it:
        print "smudge", step, t
        if step % 1 == 0:
            visualize("pre-smudge-%04d" % step, t, fields)

        print [la.norm(x) for x in smudge_rhs(t, fields)]
        fields, t, taken_dt, next_dt = stepper(fields, t, next_dt, smudge_rhs)
    print "done smudging"

    return fields



def main(flux_type_arg="upwind"):
    from avcommon import make_ui, make_discr
    from euler_airplane import AirplaneProblem
    ui = make_ui(cases=[
        SodProblem,
        LaxProblem,
        ShuOsherProblem,
        AirplaneProblem,
        SquareInChannelProblem,
        ])
    setup = ui.gather()

    rcon, mesh_data, discr = make_discr(setup)

    if setup.vis_order is not None and setup.vis_order != setup.order:
        vis_discr = rcon.make_discretization(mesh_data, order=setup.vis_order)
    else:
        vis_discr = discr

    from hedge.discretization import Projector
    vis_proj = Projector(discr, vis_discr)

    from hedge.visualization import SiloVisualizer
    vis = SiloVisualizer(vis_discr, rcon)

    # initial condition -------------------------------------------------------
    if discr.dimensions > 1:
        def approximate_func(f):
            return make_obj_array(
                discr.interpolate_volume_function(f))
    else:
        def approximate_func(f):
            from hedge.discretization import adaptive_project_function_1d
            return make_obj_array(
                    adaptive_project_function_1d(discr, f))

    if hasattr(setup.case, "get_initial_data"):
        fields = setup.case.get_initial_data().volume_interpolant(0, discr)
    else:
        initial_func = setup.case.make_initial_func()
        fields = approximate_func(initial_func)

    # {{{ operator setup ------------------------------------------------------
    from hedge.data import \
            GivenFunction, \
            TimeConstantGivenFunction, \
            TimeDependentGivenFunction
    from hedge.second_order import (
            IPDGSecondDerivative, \
            LDGSecondDerivative, \
            CentralSecondDerivative)
    from hedge.models.gas_dynamics import GasDynamicsOperator
    from hedge.mesh import TAG_ALL, TAG_NONE

    if hasattr(setup.case, "get_operator"):
        op = setup.case.get_operator(setup)
    else:
        op = GasDynamicsOperator(discr.dimensions,
                gamma=setup.case.gamma,
                mu=0,

                bc_supersonic_inflow=TimeConstantGivenFunction(
                    GivenFunction(initial_func)),

                second_order_scheme=IPDGSecondDerivative(
                    stab_coefficient=setup.stab_coefficient),
                #second_order_scheme=CentralSecondDerivative(),

                supersonic_inflow_tag=TAG_ALL,
                supersonic_outflow_tag=TAG_NONE,
                inflow_tag=TAG_NONE,
                outflow_tag=TAG_NONE,
                noslip_tag=TAG_NONE)

    if rcon.is_head_rank:
        print "%d elements" % len(discr.mesh.elements)

    # }}}

    # {{{ diagnostics setup ---------------------------------------------------
    from pytools.log import (LogManager,
            add_general_quantities,
            add_simulation_quantities,
            add_run_info,
            MultiLogQuantity, EventCounter,
            DtConsumer, TimeTracker)

    log_file_name = "euler.dat"

    logmgr = LogManager(log_file_name, "w", rcon.communicator)
    add_run_info(logmgr)
    add_general_quantities(logmgr)
    add_simulation_quantities(logmgr)
    discr.add_instrumentation(logmgr)

    logmgr.set_constant("case_name", type(setup.case).__name__)
    logmgr.set_constant("sensor", setup.sensor)
    logmgr.set_constant("smoother", str(setup.smoother))
    logmgr.set_constant("viscosity_scale", setup.viscosity_scale)

    from hedge.log import LpNorm
    def rho_getter():
        return fields[0]
    rh_getter = lambda: fields[0]
    logmgr.add_quantity(LpNorm(rho_getter, discr, p=1, name="l1_u"))

    rhs_counter = EventCounter("rhs_evaluations")
    logmgr.add_quantity(rhs_counter)

    logmgr.add_watches(["step.max", "t_sim.max", "l1_u", "t_step.max"])

    # {{{ L2 error diagnostic

    class L2Error(TimeTracker, MultiLogQuantity):
        def __init__(self):
            MultiLogQuantity.__init__(self, 
                    names=["l2_err_rho", "l2_err_e", "l2_err_rho_u"])
            TimeTracker.__init__(self, None)

        def __call__(self):
            exact_func = setup.case.make_exact_func(self.t)
            exact_fields = approximate_func(exact_func)

            return [
                    discr.norm(op.rho(fields)-op.rho(exact_fields)),
                    discr.norm(op.e(fields)-op.e(exact_fields)),
                    discr.norm(op.rho_u(fields)-op.rho_u(exact_fields)),
                    ]

    if hasattr(setup.case, "make_exact_func"):
        error_quantity = L2Error()
        logmgr.add_quantity(error_quantity, interval=40)

    # }}}

    # }}}

    # {{{ rhs setup -----------------------------------------------------------
    from avcommon import sensor_from_string
    sensor, get_extra_vis_vectors = \
            sensor_from_string(setup.sensor, discr, setup, vis_proj)

    pre_bound_sensor = sensor.bind(discr)

    from hedge.bad_cell import make_h_over_n_vector
    h_over_n = make_h_over_n_vector(discr)

    bound_characteristic_velocity = op.bind_characteristic_velocity(discr)

    def bound_sensor(fields):

        char_vel = bound_characteristic_velocity(fields)

        rho = op.rho(fields)
        return pre_bound_sensor(rho, 
                viscosity_scaling=
                setup.viscosity_scale*h_over_n*char_vel)

    if setup.smoother is not None:
        bound_smoother = setup.smoother.bind(discr)
        pre_smoother_bound_sensor = bound_sensor

        def bound_sensor(fields):
            result = bound_smoother(pre_smoother_bound_sensor(fields))
            return result

    bound_op = op.bind(discr, sensor=bound_sensor,
            #sensor_mode="blended",
            #sensor_scaling=sensor.max_viscosity
            )

    max_eigval = [0]
    dbg_step = [0]
    def rhs(t, q):
        if False:
            visualize("debug-%04d" % dbg_step[0], t, q)
            dbg_step[0] += 1

        rhs_counter.add()

        ode_rhs, speed = bound_op(t, q)
        max_eigval[0] = speed
        return ode_rhs



    # }}}

    # {{{ vis subroutine ------------------------------------------------------
    vis_times = []
    vis_arrays = {}

    def visualize(name, t, fields):
        def to_vis(f):
            return vis_proj(discr.convert_volume(f, kind="numpy"))

        def vis_tuples(q, suffix=""):
            rho = op.rho(q)
            e = op.e(q)
            rho_u = op.rho_u(q)
            u = op.u(q)

            from hedge.tools import ptwise_dot
            p = (op.gamma-1)*(e-0.5*ptwise_dot(1, 1, rho_u, u))

            return [
                    ("rho"+suffix, to_vis(rho)),
                    ("e"+suffix, to_vis(e)),
                    ("rho_u"+suffix, to_vis(rho_u)),
                    ("u"+suffix, to_vis(u)),
                    ("p"+suffix, to_vis(p)) ]

        if hasattr(setup.case, "make_exact_func"):
            exact_func = setup.case.make_exact_func(t)
            exact_fields = make_obj_array(discr.interpolate_volume_function(
                        exact_func))

            extra_fields = (
                    vis_tuples(exact_fields, "_exact")
                    + vis_tuples(exact_fields - fields, "_error"))
        else:
            extra_fields = []

        if setup.extra_vis:
            extra_fields.extend(get_extra_vis_vectors(fields[0]))

        vis_fields = (
                vis_tuples(fields)
                + [ 
                    ("sensor", to_vis(bound_sensor(fields))),
                    ("char_vel", to_vis(bound_characteristic_velocity(
                        fields))),
                    ] 
                + extra_fields)

        visf = vis.make_file(name)
        vis.add_data(visf, vis_fields, time=t, step=step)
        visf.close()

        # {{{ save vis data for quad plot
        if discr.dimensions == 1:
            for name, data in vis_fields:
                if len(data.shape) > 1:
                    data = data[0]
                vis_arrays.setdefault(name, []).append(data)
            vis_times.append(t)
        # }}}

    # }}}

    # {{{ timestepping loop ---------------------------------------------------
    from hedge.timestep import RK4TimeStepper

    stepper = make_stepper()
    stepper.add_instrumentation(logmgr)

    step = 0
    rhs(0, fields)
    adv_dt = op.estimate_timestep(discr,
            stepper=RK4TimeStepper(), t=0, max_eigenvalue=max_eigval[0])
    logmgr.set_constant("adv_dt", adv_dt)

    #fields = pre_smudge_ic(discr, op, bound_sensor, fields, adv_dt, visualize)

    if setup.vis_interval is None:
        setup.vis_interval = min(1, setup.case.final_time / 20)

    next_vis_t = 0
    try:
        next_dt = 0.005 * adv_dt

        from hedge.timestep import times_and_steps
        step_it = times_and_steps(
                final_time=setup.case.final_time, logmgr=logmgr, 
                max_dt_getter=lambda t: next_dt,
                taken_dt_getter=lambda: taken_dt)

        for step, t, dt in step_it:
            do_vis = False
            if setup.vis_interval:
                do_vis = do_vis or t >= next_vis_t
                if do_vis:
                    next_vis_t += setup.vis_interval

            if setup.vis_interval_steps:
                do_vis = do_vis or (step % setup.vis_interval_steps == 0)

            if do_vis:
                visualize("euler-%06d" % step, t, fields)

            fields, t, taken_dt, next_dt = stepper(fields, t, next_dt, rhs)

    finally:
        discr.close()
        vis.close()
        logmgr.close()

        # {{{ write out quad plot
        if vis_discr.dimensions == 1 and vis_times:
            import pylo
            with pylo.SiloFile("xtmesh.silo", mode=pylo.DB_CLOBBER) as f:
                f.put_quadmesh("xtmesh", [
                    vis_discr.nodes[:,0].copy(),
                    numpy.array(vis_times, dtype=numpy.float64)*10
                    ])

                for name, data in vis_arrays.iteritems():
                    ary_data = numpy.asarray(data)
                    f.put_quadvar1(name, "xtmesh",
                            ary_data, ary_data.shape,
                            centering=pylo.DB_NODECENT)

        # }}}
    # }}}





if __name__ == "__main__":
    main()




# vim: foldmethod=marker
