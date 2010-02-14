from __future__ import division, with_statement
import numpy
import numpy.linalg as la
from pytools.obj_array import make_obj_array




class SodProblem(object):
    final_time = 0.25

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
    a = -2
    b = 2
    gamma = 5/3
    final_time = 5

    def __init__(self, dim=1):
        self.dim = dim

    def make_initial_func(self):
        def e_from_p(p, u, rho):
            return p / (self.gamma-1) + rho / 2 * u**2

        from math import sin, pi

        def f(x, el):
            if x[0] <= self.a+(self.b-self.a)*0.125:
                rho_l = 3.857143
                p_l = 10.33333
                u_l = 2.629369

                return [rho_l, e_from_p(p_l, u_l, rho_l), rho_l*u_l] + [0]*(self.dim-1)
            else:
                rho_r = 1 + 0.2*sin(5*pi*x[0])
                p_r = 1
                u_r = 0

                return [rho_r, e_from_p(p_r, u_r, rho_r), rho_r*u_r] + [0]*(self.dim-1)

        f.shape = (2+self.dim,)

        return f




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
    from hedge.backends import guess_run_context
    rcon = guess_run_context()

    from avcommon import make_ui
    ui = make_ui(cases=[
        SodProblem,
        LaxProblem,
        ShuOsherProblem,
        ])
    setup = ui.gather()

    if rcon.is_head_rank:
        if True:
            from hedge.mesh.generator import make_uniform_1d_mesh
            mesh = make_uniform_1d_mesh(
                    setup.case.a, setup.case.b, 
                    setup.n_elements, 
                    periodic=False)
        else:
            extent_y = 4
            dx = (setup.case.b-setup.case.a)/n_elements
            subdiv = (n_elements, int(1+extent_y//dx))
            from pytools import product

            from hedge.mesh.generator import make_rect_mesh
            mesh = make_rect_mesh((setup.case.a, 0), (setup.case.b, extent_y), 
                    periodicity=(True, True), 
                    subdivisions=subdiv,
                    max_area=(setup.case.b-setup.case.a)*extent_y/(2*product(subdiv))
                    )

    if rcon.is_head_rank:
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    if setup.quad_min_degree is None:
        quad_min_degrees = {
                "gasdyn_vol": 3*setup.order,
                "gasdyn_face": 3*setup.order,
                }
    elif setup.quad_min_degree == 0:
        quad_min_degrees = {}
    else:
        quad_min_degrees = {
                "gasdyn_vol": setup.quad_min_degree,
                "gasdyn_face": setup.quad_min_degree,
                }

    discr = rcon.make_discretization(mesh_data, order=setup.order,
            quad_min_degrees=quad_min_degrees,
            debug=[
            #"dump_optemplate_stages",
            #"dump_op_code"
            ]
            )
    if setup.vis_order is not None and setup.vis_order != setup.order:
        vis_discr = rcon.make_discretization(mesh_data, order=setup.vis_order)
    else:
        vis_discr = discr

    from hedge.discretization import Projector
    vis_proj = Projector(discr, vis_discr)

    from hedge.visualization import SiloVisualizer
    vis = SiloVisualizer(vis_discr, rcon)

    # initial condition -------------------------------------------------------
    import pymbolic
    var = pymbolic.var

    initial_func = setup.case.make_initial_func()
    fields = make_obj_array(
            discr.interpolate_volume_function(initial_func))

    # operator setup ----------------------------------------------------------
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

    op = GasDynamicsOperator(mesh.dimensions,
            gamma=setup.case.gamma,
            mu=0,

            bc_inflow=TimeConstantGivenFunction(
                GivenFunction(initial_func)),
            bc_outflow=TimeConstantGivenFunction(
                GivenFunction(initial_func)),
            bc_noslip=TimeConstantGivenFunction(
                GivenFunction(initial_func)),

            second_order_scheme=IPDGSecondDerivative(
                stab_coefficient=setup.stab_coefficient),
            #second_order_scheme=CentralSecondDerivative(),

            inflow_tag=TAG_ALL,
            outflow_tag=TAG_NONE,
            noslip_tag=TAG_NONE)

    if rcon.is_head_rank:
        print "%d elements" % len(discr.mesh.elements)

    # diagnostics setup -------------------------------------------------------
    from pytools.log import (LogManager,
            add_general_quantities,
            add_simulation_quantities,
            add_run_info,
            EventCounter)

    log_file_name = "euler.dat"

    logmgr = LogManager(log_file_name, "w", rcon.communicator)
    add_run_info(logmgr)
    add_general_quantities(logmgr)
    add_simulation_quantities(logmgr)
    discr.add_instrumentation(logmgr)

    logmgr.set_constant("case_name", type(setup.case).__name__)
    logmgr.set_constant("sensor", setup.sensor)
    logmgr.set_constant("smoother", str(setup.smoother))

    from hedge.log import LpNorm
    def rho_getter():
        return fields[0]
    rh_getter = lambda: fields[0]
    logmgr.add_quantity(LpNorm(rho_getter, discr, p=1, name="l1_u"))

    rhs_counter = EventCounter("rhs_evaluations")
    logmgr.add_quantity(rhs_counter)

    logmgr.add_watches(["step.max", "t_sim.max", "l1_u", "t_step.max"])

    # {{{ timestep loop -----------------------------------------------------------
    from avcommon import sensor_from_string
    sensor, get_extra_vis_vectors = \
            sensor_from_string(setup.sensor, discr, setup, vis_proj)

    bound_sensor = sensor.bind(discr)

    if setup.smoother is not None:
        bound_smoother = setup.smoother.bind(discr)
        pre_smoother_bound_sensor = bound_sensor

        def bound_sensor(fields):
            rho = op.rho(fields)
            result = bound_smoother(pre_smoother_bound_sensor(rho))
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
            exact_fields = discr.interpolate_volume_function(
                        exact_func)

            extra_fields = vis_tuples(
                    make_obj_array(exact_fields), "_exact")
        else:
            extra_fields = []

        if setup.extra_vis:
            extra_fields.extend(get_extra_vis_vectors(fields[0]))

        vis_fields = (
                vis_tuples(fields)
                + [ ("sensor", to_vis(bound_sensor(fields))) ] 
                + extra_fields)

        visf = vis.make_file(name)
        vis.add_data(visf, vis_fields, time=t, step=step)
        visf.close()

        # {{{ save vis data for quad plot
        from pytools.obj_array import is_obj_array

        if discr.dimensions == 1:
            vis_times.append(t)
            for name, data in vis_fields:
                if len(data.shape) > 1:
                    data = data[0]
                vis_arrays.setdefault(name, []).append(data)
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
                visualize("euler-%04d" % step, t, fields)

            fields, t, taken_dt, next_dt = stepper(fields, t, next_dt, rhs)

    finally:
        discr.close()
        vis.close()
        logmgr.close()

        # {{{ write out quad plot
        if vis_discr.dimensions == 1 and vis_times:
            with pylo.SiloFile("xtmesh.silo", mode=pylo.DB_CLOBBER) as f:
                f.put_quadmesh("xtmesh", [
                    vis_discr.nodes[:,0].copy(),
                    numpy.array(vis_times)*10
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
