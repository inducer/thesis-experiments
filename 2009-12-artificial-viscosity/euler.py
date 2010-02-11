from __future__ import division
import numpy
import numpy.linalg as la
from math import sin, pi, sqrt
from pytools.obj_array import make_obj_array




class SodTestCase(object):
    a = 0
    b = 1
    final_time = 0.25
    gamma = 1.4

    def make_initial_func(self, dim):
        def e_from_p(p):
            # neglects a velocity term which is 0 for the Sod IC
            return p / (self.gamma - 1)

        class SodFunction:
            shape = (2+dim,)

            def __call__(subself, x, el):
                if x[0] <= (self.a+self.b)/2:
                    return [1, e_from_p(1)] + [0]*dim
                else:
                    return [0.125, e_from_p(0.1)] + [0]*dim

        return SodFunction()

    def make_exact_func(self, dim):
        pass



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
        SodTestCase
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
            "dump_optemplate_stages",
            "dump_op_code"
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

    initial_func = setup.case.make_initial_func(discr.dimensions)
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

    bound_op = op.bind(discr, sensor=bound_sensor)

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

    # {{{ pre-smudge loop -----------------------------------------------------

    # }}}

    # {{{ vis subroutine ------------------------------------------------------
    def visualize(name, t, fields):
        if hasattr(setup.case, "u_exact"):
            extra_fields = [
                    ("u_exact", 
                        vis_discr.interpolate_volume_function(
                            lambda x, el: setup.case.u_exact(x[0], t)))]
        else:
            extra_fields = []

        if setup.extra_vis:
            extra_fields.extend(get_extra_vis_vectors(fields[0]))

        def to_vis(f):
            return vis_proj(discr.convert_volume(f, kind="numpy"))

        visf = vis.make_file(name)

        rho = op.rho(fields)
        e = op.e(fields)
        rho_u = op.rho_u(fields)
        u = op.u(fields)

        from hedge.tools import ptwise_dot
        p = (op.gamma-1)*(e-0.5*ptwise_dot(1, 1, rho_u, u))

        vis.add_data(visf,
                [
                    ("rho", to_vis(rho)),
                    ("e", to_vis(e)),
                    ("rho_u", to_vis(rho_u)),
                    ("u", to_vis(u)),
                    ("p", to_vis(p)),
                    ("sensor", to_vis(bound_sensor(fields))), 
                ] + extra_fields,
                expressions=[
                    ],
                time=t, step=step)

        visf.close()

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

    # }}}




if __name__ == "__main__":
    main()




# vim: foldmethod=marker
