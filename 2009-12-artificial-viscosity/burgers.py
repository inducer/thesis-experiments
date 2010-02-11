from __future__ import division
import numpy
import numpy.linalg as la
from math import sin, pi, sqrt




class LeaningTriangleTestCase(object):
    a = 0
    b = 150
    final_time = 280 # that's how long the solution is exact, roughly

    def u0(self, x):
        return self.u_exact(x, 0)

    def u_exact(self, x, t):
        # CAUTION: This gets the shock speed wrong as soon as the pulse
        # starts interacting with itself.

        def f(x, shock_loc):
            if x < (t-40)/4:
                return 1/4
            else:
                if t < 40:
                    if x < (3*t)/4:
                        return (x+15)/(t+20)
                    elif x < (t+80)/4:
                        return (x-30)/(t-40)
                    else:
                        return 1/4
                else:
                    if x < shock_loc:
                        return (x+15)/(t+20)
                    else:
                        return 1/4

        shock_loc = 30*sqrt(2*t+40)/sqrt(120) + t/4 - 10
        shock_win = (shock_loc + 20) // self.b
        x += shock_win * 150 

        x -= 20

        return max(f(x, shock_loc), f(x-self.b, shock_loc-self.b))

class OffCenterMigratingTestCase(object):
    a = -pi
    b = pi
    final_time = 10

    def u0(self, x):
        return -0.4+sin(x+0.1)


class CenteredStationaryTestCase(object):
    # does funny things to P-P
    a = -pi
    b = pi
    final_time = 10

    def u0(self, x):
        return -sin(x)

class OffCenterStationaryTestCase(object):
    # does funny things to P-P
    a = -pi
    b = pi
    final_time = 10

    def u0(self, x):
        return -sin(x+0.3)



def main(flux_type_arg="upwind"):
    from hedge.backends import guess_run_context
    rcon = guess_run_context()

    from avcommon import make_ui
    ui = make_ui(cases=[
            LeaningTriangleTestCase,
            CenteredStationaryTestCase,
            OffCenterStationaryTestCase,
            OffCenterMigratingTestCase,
            ])
    setup = ui.gather()

    if rcon.is_head_rank:
        if True:
            from hedge.mesh.generator import make_uniform_1d_mesh
            mesh = make_uniform_1d_mesh(setup.case.a, setup.case.b, setup.n_elements, periodic=True)
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
        quad_min_degrees = {"quad":3*setup.order}
    elif setup.quad_min_degree == 0:
        quad_min_degrees = {}
    else:
        quad_min_degrees = {"quad": setup.quad_min_degree}

    discr = rcon.make_discretization(mesh_data, order=setup.order,
            quad_min_degrees=quad_min_degrees,
            #debug=["dump_optemplate_stages"]
            )
    if setup.vis_order is not None and setup.vis_order != setup.order:
        vis_discr = rcon.make_discretization(mesh_data, order=setup.vis_order)
    else:
        vis_discr = discr

    from hedge.discretization import Projector
    vis_proj = Projector(discr, vis_discr)

    from hedge.visualization import SiloVisualizer
    vis = SiloVisualizer(vis_discr, rcon)

    # operator setup ----------------------------------------------------------
    from hedge.data import \
            ConstantGivenFunction, \
            TimeConstantGivenFunction, \
            TimeDependentGivenFunction
    from hedge.second_order import (
            IPDGSecondDerivative, \
            LDGSecondDerivative, \
            CentralSecondDerivative)
    from hedge.models.burgers import BurgersOperator
    op = BurgersOperator(mesh.dimensions,
            viscosity_scheme=IPDGSecondDerivative(stab_coefficient=setup.stab_coefficient))

    if rcon.is_head_rank:
        print "%d elements" % len(discr.mesh.elements)

    # exact solution ----------------------------------------------------------
    import pymbolic
    var = pymbolic.var

    u = discr.interpolate_volume_function(lambda x, el: setup.case.u0(x[0]))

    # diagnostics setup -------------------------------------------------------
    from pytools.log import (LogManager,
            add_general_quantities,
            add_simulation_quantities,
            add_run_info,
            SimulationLogQuantity,
            MultiLogQuantity, EventCounter)

    log_file_name = "burgers.dat"

    logmgr = LogManager(log_file_name, "w", rcon.communicator)
    add_run_info(logmgr)
    add_general_quantities(logmgr)
    add_simulation_quantities(logmgr)
    discr.add_instrumentation(logmgr)

    logmgr.set_constant("case_name", type(setup.case).__name__)
    logmgr.set_constant("sensor", setup.sensor)
    logmgr.set_constant("smoother", str(setup.smoother))

    from hedge.log import LpNorm
    u_getter = lambda: u
    logmgr.add_quantity(LpNorm(u_getter, discr, p=1, name="l1_u"))

    class TotalVariation(MultiLogQuantity):
        def __init__(self):
            MultiLogQuantity.__init__(self,
                    names=["total_variation", "tv_change", "tv_vs_min"])
            self.last_tv = None
            self.min_tv = None

            if discr is vis_discr:
                from warnings import warn
                warn("Total variation results are likely inaccurate.")

        def __call__(self):
            hires_u = vis_proj(u)
            tv = numpy.sum(numpy.abs(numpy.diff(hires_u)))

            if self.min_tv is None:
                self.min_tv = tv
            else:
                self.min_tv = min(tv, self.min_tv)

            if self.last_tv is None:
                result = [tv, None]
            else:
                result = [tv, tv-self.last_tv, tv-self.min_tv]

            self.last_tv = tv

            return result

    logmgr.add_quantity(TotalVariation())

    rhs_counter = EventCounter("rhs_evaluations")
    logmgr.add_quantity(rhs_counter)

    class L2Error(SimulationLogQuantity):
        def __init__(self):
            SimulationLogQuantity.__init__(self, 0, "l2_error")
            self.t = 0

        def __call__(self):
            u_exact = discr.interpolate_volume_function(
                    lambda x, el: setup.case.u_exact(x[0], t))
            self.t += self.dt
            return discr.norm(u-u_exact)

    if hasattr(setup.case, "u_exact"):
        error_quantity = L2Error()
        logmgr.add_quantity(error_quantity, interval=10)

    logmgr.add_watches(["step.max", "t_sim.max", "l1_u", "t_step.max"])

    # timestep loop -----------------------------------------------------------
    from avcommon import sensor_from_string
    sensor, extra_vis_vector = sensor_from_string(setup.sensor, discr, setup, vis_proj)

    bound_sensor = sensor.bind(discr)

    if setup.smoother is not None:
        bound_smoother = setup.smoother.bind(discr)
        pre_smoother_bound_sensor = bound_sensor

        def bound_sensor(u):
            result = bound_smoother(pre_smoother_bound_sensor(u))
            return result

    bound_op = op.bind(discr, sensor=bound_sensor)

    dbg_step = [0]
    def rhs(t, u):
        rhs_counter.add()

        if False:
            sensor_val = bound_sensor(u)
            if step>420 or dbg_step[0]:
                print step, dbg_step[0], la.norm(u), la.norm(sensor_val)

                visualize("debug-%04d" % dbg_step[0], t, u)
                dbg_step[0] += 1

                if False:
                    print "u"
                    print u
                    print "sensor"
                    print sensor_val
                    print "lmc"
                    print decay_lmc(u)
                    print "expt"
                    print decay_expt(u)

        return bound_op(t, u)

    from hedge.timestep import RK4TimeStepper
    from hedge.timestep.dumka3 import Dumka3TimeStepper
    #stepper = RK4TimeStepper()
    stepper = Dumka3TimeStepper(3, rtol=1e-6)
    #stepper = Dumka3TimeStepper(4)

    stepper.add_instrumentation(logmgr)

    def visualize(name, t, u):
        if hasattr(setup.case, "u_exact"):
            extra_fields = [
                    ("u_exact", 
                        vis_discr.interpolate_volume_function(
                            lambda x, el: setup.case.u_exact(x[0], t)))]
        else:
            extra_fields = []

        if setup.extra_vis:
            extra_fields.extend(get_extra_vis_vectors(u))

        visf = vis.make_file(name)
        vis.add_data(visf, [ 
            ("u_dg", vis_proj(u)), 
            ("sensor", vis_proj(bound_sensor(u))), 
            #("u_pp", vis_proj(u2)), 
            ] + extra_fields,
            time=t,
            step=step)
        visf.close()

    next_vis_t = 0
    try:
        from hedge.timestep import times_and_steps
        # for visc=0.01
        #stab_fac = 0.1 # RK4
        #stab_fac = 1.6 # dumka3(3), central
        #stab_fac = 3 # dumka3(4), central

        adv_dt = op.estimate_timestep(discr,
                stepper=RK4TimeStepper(), t=0, fields=u)
        next_dt = 0.05 * adv_dt

        logmgr.set_constant("adv_dt", adv_dt)

        step_it = times_and_steps(
                final_time=setup.case.final_time, logmgr=logmgr, 
                max_dt_getter=lambda t: next_dt,
                taken_dt_getter=lambda: taken_dt)
        from hedge.optemplate import  InverseVandermondeOperator
        inv_vdm = InverseVandermondeOperator().bind(discr)

        for step, t, dt in step_it:
            do_vis = False
            if setup.vis_interval:
                do_vis = do_vis or t >= next_vis_t
                if do_vis:
                    next_vis_t += setup.vis_interval

            if setup.vis_interval_steps:
                do_vis = do_vis or (step % setup.vis_interval_steps == 0)

            if do_vis:
                visualize("fld-%04d" % step, t, u)

            u, t, taken_dt, next_dt = stepper(u, t, next_dt, rhs)

    finally:
        vis.close()
        logmgr.close()




if __name__ == "__main__":
    main()
