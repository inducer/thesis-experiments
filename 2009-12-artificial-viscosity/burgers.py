from __future__ import division, with_statement
import numpy
import numpy.linalg as la
from math import sin, pi, sqrt




class LeaningTriangleTestCase(object):
    a = 0
    b = 150
    is_periodic = True
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

class MultipliedLeaningTriangleTestCase(object):
    a = 0
    b = 150
    is_periodic = True
    factor = 2
    final_time = 240/factor # that's how long the solution is exact, roughly


    def u0(self, x):
        t = 0
        def f(x, shock_loc):
            if x < (t-40)/4:
                return 1/4
            elif x < (3*t)/4:
                return (x+15)/(t+20)
            elif x < (t+80)/4:
                return (x-30)/(t-40)
            else:
                return 1/4

        shock_loc = 30*sqrt(2*t+40)/sqrt(120) + t/4 - 10
        shock_win = (shock_loc + 20) // self.b
        x += shock_win * 150 

        x -= 20

        return self.factor*max(f(x, shock_loc), f(x-self.b, shock_loc-self.b))


class TimBump(object):
    a = -1
    b = 1
    is_periodic = True
    final_time = 0.5

    def u0(self, x):
        xl = -.12
        xc = -0.02
        xr = .1
        base = 0.2

        if xl < x <=xc:
            return base + (x-xl)/(xc-xl)
        elif xc < x <= xr:
            return base + (xc-xl)*(xr-x)/((xc-xl)*(xr-xc))
        else:
            return base



class TimSine(object):
    a = -1
    b = 1
    is_periodic = True
    final_time = 0.5

    def u0(self, x):
        return sin(10*pi*x)



class OffCenterMigratingTestCase(object):
    a = -pi
    b = pi
    is_periodic = True
    final_time = 10

    def u0(self, x):
        return -0.4+sin(x+0.1)


class CenteredStationaryTestCase(object):
    # does funny things to P-P
    a = -pi
    b = pi
    is_periodic = True
    final_time = 10

    def u0(self, x):
        return -sin(x)

class OffCenterStationaryTestCase(object):
    # does funny things to P-P
    a = -pi
    b = pi
    is_periodic = True
    final_time = 10

    def u0(self, x):
        return -sin(x+0.3)



def main(flux_type_arg="upwind"):
    from avcommon import make_ui, make_discr
    ui = make_ui(cases=[
            LeaningTriangleTestCase,
            MultipliedLeaningTriangleTestCase,
            CenteredStationaryTestCase,
            OffCenterStationaryTestCase,
            OffCenterMigratingTestCase,
            TimBump,
            TimSine,
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
    #vis = VtkVisualizer(vis_discr, rcon)

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
    op = BurgersOperator(discr.dimensions,
            viscosity_scheme=IPDGSecondDerivative(stab_coefficient=setup.stab_coefficient))

    if rcon.is_head_rank:
        print "%d elements" % len(discr.mesh.elements)

    # exact solution ----------------------------------------------------------
    import pymbolic
    var = pymbolic.var

    if discr.dimensions > 1:
        def approximate_func(discr, f):
            return discr.interpolate_volume_function(f)
    else:
        def approximate_func(discr, f):
            from hedge.discretization import adaptive_project_function_1d
            return adaptive_project_function_1d(discr, f)

    def initial_func(x, el): 
        return setup.case.u0(x[0])

    u = approximate_func(discr, initial_func)

    # {{{ diagnostics setup ---------------------------------------------------
    from pytools.log import (LogManager,
            add_general_quantities,
            add_simulation_quantities,
            add_run_info,
            LogQuantity,
            TimeTracker,
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

    class MaxSensor(LogQuantity):
        def __init__(self):
            LogQuantity.__init__(self, "max_sensor")

        def __call__(self):
            return numpy.max(bound_sensor(u))

    logmgr.add_quantity(MaxSensor())

    class L1Error(TimeTracker, LogQuantity):
        def __init__(self):
            LogQuantity.__init__(self, "l1_error")
            TimeTracker.__init__(self, None)

        def __call__(self):
            def to_vis(f):
                return vis_proj(discr.convert_volume(f, kind="numpy"))

            if vis_discr is discr:
                from warnings import warn
                warn("L1 norm might be inaccurate")

            u_exact = approximate_func(vis_discr,
                    lambda x, el: setup.case.u_exact(x[0], self.t))

            from avcommon import l1_norm
            return l1_norm(vis_discr, to_vis(u)-u_exact)

    if hasattr(setup.case, "u_exact"):
        error_quantity = L1Error()
        logmgr.add_quantity(error_quantity, interval=50)

    logmgr.add_watches(["step.max", "t_sim.max", "l1_u", "t_step.max"])
    # }}}

    # {{{ rhs setup -----------------------------------------------------------
    from avcommon import sensor_from_string
    sensor, get_extra_vis_vectors = sensor_from_string(setup.sensor, discr, setup, vis_proj)

    pre_bound_sensor = sensor.bind(discr)
    bound_characteristic_velocity = op.bind_characteristic_velocity(discr)

    from hedge.bad_cell import make_h_over_n_vector
    h_over_n = make_h_over_n_vector(discr)

    def bound_sensor(u):
        char_vel = bound_characteristic_velocity(u)

        return pre_bound_sensor(u, 
                viscosity_scaling=
                setup.viscosity_scale*h_over_n*char_vel)

    if setup.smoother is not None:
        bound_smoother = setup.smoother.bind(discr)
        pre_smoother_bound_sensor = bound_sensor

        def bound_sensor(u):
            result = bound_smoother(
                    pre_smoother_bound_sensor(u))
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

    # }}}

    # {{{ vis subroutine ------------------------------------------------------
    vis_times = []
    vis_arrays = {}

    def visualize(name, t, u):
        def to_vis(f):
            return vis_proj(discr.convert_volume(f, kind="numpy"))
        vis_u = vis_proj(u)

        if hasattr(setup.case, "u_exact"):
            u_exact = vis_discr.interpolate_volume_function(
                            lambda x, el: setup.case.u_exact(x[0], t))
            extra_fields = [
                    ("u_exact", u_exact),
                    ("u_err", u_exact-vis_u)
                    ]
        else:
            extra_fields = []

        if setup.extra_vis:
            extra_fields.extend(get_extra_vis_vectors(u))

        vis_tuples = [ 
            ("u_dg", vis_u), 
            ("sensor", to_vis(bound_sensor(u))), 
            ("char_vel", to_vis(bound_characteristic_velocity(
                u))),
            ] + extra_fields

        visf = vis.make_file(name)
        vis.add_data(visf, vis_tuples, time=t, step=step)
        visf.close()

        # {{{ save vis data for quad plot
        if discr.dimensions == 1:
            for name, data in vis_tuples:
                if len(data.shape) > 1:
                    data = data[0]
                vis_arrays.setdefault(name, []).append(data)
            vis_times.append(t)
        # }}}
    # }}}

    # {{{ timestep loop -------------------------------------------------------
    from hedge.timestep.runge_kutta import (
            LSRK4TimeStepper,
            ODE45TimeStepper,
            ODE23TimeStepper)
    from hedge.timestep.dumka3 import Dumka3TimeStepper
    #stepper = RK4TimeStepper()
    stepper = Dumka3TimeStepper(0, rtol=1e-6)
    #stepper = Dumka3TimeStepper(4)
    #stepper = ODE23TimeStepper(rtol=1e-6)
    #stepper = ODE45TimeStepper(rtol=1e-6)

    stepper.add_instrumentation(logmgr)

    if setup.vis_interval is None:
        setup.vis_interval = setup.case.final_time / 30

    next_vis_t = 0
    try:
        from hedge.timestep import times_and_steps
        # for visc=0.01
        #stab_fac = 0.1 # RK4
        #stab_fac = 1.6 # dumka3(3), central
        #stab_fac = 3 # dumka3(4), central

        adv_dt = op.estimate_timestep(discr,
                stepper=LSRK4TimeStepper(), t=0, fields=u)
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

        visualize("fld-%04d" % (step+1), setup.case.final_time, u)
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
