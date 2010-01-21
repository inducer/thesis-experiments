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
from math import sin, pi, sqrt




class ExactTestCase:
    a = 0
    b = 150
    final_time = 5000

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

        from math import sqrt

        shock_loc = 30*sqrt(2*t+40)/sqrt(120) + t/4 - 10
        shock_win = (shock_loc + 20) // self.b
        x += shock_win * 150 

        x -= 20

        return max(f(x, shock_loc), f(x-self.b, shock_loc-self.b))

class OffCenterMigratingTestCase:
    a = -pi
    b = pi
    final_time = 10

    def u0(self, x):
        return -0.4+sin(x+0.1)


class CenteredStationaryTestCase:
    # does funny things to P-P
    a = -pi
    b = pi
    final_time = 10

    def u0(self, x):
        return -sin(x)

class OffCenterStationaryTestCase:
    # does funny things to P-P
    a = -pi
    b = pi
    final_time = 10

    def u0(self, x):
        return -sin(x+0.3)



def main(write_output=True, flux_type_arg="upwind"):
    #case = CenteredStationaryTestCase()
    #case = OffCenterStationaryTestCase()
    case = OffCenterMigratingTestCase()
    #case = ExactTestCase()

    from hedge.backends import guess_run_context
    rcon = guess_run_context()

    order = 4
    n_elements = 20

    if rcon.is_head_rank:
        if True:
            from hedge.mesh.generator import make_uniform_1d_mesh
            mesh = make_uniform_1d_mesh(case.a, case.b, 20, periodic=True)
        else:
            extent_y = 4
            dx = (case.b-case.a)/n_elements
            subdiv = (n_elements, int(1+extent_y//dx))
            from pytools import product

            from hedge.mesh.generator import make_rect_mesh
            mesh = make_rect_mesh((case.a, 0), (case.b, extent_y), 
                    periodicity=(True, True), 
                    subdivisions=subdiv,
                    max_area=(case.b-case.a)*extent_y/(2*product(subdiv))
                    )

    if rcon.is_head_rank:
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    discr = rcon.make_discretization(mesh_data, order=order)
    vis_discr = rcon.make_discretization(mesh_data, order=10)
    #vis_discr = discr

    from hedge.discretization import Projector
    vis_proj = Projector(discr, vis_discr)

    from hedge.visualization import VtkVisualizer, SiloVisualizer
    if write_output:
        vis = SiloVisualizer(vis_discr, rcon)
        #vis = VtkVisualizer(vis_discr, rcon, "fld")

    # operator setup ----------------------------------------------------------
    from hedge.data import \
            ConstantGivenFunction, \
            TimeConstantGivenFunction, \
            TimeDependentGivenFunction
    from hedge.tools.second_order import (
            IPDGSecondDerivative, \
            LDGSecondDerivative, \
            CentralSecondDerivative)
    from hedge.models.burgers import BurgersOperator
    op = BurgersOperator(mesh.dimensions,
            viscosity_scheme=IPDGSecondDerivative())

    if rcon.is_head_rank:
        print "%d elements" % len(discr.mesh.elements)

    # exact solution ----------------------------------------------------------
    import pymbolic
    var = pymbolic.var

    u = discr.interpolate_volume_function(lambda x, el: case.u0(x[0]))

    # diagnostics setup -------------------------------------------------------
    from pytools.log import LogManager, \
            add_general_quantities, \
            add_simulation_quantities, \
            add_run_info

    if write_output:
        log_file_name = "burgers.dat"
    else:
        log_file_name = None

    logmgr = LogManager(log_file_name, "w", rcon.communicator)
    add_run_info(logmgr)
    add_general_quantities(logmgr)
    add_simulation_quantities(logmgr)
    discr.add_instrumentation(logmgr)

    from hedge.log import LpNorm
    u_getter = lambda: u
    logmgr.add_quantity(LpNorm(u_getter, discr, p=1, name="l1_u"))

    logmgr.add_watches(["step.max", "t_sim.max", "l1_u", "t_step.max"])

    # timestep loop -----------------------------------------------------------
    mesh_a, mesh_b = mesh.bounding_box()
    from pytools import product
    area = product(mesh_b[i] - mesh_a[i] for i in range(mesh.dimensions))
    h = sqrt(area/len(mesh.elements))
    from hedge.tools.bad_cell import (
            PerssonPeraireDiscontinuitySensor,
            DecayGatingDiscontinuitySensorBase)
    sub_sensor = DecayGatingDiscontinuitySensorBase(5*h/(order))
    bound_sub_sensor = sub_sensor.bind(discr)
    #sensor2 = PerssonPeraireDiscontinuitySensor(kappa=2,
            #eps0=h/order, s_0=numpy.log10(1/order**4)).bind(discr)
    decay_expt = sub_sensor.bind_quantity(discr, "decay_expt")
    decay_lmc = sub_sensor.bind_quantity(discr, "log_modal_coeffs")
    decay_estimated_lmc = sub_sensor.bind_quantity(discr, "estimated_log_modal_coeffs")
    jump_part = sub_sensor.bind_quantity(discr, "jump_part")
    jump_modes = sub_sensor.bind_quantity(discr, "modal_coeffs_jump")
    jump_lmc = sub_sensor.bind_quantity(discr, "log_modal_coeffs_jump")

    from smoother import TriBlobSmoother
    smoother = TriBlobSmoother(discr)

    sensor = lambda u: smoother(bound_sub_sensor(u))

    rhs = op.bind(discr, sensor=sensor)
    #rhs = op.bind(discr, sensor=sensor)
    #rhs2 = op.bind(discr, sensor=lambda u: smoother(sensor2(u)))

    from hedge.timestep import RK4TimeStepper
    from hedge.timestep.dumka3 import Dumka3TimeStepper
    #stepper = RK4TimeStepper()
    stepper = Dumka3TimeStepper(3, rtol=1e-6)
    #stepper = Dumka3TimeStepper(4)

    stepper.add_instrumentation(logmgr)

    u2 = u.copy()

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
                final_time=case.final_time, logmgr=logmgr, 
                max_dt_getter=lambda t: next_dt,
                taken_dt_getter=lambda: taken_dt)
        from hedge.optemplate import  InverseVandermondeOperator
        inv_vdm = InverseVandermondeOperator().bind(discr)

        for step, t, dt in step_it:
            if step % 10 == 0 and write_output:
                if hasattr(case, "u_exact"):
                    extra_fields = [
                            ("u_exact", 
                                vis_discr.interpolate_volume_function(
                                    lambda x, el: case.u_exact(x[0], t)))]
                else:
                    extra_fields = []

                visf = vis.make_file("fld-%04d" % step)
                vis.add_data(visf, [ 
                    ("u_dg", vis_proj(u)), 
                    #("u_pp", vis_proj(u2)), 
                    ("sensor_dg", vis_proj(sensor(u))), 
                    #("sensor_pp", vis_proj(sensor2(u2))), 
                    ("expt_u_dg", vis_proj(decay_expt(u))), 
                    ("jump_part", vis_proj(jump_part(u))), 
                    ("jump_modes", vis_proj(jump_modes(u))), 
                    ("lmc_u_dg", vis_proj(decay_lmc(u))), 
                    ("lmc_u_dg_jump", vis_proj(jump_lmc(u))), 
                    ("est_lmc_u_dg", vis_proj(decay_estimated_lmc(u))), 
                    ] + extra_fields,
                    time=t,
                    step=step)
                visf.close()

            u, t, taken_dt, next_dt = stepper(u, t, next_dt, rhs)

    finally:
        if write_output:
            vis.close()

        logmgr.save()




if __name__ == "__main__":
    main()
