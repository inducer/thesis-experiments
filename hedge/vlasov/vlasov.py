
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




class VlasovOperator:
    def __init__(self, v_grid_points=50):
        assert v_grid_points % 2 == 0
        # otherwise differentiation becomes non-invertible

        import spyctral.wienerfun as wienerfun
        self.v_quad_points_1d, self.v_quad_weights_1d = \
                wienerfun.quad.genwienerw_pgquad(
                        v_grid_points)

        self.v_quad_points = numpy.reshape(
                self.v_quad_points_1d,
                (len(self.v_quad_points_1d), 1))
        self.v_quad_weights = self.v_quad_points

        from numpy import dot
        from spyctral.common.indexing import integer_range
        ps = wienerfun.eval.genwienerw(self.v_quad_points_1d,
        integer_range(v_grid_points))
        dps = wienerfun.eval.dgenwienerw(self.v_quad_points_1d,
        integer_range(v_grid_points))

        diffmat = dot(dps,ps.T.conj()*
                          self.v_quad_weights_1d)

        from hedge.pde import StrongAdvectionOperator
        from hedge.data import \
                TimeConstantGivenFunction, \
                ConstantGivenFunction
        self.x_adv_operators = [
                StrongAdvectionOperator(v,
                    inflow_u=TimeConstantGivenFunction(
                        ConstantGivenFunction()),
                    flux_type="upwind")
                for v in self.v_quad_points]

    def op_template(self):
        from hedge.optemplate import \
                make_vector_field

        densities = make_vector_field(
                len(self.v_quad_points))

        from hedge.tools import make_obj_array
        return make_obj_array([
                adv_op*densities[i]
                for i, adv_op in enumerate(
                    self.x_adv_operators)
                ])

    def bind(self, discr):
        optemplate = self.op_template()






def main():
    from hedge.timestep import RK4TimeStepper
    from hedge.tools import mem_checkpoint
    from math import sin, cos, pi, sqrt
    from math import floor

    from hedge.backends import guess_run_context
    rcon = guess_run_context()

    def f(x):
        return sin(pi*x)

    def u_analytic(x, el, t):
        return f((-numpy.dot(v, x)/norm_v+t*norm_v))

    def boundary_tagger(vertices, el, face_nr, all_v):
        if numpy.dot(el.face_normals[face_nr], v) < 0:
            return ["inflow"]
        else:
            return ["outflow"]

    if rcon.is_head_rank:
        from hedge.mesh import make_uniform_1d_mesh
        mesh = make_uniform_1d_mesh(0, 2, 10, periodic=True)

    if rcon.is_head_rank:
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    discr = rcon.make_discretization(mesh_data, order=4)
    vis_discr = discr

    from hedge.visualization import VtkVisualizer, SiloVisualizer
    vis = VtkVisualizer(vis_discr, rcon, "fld")
    #vis = SiloVisualizer(vis_discr, rcon)

    # operator setup ----------------------------------------------------------
    op = VlasovOperator()

    u = discr.interpolate_volume_function(lambda x, el: u_analytic(x, el, 0))

    # timestep setup ----------------------------------------------------------
    stepper = RK4TimeStepper()

    dt = discr.dt_factor(op.max_eigenvalue())
    nsteps = int(700/dt)

    if rcon.is_head_rank:
        print "%d elements, dt=%g, nsteps=%d" % (
                len(discr.mesh.elements),
                dt,
                nsteps)

    # diagnostics setup -------------------------------------------------------
    from pytools.log import LogManager, \
            add_general_quantities, \
            add_simulation_quantities, \
            add_run_info

    logmgr = LogManager("advection.dat", "w", rcon.communicator)
    add_run_info(logmgr)
    add_general_quantities(logmgr)
    add_simulation_quantities(logmgr, dt)
    discr.add_instrumentation(logmgr)

    stepper.add_instrumentation(logmgr)

    from hedge.log import Integral, LpNorm
    u_getter = lambda: u
    logmgr.add_quantity(Integral(u_getter, discr, name="int_u"))
    logmgr.add_quantity(LpNorm(u_getter, discr, p=1, name="l1_u"))
    logmgr.add_quantity(LpNorm(u_getter, discr, name="l2_u"))

    logmgr.add_watches(["step.max", "t_sim.max", "l2_u", "t_step.max"])

    # timestep loop -----------------------------------------------------------
    rhs = op.bind(discr)
    for step in xrange(nsteps):
        logmgr.tick()

        t = step*dt

        if step % 5 == 0:
            visf = vis.make_file("fld-%04d" % step)
            vis.add_data(visf, [ ("u", u), ],
                        time=t,
                        step=step
                        )
            visf.close()


        u = stepper(u, t, dt, rhs)

    vis.close()

    logmgr.tick()
    logmgr.save()


if __name__ == "__main__":
    main()
