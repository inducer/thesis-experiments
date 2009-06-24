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
    def __init__(self, v_grid_size=42, v_method="wiener",
            hard_scale=None):
        # we're not hoping to invert the d.m. any more.
        # assert v_grid_size % 2 == 1
        # otherwise differentiation becomes non-invertible

        from v_discr import VelocityDiscretization
        self.v_discr = VelocityDiscretization(
                v_grid_size, v_method, hard_scale)

        from hedge.pde import StrongAdvectionOperator
        from hedge.data import \
                TimeConstantGivenFunction, \
                ConstantGivenFunction

        self.x_adv_operators = [
                StrongAdvectionOperator(v,
                    inflow_u=TimeConstantGivenFunction(
                        ConstantGivenFunction()),
                    flux_type="upwind")
                for v in self.v_discr.quad_points]

    def op_template(self):
        from hedge.optemplate import \
                make_vector_field

        f = make_vector_field("f",
                len(self.v_discr.quad_points))

        def adv_op_template(adv_op, f_of_v):
            from hedge.optemplate import Field, pair_with_boundary, \
                    get_flux_operator, make_nabla, InverseMassOperator

            #bc_in = Field("bc_in")

            nabla = make_nabla(adv_op.dimensions)

            return (
                    -numpy.dot(adv_op.v, nabla*f_of_v)
                    + InverseMassOperator()*(
                        get_flux_operator(adv_op.flux()) * f_of_v
                        #+ flux_op * pair_with_boundary(f_of_v, bc_in, self.inflow_tag)
                        )
                    )

        v_discr = self.v_discr

        from hedge.tools import make_obj_array
        f_v = make_obj_array([
            sum(v_discr.diffmat[i,j]*f[j] for j in range(v_discr.grid_size))
            for i in range(v_discr.grid_size)
            ])

        return make_obj_array([
                adv_op_template(adv_op, f[i])
                for i, adv_op in enumerate(
                    self.x_adv_operators)
                ]) + f_v

    def bind(self, discr):
        compiled_op_template = discr.compile(self.op_template())

        def rhs(t, densities):
            return compiled_op_template(f=densities)

        return rhs


    def max_eigenvalue(self):
        return max(la.norm(v) for v in self.v_discr.quad_points)




def main():
    from hedge.timestep import RK4TimeStepper
    from hedge.tools import mem_checkpoint
    from math import sin, cos, pi, sqrt, exp
    from math import floor

    from hedge.backends import guess_run_context
    rcon = guess_run_context()

    def f(x):
        return sin(x)

    def u_analytic(x, el, t):
        return f((-numpy.dot(v, x)/norm_v+t*norm_v))

    left = 0
    right = 2*pi

    if rcon.is_head_rank:
        from hedge.mesh import make_uniform_1d_mesh
        mesh = make_uniform_1d_mesh(left, right, 10, periodic=True)

    if rcon.is_head_rank:
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    discr = rcon.make_discretization(mesh_data, order=4)
    vis_discr = discr

    from hedge.visualization import SiloVisualizer
    vis = SiloVisualizer(vis_discr, rcon)

    # operator setup ----------------------------------------------------------
    op = VlasovOperator()

    sine_vec = discr.interpolate_volume_function(lambda x, el: sin(x[0]))
    from hedge.tools import make_obj_array

    densities = make_obj_array([
        sine_vec.copy()*exp(-(0.8*v[0]**2))
        for v in op.v_discr.quad_points])

    # timestep setup ----------------------------------------------------------
    stepper = RK4TimeStepper()

    dt = discr.dt_factor(op.max_eigenvalue())
    nsteps = int(700/dt)

    print "%d elements, dt=%g, nsteps=%d" % (
            len(discr.mesh.elements),
            dt,
            nsteps)

    # diagnostics setup -------------------------------------------------------
    from pytools.log import LogManager, \
            add_general_quantities, \
            add_simulation_quantities, \
            add_run_info

    logmgr = LogManager("vlasov.dat", "w", rcon.communicator)
    add_run_info(logmgr)
    add_general_quantities(logmgr)
    add_simulation_quantities(logmgr, dt)
    discr.add_instrumentation(logmgr)

    stepper.add_instrumentation(logmgr)

    logmgr.add_watches(["step.max", "t_sim.max", "t_step.max"])

    # timestep loop -----------------------------------------------------------
    rhs = op.bind(discr)

    for step in xrange(nsteps):
        logmgr.tick()

        t = step*dt

        if step % 5 == 0:
            img_data = numpy.array(list(densities))
            from matplotlib.pyplot import imshow, savefig, \
                    xlabel, ylabel, colorbar, clf, yticks

            clf()
            imshow(img_data, extent=(left, right, -1, 1))

            xlabel("$x$")
            ylabel("$v$")

            ytick_step = int(round(op.v_discr.grid_size / 8))
            yticks(
                    numpy.linspace(
                        -1, 1, op.v_discr.grid_size)[::ytick_step],
                    ["%.3f" % vn for vn in 
                        op.v_discr.quad_points_1d[::ytick_step]])
            colorbar()

            savefig("vlasov-%04d.png" % step)

        if False and step % 5 == 0:
            visf = vis.make_file("fld-%04d" % step)
            vis.add_data(visf, [ ("u", u), ],
                        time=t,
                        step=step
                        )
            visf.close()

        densities = stepper(densities, t, dt, rhs)

    vis.close()

    logmgr.tick()
    logmgr.save()


if __name__ == "__main__":
    main()
