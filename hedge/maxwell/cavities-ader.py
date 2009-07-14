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

from hedge.models.em import MaxwellOperator




class ADERMaxwellOperator(MaxwellOperator):
    def __init__(self, *args, **kwargs):
        self.ader_order = kwargs.pop("ader_order")
        self.ader_dt = 1
        MaxwellOperator.__init__(self, *args, **kwargs)

    def op_template(self, w=None):
        from hedge.optemplate import make_common_subexpression

        w = self.field_placeholder(w)

        w_t = w
        for rk in range(self.ader_order, 1, -1):
            w_t = make_common_subexpression(
                    w - (self.ader_dt/rk) \
                            * self.local_derivatives(w_t))

        return w + self.ader_dt*MaxwellOperator.op_template(self, w_t)




def main(write_output=True, allow_features=None, flux_type_arg=1,
        bdry_flux_type_arg=None, extra_discr_args={}):
    import sys
    sys.path.append("../../../hedge/examples/maxwell")

    from hedge.element import TetrahedralElement
    from hedge.mesh import make_ball_mesh, make_cylinder_mesh, make_box_mesh
    from hedge.tools import EOCRecorder, to_obj_array
    from math import sqrt, pi
    from analytic_solutions import \
            check_time_harmonic_solution, \
            RealPartAdapter, \
            SplitComplexAdapter, \
            CylindricalFieldAdapter, \
            CylindricalCavityMode, \
            RectangularWaveguideMode, \
            RectangularCavityMode

    from hedge.backends import guess_run_context
    rcon = guess_run_context(allow_features)

    epsilon0 = 8.8541878176e-12 # C**2 / (N m**2)
    mu0 = 4*pi*1e-7 # N/A**2.
    epsilon = 1*epsilon0
    mu = 1*mu0

    eoc_rec = EOCRecorder()

    cylindrical = False
    periodic = False

    if cylindrical:
        R = 1
        d = 2
        mode = CylindricalCavityMode(m=1, n=1, p=1,
                radius=R, height=d,
                epsilon=epsilon, mu=mu)
        r_sol = CylindricalFieldAdapter(RealPartAdapter(mode))
        c_sol = SplitComplexAdapter(CylindricalFieldAdapter(mode))

        if rcon.is_head_rank:
            mesh = make_cylinder_mesh(radius=R, height=d, max_volume=0.01)
    else:
        if periodic:
            mode = RectangularWaveguideMode(epsilon, mu, (3,2,1))
            periodicity = (False, False, True)
        else:
            periodicity = None
        mode = RectangularCavityMode(epsilon, mu, (1,2,2))

        if rcon.is_head_rank:
            mesh = make_box_mesh(max_volume=0.0001, periodicity=periodicity)

    if rcon.is_head_rank:
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    #for order in [1,2,3,4,5,6]:
    for order in [4]:
        op = ADERMaxwellOperator(epsilon, mu, \
                flux_type=flux_type_arg, \
                bdry_flux_type=bdry_flux_type_arg,
                ader_order=order)

        discr = rcon.make_discretization(mesh_data, order=order,
                tune_for=op.op_template(),
                **extra_discr_args)

        from hedge.visualization import VtkVisualizer
        if write_output:
            vis = VtkVisualizer(discr, rcon, "em-%d" % order)

        mode.set_time(0)
        def get_true_field():
            return discr.convert_volume(
                to_obj_array(mode(discr)
                    .real.astype(discr.default_scalar_type).copy()),
                kind=discr.compute_kind)
        fields = get_true_field()

        dt = discr.dt_factor(op.max_eigenvalue()) / 4
        final_time = 0.5e-9
        nsteps = int(final_time/dt)+1
        dt = final_time/nsteps

        op.ader_dt = dt

        if rcon.is_head_rank:
            print "---------------------------------------------"
            print "order %d" % order
            print "---------------------------------------------"
            print "dt", dt
            print "nsteps", nsteps
            print "#elements=", len(mesh.elements)

        # diagnostics setup ---------------------------------------------------
        from pytools.log import LogManager, add_general_quantities, \
                add_simulation_quantities, add_run_info

        if write_output:
            log_file_name = "maxwell-%d.dat" % order
        else:
            log_file_name = None

        logmgr = LogManager(log_file_name, "w", rcon.communicator)

        add_run_info(logmgr)
        add_general_quantities(logmgr)
        add_simulation_quantities(logmgr, dt)
        discr.add_instrumentation(logmgr)

        from pytools.log import IntervalTimer
        vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
        logmgr.add_quantity(vis_timer)

        from hedge.log import EMFieldGetter, add_em_quantities
        #field_getter = EMFieldGetter(discr, op, lambda: fields)
        #add_em_quantities(logmgr, op, field_getter)

        logmgr.add_watches(["step.max", "t_sim.max", "t_step.max"])

        # timestep loop -------------------------------------------------------
        t = 0
        bound_op = op.bind(discr)

        try:
            for step in range(nsteps):
                logmgr.tick()

                if step % 1000 == 0 and write_output:
                    sub_timer = vis_timer.start_sub_timer()
                    e, h = op.split_eh(fields)
                    visf = vis.make_file("em-%d-%04d" % (order, step))
                    vis.add_data(visf,
                            [
                                ("e", 
                                    discr.convert_volume(e, kind="numpy")), 
                                ("h", 
                                    discr.convert_volume(h, kind="numpy")),],
                            time=t, step=step
                            )
                    visf.close()
                    sub_timer.stop().submit()

                fields = bound_op(t, fields)
                t += dt

            mode.set_time(t)

            eoc_rec.add_data_point(order, 
                    discr.norm(fields-get_true_field()))

        finally:
            if write_output:
                vis.close()

            logmgr.close()
            discr.close()

        if rcon.is_head_rank:
            print
            print eoc_rec.pretty_print("P.Deg.", "L2 Error")

    assert eoc_rec.estimate_order_of_convergence()[0,1] > 6





if __name__ == "__main__":
    if True:
        main(allow_features=["cuda"], extra_discr_args=dict(
            default_scalar_type=numpy.float32,
            debug=["dump_dataflow_graph", "cuda_no_plan_el_local"],
            ))
    else:
        main(write_output=False, extra_discr_args=dict(
            default_scalar_type=numpy.float32))
