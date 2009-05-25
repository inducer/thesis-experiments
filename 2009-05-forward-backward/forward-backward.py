# Hedge - the Hybrid'n'Easy DG Environment
# Copyright (C) 2009 Andreas Kloeckner
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




def main() :
    from math import sin, cos, pi, exp, sqrt

    from hedge.backends import guess_run_context
    #rcon = guess_run_context(disable=set(["cuda"]))
    rcon = guess_run_context()

    # operator setup ----------------------------------------------------------
    def c_speed(x, el):
        return 1
        if la.norm(x[1]) < 0.4:
            return 1
        else:
            return 0.5**0.5

    from hedge.data import \
            TimeIntervalGivenFunction, \
            TimeConstantGivenFunction, \
            GivenFunction
    from hedge.mesh import TAG_ALL, TAG_NONE
    from hedge.pde import VariableVelocityStrongWaveOperator

    op_kwargs = dict(
            c=TimeConstantGivenFunction(
                GivenFunction(c_speed)), 
            dimensions=2, 
            dirichlet_tag=TAG_ALL,
            neumann_tag=TAG_NONE,
            radiation_tag=TAG_NONE,
            flux_type="upwind",
            )

    fwd_op = VariableVelocityStrongWaveOperator(
            time_sign=1,
            **op_kwargs)
    bwd_op = VariableVelocityStrongWaveOperator(
            time_sign=-1,
            **op_kwargs)

    # mesh/discretization setup -----------------------------------------------
    from hedge.mesh import make_rect_mesh_with_corner

    if rcon.is_head_rank:
        mesh = make_rect_mesh_with_corner(a=(-1,-1),b=(1,1),max_area=0.006)
        #coarse_mesh = make_rect_mesh_with_corner(a=(-1,-1),b=(1,1),max_area=0.01)

    if rcon.is_head_rank:
        #print "fine: %d elements, coarse: %d elements" % (
                #len(fine_mesh.elements), len(coarse_mesh.elements))
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    discr = rcon.make_discretization(mesh_data, order=4,
            default_scalar_type=numpy.float64,
            #debug=["cuda_no_plan"],
            #init_cuda=True,
            #tune_for=fwd_op.op_template()
            )
    coarse_discr = rcon.make_discretization(mesh_data, order=2,
            default_scalar_type=numpy.float64
            #debug=["cuda_no_plan"],
            #init_cuda=False,
            )

    def to_numpy(fld):
        return discr.convert_volume(fld, kind="numpy")
    def from_numpy(fld):
        return discr.convert_volume(fld, kind=discr.compute_kind)

    from hedge.visualization import SiloVisualizer, VtkVisualizer
    vis = SiloVisualizer(discr, rcon)

    def sender_ic_u(x, el):
        x = x - numpy.array([0.5, -0.5])
        return exp(-numpy.dot(x, x)*64)

    def receiver_ic_u(x, el):
        x = x - numpy.array([-0.5, 0.5])
        return exp(-numpy.dot(x, x)*64)


    from hedge.tools import join_fields
    sender_fields = join_fields(discr.interpolate_volume_function(sender_ic_u),
            [discr.volume_zeros() for i in range(discr.dimensions)])

    dt = discr.dt_factor(1)/4
    nsteps = int(6/dt)
    if rcon.is_head_rank:
        print "dt", dt
        print "nsteps", nsteps

    # diagnostics setup -------------------------------------------------------
    from pytools.log import LogManager, \
            add_general_quantities, \
            add_simulation_quantities, \
            add_run_info

    logmgr = LogManager("wave.dat", "w", rcon.communicator)
    add_run_info(logmgr)
    add_general_quantities(logmgr)
    add_simulation_quantities(logmgr, dt)
    discr.add_instrumentation(logmgr)

    from pytools.log import IntervalTimer
    vis_timer = IntervalTimer("t_vis", "Time spent visualizing")
    logmgr.add_quantity(vis_timer)

    logmgr.add_watches(["step.max", "t_sim.max", "t_step.max"])

    # forward timestep loop ---------------------------------------------------
    fwd_rhs = fwd_op.bind(discr)

    from hedge.timestep import RK4TimeStepper
    stepper = RK4TimeStepper()

    def do_fw_vis(step, t, fields):
        visf = vis.make_file("fld-%04d" % step)

        vis.add_data(visf,
                [(nm, to_numpy(fld)) for (nm, fld) in [
                    ("s_u", fields[0]),
                    ("s_v", fields[1:]), 
                    ("c", fwd_op.c.volume_interpolant(0, discr)), 
                ]],
                time=t,
                step=step)
        visf.close()

    def tsloop(fields):
        for step in range(nsteps):
            logmgr.tick()

            t = step*dt

            if step % 10 == 0:
                do_fw_vis(step, t, fields)

            fields = stepper(fields, t, dt, fwd_rhs)
        do_fw_vis(step, t, fields)

        return fields

    sender_fields = tsloop(sender_fields)
    logmgr.save()

    # backward timestep loop --------------------------------------------------
    print "SWITCH DIRECTIONS"
    bwd_rhs = bwd_op.bind(discr)

    from hedge.discretization import Projector
    fine_to_coarse = Projector(discr, coarse_discr)
    coarse_to_fine = Projector(coarse_discr, discr)

    receiver_source_fine = \
            discr.interpolate_volume_function(receiver_ic_u)
    receiver_source_coarse = fine_to_coarse(to_numpy(receiver_source_fine))

    receiver_cf_diff = (from_numpy(
            coarse_to_fine(receiver_source_coarse))
            - receiver_source_fine)

    from hedge.tools import join_fields
    all_fields = join_fields(
            sender_fields, # snd
            receiver_source_fine, # rec
            [discr.volume_zeros() for i in range(discr.dimensions)],
            receiver_cf_diff, # rec_diff
            [discr.volume_zeros() for i in range(discr.dimensions)],
            discr.volume_zeros() # correlation
            )

    from hedge.timestep import RK4TimeStepper
    all_stepper = RK4TimeStepper()

    def combined_rhs(t, flds):
        d = discr.dimensions
        sender = all_fields[:(d+1)]
        recv = all_fields[d+1:2*(d+1)]
        recv_diff = all_fields[2*(d+1):3*(d+1)]
        corr = all_fields[3*(d+1)]

        return join_fields(
                bwd_rhs(t, sender),
                bwd_rhs(t, recv),
                bwd_rhs(t, recv_diff),
                sender[0]*recv[0])

    def do_bw_vis():
        d = discr.dimensions
        visf = vis.make_file("bwd-fld-%04d" % step)

        s_u = all_fields[0]
        r_u = all_fields[1+d]

        vis.add_data(visf,
                [(nm, to_numpy(fld)) for nm, fld in [
                    ("s_u", all_fields[0]),
                    ("s_v", all_fields[1:1+d]), 
                    ("r_u", all_fields[1+d]),
                    ("r_v", all_fields[2+d:2+2*d]), 
                    ("rdiff_u", all_fields[2*(1+d)]),
                    ("rdiff_v", all_fields[2*(1+d)+1:3*(1+d)]), 
                    ("corr", all_fields[3*(1+d)]), 
                    ("claimed_dt_corr", s_u*r_u*discr.interpolate_volume_function(receiver_ic_u)), 
                    ("c", bwd_op.c.volume_interpolant(0, discr)), 
                ]],
                time=t,
                step=step)
        visf.close()

    for step in range(nsteps):
        logmgr.tick()

        t = step*dt

        if step % 100 == 0:
            do_bw_vis()

        all_fields = all_stepper(all_fields, t, dt, combined_rhs)

    do_bw_vis()

    # finish up ---------------------------------------------------------------
    vis.close()

    logmgr.tick()
    logmgr.save()

    coarse_discr.close()
    discr.close()

if __name__ == "__main__":
    main()

