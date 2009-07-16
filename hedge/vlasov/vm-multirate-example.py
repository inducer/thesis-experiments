from __future__ import division, with_statement
import numpy
import numpy.linalg as la




def main():
    from hedge.tools import mem_checkpoint
    from math import sin, cos, pi, sqrt, exp
    from math import floor

    from hedge.backends import guess_run_context
    rcon = guess_run_context()

    from hedge.mesh import make_uniform_1d_mesh
    mesh = make_uniform_1d_mesh(-pi, pi, 20, periodic=True)

    discr = rcon.make_discretization(mesh, order=4,
            debug=[
                #"print_op_code",
                "jit_wait_on_compile_error",
                "jit_dont_optimize_large_exprs",
                ])

    from hedge.visualization import SiloVisualizer
    vis = SiloVisualizer(discr, rcon)

    # operator setup ----------------------------------------------------------
    from pyrticle.units import SIUnitsWithUnityConstants
    units = SIUnitsWithUnityConstants()

    from hedge.models.em import TE1DMaxwellOperator
    max_op = TE1DMaxwellOperator(
            units.EPSILON0, units.MU0, 
            flux_type=1, dimensions=1)

    from vlasov import VlasovMaxwellOperator
    vlas_op = VlasovMaxwellOperator(
            x_dim=mesh.dimensions, v_dim=2,
            maxwell_op=max_op, units=units,
            species_mass=units.EL_MASS, 
            species_charge=-units.EL_CHARGE,
            grid_size=16, filter_type="exponential",
            hard_scale=0.2, bounded_fraction=0.8,
            filter_parameters=dict(eta_cutoff=0.3))

    base_vec = discr.interpolate_volume_function(lambda x, el: cos(0.5*x[0]))
    #base_vec = discr.interpolate_volume_function(lambda x, el: 1)
    from hedge.tools import make_obj_array, join_fields

    from vlasov import find_multirate_split
    v_points = list(vlas_op.v_points)
    substep_counts, rate_index_groups = find_multirate_split(v_points, 2)

    if False:
        for rig in rate_index_groups:
            print [vlas_op.p_grid.tuple_from_linear(i) for i in rig], \
                    max(la.norm(v_points[i]) for i in rig)
        print "substep counts:", substep_counts

    densities = make_obj_array([
        base_vec*exp(-(0.5*numpy.dot(v, v)))#*v[0]
        for v in vlas_op.v_points])

    densities_by_rate_group = [densities[rig] for rig in rate_index_groups]

    # em fields and fastest densities propagate at about the
    # speed of light, so stick them in the same rate group
    fields = ([join_fields(
            max_op.assemble_eh(discr=discr),
            densities_by_rate_group[0])]
            + densities_by_rate_group[1:])

    mfc = vlas_op.maxwell_field_count
    joint_mr_to_field_map = numpy.empty(len(vlas_op.p_grid)+mfc, dtype=numpy.intp)
    joint_mr_to_field_map[:mfc] = numpy.arange(mfc)
    base = 0
    for rig in rate_index_groups:
        joint_mr_to_field_map[mfc+rig] = mfc+base+numpy.arange(len(rig))
        base += len(rig)

    print joint_mr_to_field_map

    # timestep setup ----------------------------------------------------------
    ab_order = 3
    from hedge.timestep.ab import AdamsBashforthTimeStepper
    from hedge.timestep.multirate_ab import TwoRateAdamsBashforthTimeStepper
    stepper = TwoRateAdamsBashforthTimeStepper(
            "fastest_first_1a", 
            large_dt=discr.dt_factor(vlas_op.max_eigenvalue(),
                AdamsBashforthTimeStepper, ab_order),
            order=3, substep_count=substep_counts[0])

    nsteps = int(10/stepper.large_dt)

    print "%d elements, dt=%g, nsteps=%d" % (
            len(discr.mesh.elements), stepper.large_dt, nsteps)

    # diagnostics setup -------------------------------------------------------
    from pytools.log import LogManager, \
            add_general_quantities, \
            add_simulation_quantities, \
            add_run_info

    logmgr = LogManager("vmax.dat", "w", rcon.communicator)
    add_run_info(logmgr)
    add_general_quantities(logmgr)
    add_simulation_quantities(logmgr, stepper.large_dt)
    discr.add_instrumentation(logmgr)

    logmgr.add_watches(["step.max", "t_sim.max", "t_step.max"])

    # timestep loop -----------------------------------------------------------
    def bind(op_template):
        compiled = discr.compile(op_template)

        def rhs(t, q_fast, q_slow):
            q = join_fields(q_fast(), q_slow())[joint_mr_to_field_map]
            max_w = q[:vlas_op.maxwell_field_count]
            densities = q[vlas_op.maxwell_field_count:]

            return compiled(w=max_w, f=densities)

        return rhs

    from hedge.tools import join_fields
    from vlasov import split_optemplate_for_multirate
    rhs_optemplates = split_optemplate_for_multirate(
                join_fields(
                    [0]*vlas_op.maxwell_field_count, # leave Maxwell fields alone
                    vlas_op.make_densities_placeholder()),
                vlas_op.op_template(),
                [numpy.hstack([
                    numpy.arange(vlas_op.maxwell_field_count),
                    rate_index_groups[0] + vlas_op.maxwell_field_count
                    ])
                    ]
                + [ig+vlas_op.maxwell_field_count for ig in rate_index_groups[1:]])

    if False:
        for i, ot in enumerate(rhs_optemplates):
            print "--------------------------------------------"
            print i
            for j, otc in enumerate(ot):
                print j, otc
                print

    rhss = [bind(optemplate) for optemplate in rhs_optemplates]

    j_op = vlas_op.bind(discr, vlas_op.j(vlas_op.make_densities_placeholder()))

    forces_ops = [vlas_op.bind(discr, force_op)
            for force_op in vlas_op.forces_T(
                vlas_op.make_densities_placeholder(),
                *vlas_op.make_maxwell_eh_placeholders())]

    def real_part(fld):
        from hedge.tools import with_object_array_or_scalar
        return with_object_array_or_scalar(lambda x: x.real, fld)

    try:
        for step in xrange(nsteps):
            logmgr.tick()

            t = step*stepper.large_dt

            if step % 20 == 0:
                with vis.make_file("vlasov-%04d" % step) as visf:
                    joint_fields = join_fields(*fields)[joint_mr_to_field_map]
                    e, h, densities = vlas_op.split_e_h_densities(
                            joint_fields)
                    from vlasov import add_xv_to_silo

                    AXES = ["x", "y", "z"]

                    add_xv_to_silo(visf, vlas_op, discr, [
                        ("f", densities),
                        ] + [
                        ("forces"+AXES[i], force_op(t, joint_fields))
                        for i, force_op in enumerate(forces_ops)
                        ])
                    vis.add_data(visf, [
                        ("e", real_part(e)),
                        ("h", real_part(h)),
                        ("j", real_part(j_op(t, joint_fields))),
                        ], time=t, step=step)

            fields = stepper(fields, t, rhss)
    finally:
        logmgr.close()
        discr.close()




if __name__ == "__main__":
    main()

