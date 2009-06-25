from __future__ import division, with_statement
import numpy
import numpy.linalg as la




class VlasovOperator:
    def __init__(self, *args, **kwargs):
        from v_discr import VelocityDiscretization
        self.v_discr = VelocityDiscretization(*args, **kwargs)

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

    def visualize_densities_with_matplotlib(self, discr, filename, densities):
        left, right = discr.mesh.bounding_box()
        left = left[0]
        right = right[0]

        img_data = numpy.array(list(densities))
        from matplotlib.pyplot import imshow, savefig, \
                xlabel, ylabel, colorbar, clf, yticks

        clf()
        imshow(img_data, extent=(left, right, -1, 1))

        xlabel("$x$")
        ylabel("$v$")

        ytick_step = int(round(self.v_discr.grid_size / 8))
        yticks(
                numpy.linspace(
                    -1, 1, self.v_discr.grid_size)[::ytick_step],
                ["%.3f" % vn for vn in 
                    self.v_discr.quad_points_1d[::ytick_step]])
        colorbar()

        savefig(filename)

    def visualize_densities_with_silo(self, discr, filename, densities):
        from pylo import SiloFile, DB_NODECENT

        scheme_dtype = self.v_discr.diffmat.dtype
        is_complex = (numpy.complexfloating in scheme_dtype.type.__mro__)
        f_data = numpy.array(list(densities), dtype=scheme_dtype)

        with SiloFile(filename) as silo:
            silo.put_quadmesh("xvmesh", [
                discr.nodes.reshape((len(discr.nodes),)),
                self.v_discr.quad_points_1d,
                ])

            if is_complex:
                silo.put_quadvar1("f_r", "xvmesh", f_data.real.copy(), f_data.shape, 
                        DB_NODECENT)
                silo.put_quadvar1("f_i", "xvmesh", f_data.imag.copy(), f_data.shape, 
                        DB_NODECENT)
            else:
                silo.put_quadvar1("f", "xvmesh", f_data, f_data.shape, 
                        DB_NODECENT)


def main():
    from hedge.timestep import RK4TimeStepper
    from hedge.tools import mem_checkpoint
    from math import sin, cos, pi, sqrt, exp
    from math import floor

    from hedge.backends import guess_run_context
    rcon = guess_run_context()

    from hedge.mesh import make_uniform_1d_mesh
    mesh = make_uniform_1d_mesh(0, 2*pi, 20, periodic=True)

    discr = rcon.make_discretization(mesh, order=4)

    # operator setup ----------------------------------------------------------
    op = VlasovOperator(grid_size=20, filter_type="exponential",
            hard_scale=5, bounded_fraction=0.8)

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
            len(discr.mesh.elements), dt, nsteps)

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

        if step % 20 == 0:
            #op.visualize_densities_with_matplotlib(discr,
                    #"vlasov-%04d.png" % step, densities)
            op.visualize_densities_with_silo(discr,
                    "vlasov-%04d.silo" % step, densities)

        densities = stepper(densities, t, dt, rhs)

    vis.close()

    logmgr.tick()
    logmgr.save()


if __name__ == "__main__":
    main()
