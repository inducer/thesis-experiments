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
from hedge.models import Operator




class GradientOperator(Operator):
    def __init__(self, dimensions):
        self.dimensions = dimensions

    def flux(self):
        from hedge.flux import make_normal, FluxScalarPlaceholder
        u = FluxScalarPlaceholder()

        normal = make_normal(self.dimensions)
        return u.int*normal - u.avg*normal

    def op_template(self):
        from hedge.mesh import TAG_ALL
        from hedge.optemplate import Field, BoundaryPair, \
                make_nabla, InverseMassOperator, get_flux_operator

        u = Field("u")
        bc = Field("bc")

        nabla = make_nabla(self.dimensions)
        flux_op = get_flux_operator(self.flux())

        return nabla*u - InverseMassOperator()(
                flux_op(u) +
                flux_op(BoundaryPair(u, bc, TAG_ALL)))

    def bind(self, discr):
        compiled_op_template = discr.compile(self.op_template())

        def op(u):
            from hedge.mesh import TAG_ALL

            return compiled_op_template(u=u,
                    bc=discr.boundarize_volume_field(u, TAG_ALL))

        return op




class DivergenceOperator(Operator):
    def __init__(self, dimensions, subset=None):
        self.dimensions = dimensions

        if subset is None:
            self.subset = dimensions * [True,]
        else:
            # chop off any extra dimensions
            self.subset = subset[:dimensions]

        from hedge.tools import count_subset
        self.arg_count = count_subset(self.subset)

    def flux(self):
        from hedge.flux import make_normal, FluxVectorPlaceholder

        v = FluxVectorPlaceholder(self.arg_count)

        normal = make_normal(self.dimensions)

        flux = 0
        idx = 0

        for i, i_enabled in enumerate(self.subset):
            if i_enabled and i < self.dimensions:
                flux += (v.int-v.avg)[idx]*normal[i]
                idx += 1

        return flux

    def op_template(self):
        from hedge.mesh import TAG_ALL
        from hedge.optemplate import make_vector_field, BoundaryPair, \
                get_flux_operator, make_nabla, InverseMassOperator

        nabla = make_nabla(self.dimensions)
        m_inv = InverseMassOperator()

        v = make_vector_field("v", self.arg_count)
        bc = make_vector_field("bc", self.arg_count)

        local_op_result = 0
        idx = 0
        for i, i_enabled in enumerate(self.subset):
            if i_enabled and i < self.dimensions:
                local_op_result += nabla[i]*v[idx]
                idx += 1

        flux_op = get_flux_operator(self.flux())

        return local_op_result - m_inv(
                flux_op(v) +
                flux_op(BoundaryPair(v, bc, TAG_ALL)))

    def bind(self, discr):
        compiled_op_template = discr.compile(self.op_template())

        def op(v):
            from hedge.mesh import TAG_ALL
            return compiled_op_template(v=v,
                    bc=discr.boundarize_volume_field(v, TAG_ALL))

        return op

class TVLimiter:
    def __init__(self, discr, vis):
        self.discr = discr
        self.vis = vis

        self.bound_grad = GradientOperator(discr.dimensions).bind(discr)
        self.bound_div = DivergenceOperator(discr.dimensions).bind(discr)
        self.stage = 0

        from hedge.discretization import Filter, ExponentialFilterResponseFunction
        self.mode_filter = Filter(self.discr,
                ExponentialFilterResponseFunction(min_amplification=0.2, order=1))

    def __call__(self, g):
        tau = 1/8
        lambda_ = 1

        grad = self.bound_grad
        div = self.bound_div

        from pytools.obj_array import make_obj_array
        p = make_obj_array(
                self.discr.volume_zeros(shape=(2,)))

        def two_norm(p):
            return numpy.sqrt(sum(p_i**2 for p_i in p)+100)

        for i in range(10):
            gd_term = grad(div(p) - g/lambda_)
            denominator = 1 + tau*two_norm(gd_term)
            p = make_obj_array([
                (p_i + tau*gd_term_i)/denominator
                for p_i, gd_term_i in zip(p, gd_term)])

            #p = self.mode_filter(p)

            if True:
                visf = self.vis.make_file(
                        "tvlimit-debug-%05d-%03d" % (self.stage, i))
                self.vis.add_data(visf, 
                        [
                            ("g", g), 
                            ("grad_g_over_lambda", -grad(g/lambda_)), 
                            ("gd_term", gd_term), 
                            ("denominator", denominator), 
                            ("p", p), 
                            ("lambda_div_p", lambda_*div(p))
                            ],
                            )
                visf.close()

                #raw_input("Enter:")

        self.stage += 1

        return g - lambda_*div(p)



def main(write_output=True, flux_type_arg="upwind"):
    from hedge.tools import mem_checkpoint
    from math import sin, cos, pi, sqrt
    from math import floor

    from hedge.backends import guess_run_context
    rcon = guess_run_context()

    def f(x):
        #return int(x % 2)
        return numpy.tanh(5*x)

    def u_analytic(x, el, t):
        return f((-numpy.dot(v, x)/norm_v+t*norm_v))

    def boundary_tagger(vertices, el, face_nr, all_v):
        if numpy.dot(el.face_normals[face_nr], v) < 0:
            return ["inflow"]
        else:
            return ["outflow"]

    dim = 1

    if dim == 1:
        v = numpy.array([1])
        if rcon.is_head_rank:
            from hedge.mesh.generator import make_uniform_1d_mesh
            mesh = make_uniform_1d_mesh(-1, 1, 10,
                    boundary_tagger=boundary_tagger)
    elif dim == 2:
        v = numpy.array([2,0])
        if rcon.is_head_rank:
            from hedge.mesh.generator import make_disk_mesh
            mesh = make_disk_mesh(boundary_tagger=boundary_tagger)

    norm_v = la.norm(v)

    if rcon.is_head_rank:
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    discr = rcon.make_discretization(mesh_data, order=10)
    vis_discr = discr

    from hedge.visualization import SiloVisualizer
    if write_output:
        vis = SiloVisualizer(vis_discr, rcon)

    # operator setup ----------------------------------------------------------
    from hedge.data import \
            ConstantGivenFunction, \
            TimeConstantGivenFunction, \
            TimeDependentGivenFunction
    from hedge.models.advection import StrongAdvectionOperator, WeakAdvectionOperator
    op = WeakAdvectionOperator(v, 
            inflow_u=TimeDependentGivenFunction(u_analytic),
            flux_type=flux_type_arg)

    u = discr.interpolate_volume_function(lambda x, el: u_analytic(x, el, 0))

    # timestep setup ----------------------------------------------------------
    limiter = TVLimiter(discr, vis)
    from hedge.timestep import SSPRK3TimeStepper
    stepper = SSPRK3TimeStepper(limiter=limiter)

    if rcon.is_head_rank:
        print "%d elements" % len(discr.mesh.elements)

    # diagnostics setup -------------------------------------------------------
    from pytools.log import LogManager, \
            add_general_quantities, \
            add_simulation_quantities, \
            add_run_info

    if write_output:
        log_file_name = "advection.dat"
    else:
        log_file_name = None

    logmgr = LogManager(log_file_name, "w", rcon.communicator)
    add_run_info(logmgr)
    add_general_quantities(logmgr)
    add_simulation_quantities(logmgr)
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

    try:
        from hedge.timestep import times_and_steps
        step_it = times_and_steps(
                final_time=3, logmgr=logmgr,
                max_dt_getter=lambda t: op.estimate_timestep(discr,
                    stepper=stepper, t=t, fields=u))

        for step, t, dt in step_it:
            if step % 1 == 0 and write_output:
                visf = vis.make_file("fld-%04d" % step)
                vis.add_data(visf, [ ("u", u), ],
                            time=t,
                            step=step
                            )
                visf.close()

            u = stepper(u, t, dt, rhs)

    finally:
        if write_output:
            vis.close()

        logmgr.save()

    true_u = discr.interpolate_volume_function(lambda x, el: u_analytic(x, el, t))
    print discr.norm(u-true_u)
    assert discr.norm(u-true_u) < 1e-2



if __name__ == "__main__":
    main()




# entry points for py.test ----------------------------------------------------
def test_advection():
    from pytools.test import mark_test
    mark_long = mark_test.long

    for flux_type in ["upwind", "central", "lf"]:
        yield "advection with %s flux" % flux_type, \
                mark_long(main), False, flux_type
