from __future__ import division
import numpy
import numpy.linalg as la
from pytools import memoize_method




class VlasovOperatorBase:
    def __init__(self, units, species_mass, *args, **kwargs):
        from p_discr import MomentumDiscretization
        self.p_discr = MomentumDiscretization(*args, **kwargs)
        self.units = units
        self.species_mass = species_mass

        self.v_dim = 1

        self.velocity_points = [units.v_from_p(species_mass, p)
                for p in self.p_discr.quad_points]

        from hedge.models.advection import StrongAdvectionOperator
        self.x_adv_operators = [
                StrongAdvectionOperator(v, flux_type="upwind")
                for v in self.velocity_points]

    @memoize_method
    def make_densities_placeholder(self, base_number=0):
        from hedge.optemplate import make_vector_field
        return make_vector_field("f",
                range(base_number, 
                    base_number+len(self.p_discr.quad_points)))

    def op_template(self, forces_T, f=None):
        from hedge.optemplate import make_vector_field
        from hedge.tools import make_obj_array

        if f is None:
            f = self.make_densities_placeholder()

        def adv_op_template(adv_op, f_of_p):
            from hedge.optemplate import Field, pair_with_boundary, \
                    get_flux_operator, make_nabla, InverseMassOperator

            nabla = make_nabla(adv_op.dimensions)

            return (-numpy.dot(adv_op.v, nabla*f_of_p)
                    + InverseMassOperator()*(
                        get_flux_operator(adv_op.flux()) * f_of_p
                        ))

        p_discr = self.p_discr

        if hasattr(p_discr, "diff_function"):
            f_p = [p_discr.diff_function(f)]
        else:
            f_p = [make_obj_array([
                sum(p_discr.diffmat[i,j]*f[j] for j in range(p_discr.grid_size))
                for i in range(p_discr.grid_size)
                ])]

        return make_obj_array([
                adv_op_template(adv_op, f[i])
                for i, adv_op in enumerate(
                    self.x_adv_operators)
                ]) + sum(
                        forces_i*f_p_i
                        for forces_i, f_p_i in zip(forces_T, f_p))

    def max_eigenvalue(self):
        return max(la.norm(v) for v in self.velocity_points)


    def integral_dp(self, p_values):
        return sum(qw_i * v_i 
                for qw_i, v_i in zip(self.p_discr.quad_weights, values))

    @memoize_method
    def int_dv_quad_weights(self):
        weights = numpy.zeros(self.p_discr.grid_size, dtype=numpy.float64)
        c = self.units.VACUUM_LIGHT_SPEED
        for i, p in enumerate(self.p_discr.quad_points):
            p_square = numpy.dot(p, p)
            denom = p_square + c**2 * self.species_mass**2
            weights[i] =(
                    c/denom**0.5
                    - c*p_square / denom**1.5)

        return weights*self.p_discr.quad_weights

    def integral_dv(self, values):
        return sum(qw_i * v_i 
                for qw_i, v_i in zip(self.int_dv_quad_weights(), values))






class VlasovOperator(VlasovOperatorBase):
    def __init__(self, units, species_mass, forces_func, 
            *args, **kwargs):
        VlasovOperatorBase.__init__(self, units, species_mass, 
                *args, **kwargs)
        self.forces_func = forces_func

    def op_template(self):
        from hedge.optemplate import Field
        force_base = Field("force")

        from hedge.tools import make_obj_array
        forces_T = [make_obj_array([force_base[i][p_axis]
            for i, p in enumerate(self.p_discr.quad_points)])
            for p_axis in [0]]

        return VlasovOperatorBase.op_template(self, forces_T)

    def bind(self, discr):
        compiled_op_template = discr.compile(self.op_template())

        def rhs(t, densities):
            return compiled_op_template(
                    f=densities,
                    force=self.forces_func(t))

        return rhs




class VlasovMaxwellOperator(VlasovOperatorBase):
    def __init__(self, maxwell_op, units, species_mass, species_charge, 
            *args, **kwargs):
        VlasovOperatorBase.__init__(self, units, species_mass, *args, **kwargs)

        self.maxwell_op = maxwell_op
        self.species_charge = species_charge

        from hedge.tools import count_subset
        self.maxwell_field_count = \
                count_subset(self.maxwell_op.get_eh_subset())

    def op_template(self):
        max_op = self.maxwell_op

        from hedge.optemplate import make_vector_field
        max_fields = make_vector_field("w", self.maxwell_field_count)
        max_e, max_h = max_op.split_eh(max_fields)

        densities = self.make_densities_placeholder()

        j = self.integral_dv(
                (self.species_charge*v)*f_i
                for v, f_i in zip(self.velocity_points, densities))

        # build forces --------------------------------------------------------
        from hedge.optemplate import make_common_subexpression as cse
        max_b = cse(max_op.mu * max_h)

        v_subset = [True] * self.v_dim + [False] * (3-self.v_dim)
        e_subset = max_op.get_eh_subset()[:3]
        b_subset = max_op.get_eh_subset()[3:]
        from hedge.tools import SubsettableCrossProduct
        v_b_cross = SubsettableCrossProduct(
                v_subset, b_subset, v_subset)

        def pick_subset(tgt_subset, vec_subset, vec):
            from hedge.tools import count_subset
            result = numpy.zeros(count_subset(tgt_subset), 
                    dtype = object)

            tgt_idx = 0
            for ts, vs, v_i in zip(tgt_subset, vec_subset, vec):
                if ts:
                    if vs:
                        result[tgt_idx] = v_i
                    else:
                        result[tgt_idx] = 0
                    tgt_idx += 1

            return result

        el_force = pick_subset(v_subset, e_subset,
                cse(self.species_charge * max_e))

        from hedge.tools import make_obj_array
        q_times_b = cse(self.species_charge*max_b)
        forces = make_obj_array([
            el_force + v_b_cross(v, q_times_b)
            for v in self.velocity_points])

        forces_T = [[forces[v_node_idx][v_axis]
            for v_node_idx in range(self.p_discr.grid_size)]
            for v_axis in range(self.v_dim)]

        # assemble rhs --------------------------------------------------------
        max_e_rhs, max_h_rhs = max_op.split_eh(
                max_op.op_template(max_fields))
        vlasov_rhs = VlasovOperatorBase.op_template(
                self, forces_T, densities)

        from hedge.tools import join_fields
        return join_fields(
                max_e_rhs + j/max_op.epsilon,
                max_h_rhs,
                vlasov_rhs)

    def bind(self, discr):
        compiled = discr.compile(self.op_template())

        def rhs(t, q):
            max_w = q[:self.maxwell_field_count]
            densities = q[self.maxwell_field_count:]

            return compiled(w=max_w, f=densities)

        return rhs

    def split_e_h_densities(self, fields):
        max_w = fields[:self.maxwell_field_count]
        e, h = self.maxwell_op.split_eh(max_w)
        densities = fields[self.maxwell_field_count:]
        return e, h, densities

    def max_eigenvalue(self):
        return max(
                max(la.norm(v) for v in self.velocity_points),
                self.maxwell_op.max_eigenvalue())





def visualize_densities_with_matplotlib(vlasov_op, discr, filename, densities):
    left, right = discr.mesh.bounding_box()
    left = left[0]
    right = right[0]

    img_data = numpy.array(list(densities))
    from matplotlib.pyplot import imshow, savefig, \
            xlabel, ylabel, colorbar, clf, yticks

    clf()
    imshow(img_data, extent=(left, right, -1, 1))

    xlabel("$x$")
    ylabel("$p$")

    ytick_step = int(round(vlasov_op.p_discr.grid_size / 8))
    yticks(
            numpy.linspace(
                -1, 1, vlasov_op.p_discr.grid_size)[::ytick_step],
            ["%.3f" % vn for vn in 
                vlasov_op.p_discr.quad_points_1d[::ytick_step]])
    colorbar()

    savefig(filename)




def add_densities_to_silo(silo, vlasov_op, discr, densities):
    from pylo import SiloFile, DB_NODECENT

    scheme_dtype = vlasov_op.p_discr.diffmat.dtype
    is_complex = scheme_dtype.kind == "c"
    f_data = numpy.array(list(densities), dtype=scheme_dtype)

    silo.put_quadmesh("xpmesh", [
        discr.nodes.reshape((len(discr.nodes),)),
        vlasov_op.p_discr.quad_points_1d,
        ])

    if is_complex:
        silo.put_quadvar1("f_r", "xpmesh", f_data.real.copy(), f_data.shape, 
                DB_NODECENT)
        silo.put_quadvar1("f_i", "xpmesh", f_data.imag.copy(), f_data.shape, 
                DB_NODECENT)
    else:
        silo.put_quadvar1("f", "xpmesh", f_data, f_data.shape, 
                DB_NODECENT)
