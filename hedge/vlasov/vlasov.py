from __future__ import division
import numpy
import numpy.linalg as la
from pytools import memoize_method




class TensorProductGrid:
    def __init__(self, dim_points):
        self.dim_points = dim_points
        self.shape = tuple(len(dp) for dp in dim_points)

    def __len__(self):
        from pytools import product
        return product(self.shape)

    def __iter__(self):
        from pytools import indices_in_shape
        for i in indices_in_shape(self.shape):
            yield self.point_from_tuple(i)

    def iterindex(self):
        from pytools import indices_in_shape
        return indices_in_shape(self.shape)

    def point_from_tuple(self, tp):
        return numpy.array([
            dp[i] for dp, i in zip(self.dim_points, tp)])

    pft = point_from_tuple

    def linear_from_tuple(self, tp):
        result = 0
        for shape_i, i in zip(self.shape, tp):
            result = result*shape_i + i
        return result

    def tuple_from_linear(self, idx):
        result = []
        for shape_i in self.shape[::-1]:
            result.insert(0, idx % shape_i)
            idx //= shape_i
        return tuple(result)




class VlasovOperatorBase:
    def __init__(self, x_dim, v_dim, units, species_mass, *args, **kwargs):
        from p_discr import MomentumDiscretization
        self.p_discr = MomentumDiscretization(*args, **kwargs)
        self.units = units
        self.species_mass = species_mass

        self.x_dim = x_dim
        self.v_dim = v_dim

        self.p_grid = TensorProductGrid(
                v_dim*[self.p_discr.quad_points_1d])

        from hedge.models.advection import StrongAdvectionOperator
        self.x_adv_operators = [
                StrongAdvectionOperator(v[:self.x_dim], flux_type="upwind")
                for v in self.v_points]

    @property
    def v_points(self):
        for p in self.p_grid:
            yield self.units.v_from_p(self.species_mass, p)

    @memoize_method
    def quad_weights(self):
        from pytools import product
        pg = self.p_grid
        qw = self.p_discr.quad_weights_1d

        return numpy.array(
                [product(qw[j] for j in i) for i in pg.iterindex()], 
                dtype=numpy.float64)

    @memoize_method
    def make_densities_placeholder(self, base_number=0):
        from hedge.optemplate import make_vector_field
        return make_vector_field("f",
                range(base_number, 
                    base_number+len(self.p_grid)))

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

        #if hasattr(p_discr, "diff_function"):
            #f_p = [p_discr.diff_function(f)]
        #else:

        p_grid = self.p_grid

        def replace_tuple_entry(tp, i, new_value):
            return tp[:i] + (new_value,) + tp[(i+1):]
        f_p = [make_obj_array([
            sum(
                p_discr.diffmat[
                    p_grid.tuple_from_linear(i)[diff_dir],
                    j]
                *f[p_grid.linear_from_tuple(
                    replace_tuple_entry(
                        p_grid.tuple_from_linear(i),
                        diff_dir, j))]
                for j in range(p_discr.grid_size))
            for i in range(len(p_grid))
            ])
            for diff_dir in range(self.v_dim)]

        return make_obj_array([
                adv_op_template(adv_op, f[i])
                for i, adv_op in enumerate(
                    self.x_adv_operators)
                ]) + sum(
                        forces_i*f_p_i
                        for forces_i, f_p_i in zip(forces_T, f_p))

    def max_eigenvalue(self):
        return max(la.norm(v) for v in self.v_points)


    def integral_dp(self, p_values):
        return numpy.dot(self.quad_weights(), p_values)

    @memoize_method
    def int_dv_quad_weights(self):
        weights = numpy.zeros(len(self.p_grid), dtype=numpy.float64)
        c = self.units.VACUUM_LIGHT_SPEED

        for i, p in enumerate(self.p_grid):
            p_square = numpy.dot(p, p)
            denom = p_square + c**2 * self.species_mass**2
            weights[i] =(
                    c/denom**0.5
                    - c*p_square / denom**1.5)

        return weights*self.quad_weights()

    def integral_dv(self, values):
        return numpy.dot(self.int_dv_quad_weights(), values)




class VlasovOperator(VlasovOperatorBase):
    def __init__(self, x_dim, v_dim, units, species_mass, forces_func, 
            *args, **kwargs):
        VlasovOperatorBase.__init__(self, x_dim, v_dim, units, species_mass, 
                *args, **kwargs)
        self.forces_func = forces_func

    def op_template(self):
        from hedge.optemplate import Field
        force_base = Field("force")

        from hedge.tools import make_obj_array
        forces_T = [make_obj_array([force_base[i][p_axis]
            for i, p in enumerate(self.p_grid)])
            for p_axis in range(self.v_dim)]

        return VlasovOperatorBase.op_template(self, forces_T)

    def bind(self, discr):
        compiled_op_template = discr.compile(self.op_template())

        def rhs(t, densities):
            return compiled_op_template(
                    f=densities,
                    force=self.forces_func(t))

        return rhs




class VlasovMaxwellOperator(VlasovOperatorBase):
    def __init__(self, x_dim, v_dim, maxwell_op, units, species_mass, species_charge, 
            *args, **kwargs):
        VlasovOperatorBase.__init__(self, x_dim, v_dim, units, species_mass, *args, **kwargs)

        self.maxwell_op = maxwell_op
        self.species_charge = species_charge

        from hedge.tools import count_subset
        self.maxwell_field_count = \
                count_subset(self.maxwell_op.get_eh_subset())

    def make_maxwell_eh_placeholders(self):
        max_op = self.maxwell_op

        from hedge.optemplate import make_vector_field
        max_fields = make_vector_field("w", self.maxwell_field_count)
        return max_op.split_eh(max_fields)

    def j(self, densities):
        return self.integral_dv([
                (self.species_charge*v)*f_i
                for v, f_i in zip(self.v_points, densities)
                ])

    def forces_T(self, densities, max_e, max_h):
        max_op = self.maxwell_op
        max_e, max_h = self.make_maxwell_eh_placeholders()

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
            for v in self.v_points])

        return [
                make_obj_array([
                    forces[v_node_idx][v_axis]
                    for v_node_idx in xrange(len(self.p_grid))
                    ])
                for v_axis in xrange(self.v_dim)]

    def op_template(self):
        from hedge.tools import join_fields, make_obj_array
        max_op = self.maxwell_op
        max_e, max_h = self.make_maxwell_eh_placeholders()

        densities = self.make_densities_placeholder()

        j = self.j(densities)

        # assemble rhs --------------------------------------------------------
        max_e_rhs, max_h_rhs = max_op.split_eh(
                max_op.op_template(join_fields(max_e, max_h)))
        vlasov_rhs = VlasovOperatorBase.op_template(
                self, self.forces_T(densities, max_e, max_h), densities)

        return join_fields(
                max_e_rhs + j/max_op.epsilon,
                max_h_rhs,
                vlasov_rhs)

    def bind(self, discr, op_template=None):
        if op_template is None:
            op_template = self.op_template()
        compiled = discr.compile(op_template)

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
                max(la.norm(v) for v in self.v_points),
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




def add_xv_to_silo(silo, vlasov_op, discr, 
        names_and_quantities):
    from pylo import SiloFile, DB_NODECENT

    scheme_dtype = vlasov_op.p_discr.diffmat.dtype
    is_complex = scheme_dtype.kind == "c"

    silo.put_quadmesh("xpmesh", [
        discr.nodes.reshape((len(discr.nodes),)),
        ] + vlasov_op.p_grid.dim_points)

    for name, quant in names_and_quantities:
        q_data = numpy.array(list(quant), dtype=scheme_dtype)

        if is_complex:
            silo.put_quadvar1(name+"_r", "xpmesh", q_data.real.copy(), q_data.shape, 
                    DB_NODECENT)
            silo.put_quadvar1(name+"_i", "xpmesh", q_data.imag.copy(), q_data.shape, 
                    DB_NODECENT)
        else:
            silo.put_quadvar1(name, "xpmesh", q_data, q_data.shape, 
                    DB_NODECENT)



def test_tp_grid():
    tpg = TensorProductGrid([[5,6,7], [3,4,5]])

    for i, p in enumerate(tpg):
        tp = tpg.tuple_from_linear(i)
        print i, tp, p
        p2 = tpg.point_from_tuple(tp)

        assert (p2 == p).all()
        assert tpg.linear_from_tuple(tp) == i





if __name__ == "__main__":
    test_tp_grid()
