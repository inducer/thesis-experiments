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

    def to_nd_array(self, linear_array, obj_array=True):
        if obj_array:
            result = numpy.zeros(self.shape, dtype=object)
        else:
            result = numpy.zeros(self.shape+linear_array[0].shape, 
                    dtype=linear_array[0].dtype)
        for idx_tuple, entry in zip(self.iterindex(), linear_array):
            result[idx_tuple] = entry

        return result

    def to_linear_array(self, nd_array):
        result = numpy.zeros(len(self), dtype=object)
        for lin_index, idx_tuple in enumerate(self.iterindex()):
            result[lin_index] = nd_array[idx_tuple]

        return result




class AdvectionOperator:
    def __init__(self, dimensions):
        self.dimensions = dimensions

    def flux(self):
        from hedge.flux import \
                make_normal, FluxScalarPlaceholder, IfPositive, \
                FluxScalarParameter

        from hedge.tools import make_obj_array
        v = make_obj_array([
            FluxScalarParameter("v%d" % i) 
            for i in range(self.dimensions)])

        u = FluxScalarPlaceholder(0)
        normal = make_normal(self.dimensions)

        weak_flux = (numpy.dot(normal, v)*
                IfPositive(numpy.dot(normal, v),
                    u.int, # outflow
                    u.ext, # inflow
                    ))

        return u.int * numpy.dot(normal, v) - weak_flux

    def op_template(self):
        from hedge.optemplate import Field, pair_with_boundary, \
                get_flux_operator, make_nabla, InverseMassOperator, \
                ScalarParameter

        from hedge.tools import make_obj_array
        v = make_obj_array([
            ScalarParameter("v%d" % i) 
            for i in range(self.dimensions)])

        u = Field("u")
        flux_op = get_flux_operator(self.flux())

        return (-numpy.dot(v, make_nabla(self.dimensions)*u)
                + InverseMassOperator()*(flux_op * u))

    def bind(self, discr):
        compiled_op_template = discr.compile(self.op_template())

        def rhs(v, t, u):
            velocity_constants = dict(
                    ("v%d" % i, v[i]) 
                    for i in range(self.dimensions))

            return compiled_op_template(u=u, **velocity_constants)

        return rhs




class VlasovOperatorBase:
    def __init__(self, x_discr, p_discrs, units, species_mass,
            use_fft=False):
        from p_discr import MomentumDiscretization
        self.x_discr = x_discr
        self.p_discrs = p_discrs
        self.units = units
        self.species_mass = species_mass
        self.use_fft = use_fft

        self.p_grid = TensorProductGrid(
                [p_discr.quad_points_1d for p_discr in self.p_discrs])

        self.advection_op = AdvectionOperator(x_discr.dimensions)
        self.bound_advection_op= self.advection_op.bind(x_discr)

    @property
    def v_points(self):
        for p in self.p_grid:
            yield self.units.v_from_p(self.species_mass, p)

    @property
    def v_dim(self):
        return len(self.p_discrs)

    @memoize_method
    def quad_weights(self):
        from pytools import product
        pg = self.p_grid

        return numpy.array([
            product(p_discr.quad_weights_1d[j] 
                for j, p_discr in zip(idx_tuple, self.p_discrs)) 
            for idx_tuple in pg.iterindex()], dtype=numpy.float64)

    @memoize_method
    def make_densities_placeholder(self):
        from hedge.optemplate import make_vector_field
        return make_vector_field("f",
                range(len(self.p_grid)))

    def apply_p1d_function_to_axis(self, axis, func, f_ary):
        result = numpy.zeros(f_ary.shape, dtype=object)
        remaining_shape = (
                self.p_grid.shape[:axis] 
                + self.p_grid.shape[axis+1:])

        from pytools import indices_in_shape
        from hedge.tools import numpy_linear_comb, make_obj_array

        for idx_tuple in indices_in_shape(remaining_shape):
            this_slice = (
                    idx_tuple[:axis] 
                    + (slice(None),)
                    + idx_tuple[axis:])

            result[this_slice] = make_obj_array(func(f_ary[this_slice]))

        return result

    def apply_p1d_matrix_to_axis(self, axis, matrix, f_ary):
        result = numpy.zeros(f_ary.shape, dtype=object)
        remaining_shape = (
                self.p_grid.shape[:axis] 
                + self.p_grid.shape[axis+1:])

        #if self.use_fft and hasattr(p_discr, "diff_function"):
            #result[this_slice] = p_discr.diff_function(
                    #f_ary[this_slice])
        #else:

        from pytools import indices_in_shape
        from hedge.tools import numpy_linear_comb

        for idx_tuple in indices_in_shape(remaining_shape):
            this_slice = (
                    idx_tuple[:axis] 
                    + (slice(None),)
                    + idx_tuple[axis:])

            for row_idx in range(self.p_discrs[axis].grid_size):
                lc = numpy_linear_comb(
                        zip(matrix[row_idx], f_ary[this_slice]))
                dest_idx = (
                        idx_tuple[:axis] 
                        + (row_idx,)
                        + idx_tuple[axis:])
                result[dest_idx] = lc

        return result

    def __call__(self, t, fields, forces_T=None):
        f_ary = self.p_grid.to_nd_array(fields)

        def nd_diff_function(axis):
            return self.p_grid.to_linear_array(
                    self.apply_p1d_matrix_to_axis(
                        axis, self.p_discrs[axis].diffmat, f_ary))

        # list of densities differentiated along each p axis
        f_p = [nd_diff_function(diff_axis)
            for diff_axis in range(len(self.p_discrs))]

        if forces_T is None:
            forces_T = self.forces_T(t)

        from hedge.tools import make_obj_array
        return (
                make_obj_array([
                    self.bound_advection_op(v, t, f_of_p)
                    for v, f_of_p in zip(self.v_points, fields)
                    ]) 
                - sum(
                    forces_i*f_p_i
                    for forces_i, f_p_i in zip(forces_T, f_p))
                )

    def apply_filter_matrix(self, densities):
        f_ary = self.p_grid.to_nd_array(densities)

        for axis, p_discr in enumerate(self.p_discrs):
            f_ary = self.apply_p1d_matrix_to_axis(
                    axis, p_discr.filter_matrix, f_ary)

        return self.p_grid.to_linear_array(f_ary)

    def apply_1d_function(self, p_discr_to_func_map, densities):
        f_ary = self.p_grid.to_nd_array(densities)

        for axis, p_discr in enumerate(self.p_discrs):
            f_ary = self.apply_p1d_function_to_axis(
                    axis, p_discr_to_func_map(p_discr), f_ary)

        return self.p_grid.to_linear_array(f_ary)

    def apply_filter(self, densities):
        return self.apply_1d_function(
                lambda p_discr: p_discr.apply_filter,
                densities)

    def max_eigenvalue(self):
        return max(la.norm(v) for v in self.v_points)

    def integral_dp(self, values):
        return sum(qw_i*val_i
                for qw_i, val_i in 
                zip(self.quad_weights(), values))

    def integral_p_squared_dp(self,values):
        return sum(qw_i*val_i*numpy.dot(p_i, p_i)
                for qw_i, val_i, p_i in 
                zip(self.quad_weights(), values, self.p_grid))

    @memoize_method
    def int_dv_quad_weights(self):
        weights = numpy.zeros(len(self.p_grid), dtype=numpy.float64)
        c = self.units.VACUUM_LIGHT_SPEED()

        for i, p in enumerate(self.p_grid):
            p_square = numpy.dot(p, p)
            denom = p_square + c**2 * self.species_mass**2
            weights[i] =(
                    c/denom**0.5
                    - c*p_square / denom**1.5)

        return weights*self.quad_weights()

    def integral_dv(self, values):
        return sum(qw_i*val_i
                for qw_i, val_i in 
                zip(self.int_dv_quad_weights(), values))




class PhaseSpaceTransportOperator(VlasovOperatorBase):
    def __init__(self, x_discr, p_discrs, units, species_mass, forces_T_func, 
            use_fft=False):
        VlasovOperatorBase.__init__(self, x_discr, p_discrs, units, species_mass, 
                use_fft)
        self.forces_T = forces_T_func





class VlasovMaxwellOperator(VlasovOperatorBase):
    def __init__(self, x_discr, p_discrs, maxwell_op, units, species_mass, species_charge, 
            use_fft=False):
        VlasovOperatorBase.__init__(self, x_discr, p_discrs, units, species_mass, 
                use_fft)

        self.maxwell_op = maxwell_op
        self.bound_maxwell_op = maxwell_op.bind(x_discr)
        self.species_charge = species_charge
        self.charge_mass_ratio = self.species_charge/self.species_mass

        from hedge.tools import count_subset
        self.maxwell_field_count = \
                count_subset(self.maxwell_op.get_eh_subset())

        self.v_subset = [True] * self.v_dim + [False] * (3-self.v_dim)
        self.e_subset = maxwell_op.get_eh_subset()[:3]
        self.b_subset = maxwell_op.get_eh_subset()[3:]
        from hedge.tools import SubsettableCrossProduct
        self.v_b_cross = SubsettableCrossProduct(
                self.v_subset, self.b_subset, self.v_subset)


    def make_maxwell_eh_placeholders(self):
        max_op = self.maxwell_op

        from hedge.optemplate import make_vector_field
        max_fields = make_vector_field("w", self.maxwell_field_count)
        return max_op.split_eh(max_fields)

    def j(self, densities):
        def times(vec, scalar):
            result = numpy.zeros(vec.shape, dtype=object)
            for i in range(len(vec)):
                result[i] = scalar*vec[i]
            return result

        return self.integral_dv([
                times(self.species_charge*v, f_i)
                for v, f_i in zip(self.v_points, densities)
                ])

    def forces_T(self, densities, max_e, max_h):
        max_op = self.maxwell_op

        max_b = max_op.mu * max_h

        def pick_subset(tgt_subset, vec_subset, vec):
            from hedge.tools import count_subset
            result = numpy.zeros(count_subset(tgt_subset), 
                    dtype=object)

            tgt_idx = 0
            for ts, vs, v_i in zip(tgt_subset, vec_subset, vec):
                if ts:
                    if vs:
                        result[tgt_idx] = v_i
                    else:
                        result[tgt_idx] = 0
                    tgt_idx += 1

            return result

        el_force = pick_subset(self.v_subset, self.e_subset,
                self.species_charge * max_e)

        from hedge.tools import make_obj_array
        q_times_b = self.species_charge*max_b
        forces = make_obj_array([
            el_force + self.v_b_cross(v, q_times_b)
            for v in self.v_points])

        return [
                make_obj_array([
                    forces[v_node_idx][v_axis]
                    for v_node_idx in xrange(len(self.p_grid))
                    ])
                for v_axis in xrange(self.v_dim)]

    def __call__(self, t, q):
        max_w = q[:self.maxwell_field_count]
        max_e, max_h = self.maxwell_op.split_eh(max_w)
        densities = q[self.maxwell_field_count:]

        from hedge.tools import join_fields
        max_e_rhs, max_h_rhs = self.maxwell_op.split_eh(
                self.bound_maxwell_op(t, max_w))

        return join_fields(
                max_e_rhs + self.j(densities)/self.maxwell_op.epsilon,
                max_h_rhs,
                VlasovOperatorBase.__call__(self, t,
                    densities, forces_T=self.forces_T(densities, max_e, max_h)))

    def apply_filter(self, q):
        max_w = q[:self.maxwell_field_count]
        densities = q[self.maxwell_field_count:]

        from hedge.tools import join_fields
        return join_fields(
                max_w,
                VlasovOperatorBase.apply_filter(self, densities))

    def split_e_h_densities(self, fields):
        max_w = fields[:self.maxwell_field_count]
        e, h = self.maxwell_op.split_eh(max_w)
        densities = fields[self.maxwell_field_count:]
        return e, h, densities

    def max_eigenvalue(self):
        return max(
                max(la.norm(v) for v in self.v_points),
                self.maxwell_op.max_eigenvalue())





def add_xp_to_silo(silo, vlasov_op, discr, 
        names_and_quantities):
    from pylo import SiloFile, DB_NODECENT

    from pytools import common_dtype
    scheme_dtype = common_dtype(
            p_discr.diffmat.dtype
            for p_discr in vlasov_op.p_discrs)
    is_complex = scheme_dtype.kind == "c"

    silo.put_quadmesh("xpmesh", [
        discr.nodes.reshape((len(discr.nodes),)),
        ] + vlasov_op.p_grid.dim_points)

    for name, quant in names_and_quantities:
        q_data = numpy.array(list(quant), dtype=scheme_dtype)

        dim_lengths = (
                tuple(len(dp) for dp in vlasov_op.p_grid.dim_points)
                + (len(discr.nodes),))

        v_dim = vlasov_op.v_dim

        q_data = numpy.reshape(q_data, dim_lengths)
        q_data = q_data.transpose(
                 tuple(range(v_dim-1, -1, -1))
                 + (v_dim,)
                 )

        if is_complex:
            silo.put_quadvar1(name+"_r", "xpmesh", q_data.real.copy(), q_data.shape, 
                    DB_NODECENT)
            silo.put_quadvar1(name+"_i", "xpmesh", q_data.imag.copy(), q_data.shape, 
                    DB_NODECENT)
        else:
            silo.put_quadvar1(name, "xpmesh", q_data, q_data.shape, 
                    DB_NODECENT)




def find_multirate_split(v_points, levels=2):
    def generate_splits_and_work_amount(v_points_with_indices):
        max_v_fast = la.norm(v_points_with_indices[-1][1])
        for first_fast_index in range(1, len(v_points_with_indices)):
            max_v_slow = la.norm(v_points_with_indices[first_fast_index-1][1])

            from math import ceil
            substep_count = int(ceil(max_v_fast / max_v_slow))

            fast_v_points_with_indices = v_points_with_indices[first_fast_index:]
            slow_v_points_with_indices = v_points_with_indices[:first_fast_index]

            yield ((substep_count, [fast_v_points_with_indices, slow_v_points_with_indices]),
                    first_fast_index/substep_count + len(fast_v_points_with_indices),
                    )

    v_points_with_indices = list(enumerate(v_points))
    v_points_with_indices.sort(
            key=lambda (i, v): la.norm(v))

    if levels < 2:
        raise ValueError("invalid level count")

    from pytools import argmin2
    substep_counts = []
    rate_groups = []
    for level in range(levels-1):
        substep_count, (fast_group, v_points_with_indices) = \
                argmin2(generate_splits_and_work_amount(v_points_with_indices))
        substep_counts.append(substep_count)
        rate_groups.append(fast_group)

    rate_groups.append(v_points_with_indices)

    # peel away the velocities before returning rate groups
    return substep_counts, [numpy.array([idx for idx, v in rg], dtype=numpy.intp)
            for rg in rate_groups]




# tests -----------------------------------------------------------------------
def test_tp_grid():
    tpg = TensorProductGrid([[5,6,7], [3,4,5]])

    for i, p in enumerate(tpg):
        tp = tpg.tuple_from_linear(i)
        p2 = tpg.point_from_tuple(tp)

        assert (p2 == p).all()
        assert tpg.linear_from_tuple(tp) == i




def test_mrab_split():
    from p_discr import MomentumDiscretization
    pd = MomentumDiscretization(
            grid_size=16, filter_type="exponential",
            hard_scale=5, bounded_fraction=0.8)
    print best_multirate_split(pd.quad_points_1d)


if __name__ == "__main__":
    test_tp_grid()
    test_mrab_split()
