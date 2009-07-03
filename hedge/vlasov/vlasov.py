from __future__ import division
import numpy
import numpy.linalg as la




class VlasovOperatorBase:
    def __init__(self, units, species_mass, *args, **kwargs):
        from p_discr import MomentumDiscretization
        self.p_discr = MomentumDiscretization(*args, **kwargs)
        self.units = units
        self.species_mass = species_mass

        self.velocity_points = [units.v_from_p(species_mass, p)
                for p in self.p_discr.quad_points]

        from hedge.models.advection import StrongAdvectionOperator
        self.x_adv_operators = [
                StrongAdvectionOperator(v, flux_type="upwind")
                for v in self.velocity_points]

    def op_template(self, forces, densities_base=0):
        from hedge.optemplate import make_vector_field
        from hedge.tools import make_obj_array

        f = make_vector_field("f",
                range(densities_base, 
                    densities_base
                    +len(self.p_discr.quad_points)))

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
                        for forces_i, f_p_i in zip(forces, f_p))

    def max_eigenvalue(self):
        return max(la.norm(v) for v in self.velocity_points)




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
        forces = [make_obj_array([force_base[i][p_axis]
            for i, p in enumerate(self.p_discr.quad_points)])
            for p_axis in [0]]

        return VlasovOperatorBase.op_template(self, forces)

    def bind(self, discr):
        compiled_op_template = discr.compile(self.op_template())

        def rhs(t, densities):
            return compiled_op_template(
                    f=densities,
                    force=self.forces_func(t))

        return rhs



class VlasovMaxwellOperator(VlasovOperatorBase):
    def __init__(self, maxwell_op, *args, **kwarg):
        VlasovOperator.__init__(self, *args, **kwargs)

        self.maxwell_op = maxwell_op

        from hedge.tools import count_subset
        self.maxwell_field_count = \
                count_subset(self.maxwell_op.get_eh_subset())

    def op_template(self):
        max_op_template = self.maxwell_op.op_template()

    def bind(discr):
        bound_max_op = self.maxwell_op.bind(discr)
        compiled_vlasov_op_template = \
                discr.compile(self.op_template())

        def rhs(t, eh_densities):
            max_w = eh_densities[:self.maxwell_field_count]
            densities = eh_densities[self.maxwell_field_count]

            from hedge.tools import join_fields
            return join_fields(
                    bound_max_op(max_w),
                    compiled_vlasov_op_template(f=densities))





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




