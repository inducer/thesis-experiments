from __future__ import division as _div
from pyrticle.units import SIUnitsWithUnityConstants as _SIU

units = _SIU()

def get_densities(discr, vlasov_op):
    from math import cos, exp

    #base_vec = discr.interpolate_volume_function(
            #lambda x, el: cos(0.5*x[0]))
    base_vec = discr.interpolate_volume_function(
            lambda x, el: 1+cos(2*x[0]))
    #base_vec = discr.interpolate_volume_function(lambda x, el: 1)
    from hedge.tools import make_obj_array, join_fields

    v_stretch = numpy.array([1,2])[:v_dim]
    densities = make_obj_array([
        base_vec*exp(
            -(16*units.VACUUM_LIGHT_SPEED() 
                * numpy.dot(v, v_stretch*v)))#*v[0]
        for v in vlasov_op.v_points])

    return densities

species_mass = units.EL_MASS
species_charge = -units.EL_CHARGE

final_time = 100
multirate_dt_scale = 0.1
vis_interval = 5

v_dim = 2
p_grid_size = 16
x_element_count = 10
p_discr_args = dict(
        filter_type="exponential",
        hard_scale=0.6, 
        bounded_fraction=0.8,
        filter_parameters=dict(preservation_ratio=0.3))
