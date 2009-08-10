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

    v_stretch = numpy.array([1,2])[:_v_dim]
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

x_element_count = 60
x_dg_order = 4
_p_discr_args = dict(
        filter_type="exponential",
        hard_scale=1, 
        bounded_fraction=0.8,
        filter_parameters=dict(
            preservation_ratio=0.7,
            truncation_value=1e-8,
            ),
        use_fft=False, # for filter--not working yet
        )

p_discrs = [
        PDiscr(45, **_p_discr_args),
        PDiscr(8, **_p_discr_args),
        ]

_v_dim = len(p_discrs)

filter_interval = 1
