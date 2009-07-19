from __future__ import division as _div
from pyrticle.units import SIUnitsWithUnityConstants as _SIU

# see http://webapp.dam.brown.edu/piki/WeibelInstability

units = _SIU()

epsilon = units.EPSILON0
mu = units.MU0
_c0 = units.VACUUM_LIGHT_SPEED()

# _u is the thermal (ie. reference) velocity
_u = numpy.array([0.25*_c0, 0.05*_c0])

_n0 = 1.2e5

def get_densities(discr, vlasov_op):
    from hedge.discretization import ones_on_volume
    base_vec = ones_on_volume(discr)
    from hedge.tools import make_obj_array, join_fields

    densities = make_obj_array([
        base_vec*_n0/numpy.product(_u)*exp(-numpy.sum(v**2/u**2))
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
