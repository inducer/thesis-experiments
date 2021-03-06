from __future__ import division as _div
from pyrticle.units import SIUnitsWithNaturalConstants as _SIU

# see http://webapp.dam.brown.edu/piki/WeibelInstability
units = _SIU()

epsilon = units.EPSILON0
mu = units.MU0
_c0 = units.VACUUM_LIGHT_SPEED()

# _u is the thermal (ie. reference) velocity
#_u = numpy.array([0.25*_c0, 0.05*_c0])
_u = numpy.array([0.25*_c0, 0.15*_c0])

_n0 = 1.2e5

def get_densities(discr, vlasov_op):
    from hedge.discretization import ones_on_volume
    base_vec = ones_on_volume(discr)
    from hedge.tools import make_obj_array, join_fields

    densities = make_obj_array([
        base_vec*_n0/numpy.product(_u)*exp(-numpy.sum(v**2/_u**2))
        for v in vlasov_op.v_points])

    return densities

species_mass = units.EL_MASS
species_charge = -units.EL_CHARGE

_omega_pe = sqrt(
        _n0*abs(species_charge)
        /
        (epsilon * species_mass))

_extent = 15*_c0/_omega_pe
print _extent

from hedge.mesh import make_uniform_1d_mesh as _make_u1d_mesh
x_mesh = _make_u1d_mesh(0, _extent, 10, periodic=True)

final_time = 100
#dt_scale = 0.1
multirate_dt_scale = 0.1
vis_interval = 1

_v_hard_scale = _c0*0.8
_p_hard_scale = (
        species_mass*units.gamma_from_v(_v_hard_scale)*_v_hard_scale)

_p_discr_args = dict(
        filter_type="exponential",
        hard_scale=_p_hard_scale, 
        bounded_fraction=0.8,
        filter_parameters=dict(preservation_ratio=0.3))

p_discrs = [
        PDiscr(16, **_p_discr_args),
        PDiscr(16, **_p_discr_args),
        ]

x_element_count = 10
#use_fft = True
