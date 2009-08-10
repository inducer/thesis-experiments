from __future__ import division
from pytools.log import LogQuantity
from hedge.log import EMFieldGetter




class VlasovMaxwellFGetter(object):
    """Makes E and H field accessible as self.e and self.h from a variable lookup.
    To be used with the EM log quantities in this module."""
    def __init__(self, discr, maxwell_op, vlasov_op, fgetter):
        self.discr = discr
        self.maxwell_op = maxwell_op
        self.vlasov_op = vlasov_op
        self.fgetter = fgetter

    @property
    def e(self):
        e, h, densities = self.vlasov_op.split_e_h_densities(
                self.fgetter())
        return e

    @property
    def h(self):
        e, h, densities = self.vlasov_op.split_e_h_densities(
                self.fgetter())
        return h

    @property
    def densities(self):
        e, h, densities = self.vlasov_op.split_e_h_densities(
                self.fgetter())
        return densities




class TotalCharge(LogQuantity):
    def __init__(self, fields, name="Q_total"):
        LogQuantity.__init__(self, name, "C", "Total coulombic charge")
        self.fields = fields

    @property
    def default_aggregator(self): 
        from pytools import norm_2
        return norm_2

    def __call__(self):
        vlasov_op = self.fields.vlasov_op

        q_times_f = vlasov_op.species_charge*self.fields.densities
        charge_density = vlasov_op.integral_dp(q_times_f)
        return self.fields.discr.integral(charge_density)




class DensityKineticEnergy(LogQuantity):
    def __init__(self, fields, name="W_kin"):
        LogQuantity.__init__(self, name, "J", "Kinetic energy of the particle density")
        self.fields = fields

    @property
    def default_aggregator(self): 
        from pytools import norm_2
        return norm_2

    def __call__(self):
        vlasov_op = self.fields.vlasov_op
        max_op = self.fields.maxwell_op

        f_over_2m = self.fields.densities/(2*vlasov_op.species_mass)
        rel_energy_vs_x = vlasov_op.integral_p_squared_dp(f_over_2m)

        from hedge.tools import ptwise_dot
        energy_density = 1/2*(rel_energy_vs_x*rel_energy_vs_x)
        return self.fields.discr.integral(energy_density)





def add_density_quantities(mgr, vlasov_op, fields):
    mgr.add_quantity(DensityKineticEnergy(fields))
    mgr.add_quantity(TotalCharge(fields))
