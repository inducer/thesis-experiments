import pytools
import numpy
import numpy.linalg as la




def default_get_eh(discr, vlasov_op, densities):
    return None, None




class VlasovMaxwellCPyUserInterface(pytools.CPyUserInterface):
    def __init__(self):
        constants = {
                "numpy": numpy,
                "la": la,
                }

        variables = {
                "final_time": None,
                "dt_scale": 1,

                "units": None,

                "epsilon": None,
                "mu": None,

                "x_element_count": None,
                "x_mesh": None,
                "x_dg_order": 3,
                "discr_debug_flags": [],

                "v_dim": None,
                "p_grid_size": None,
                "p_discr_args": dict(
                    filter_type="exponential",
                    hard_scale=0.6, 
                    bounded_fraction=0.8,
                    filter_parameters=dict(preservation_ratio=0.3)),

                "species_mass": None,
                "species_charge": None,

                "get_densities": None,
                "get_eh": default_get_eh,

                "vis_interval": 40,
                }

        doc = {}

        pytools.CPyUserInterface.__init__(self, variables, constants, doc)

    def validate(self, setup):
        if setup.x_mesh is None:
            from hedge.mesh import make_uniform_1d_mesh
            from math import pi
            setup.x_mesh = make_uniform_1d_mesh(-pi, pi, 
                    setup.x_element_count, periodic=True)

        must_specify = ["species_mass", "species_charge",
                "v_dim", "p_grid_size", "final_time"]

        for name in must_specify:
            if getattr(setup, name) is None:
                raise ValueError("must specify %s" % name)

        if setup.epsilon is None:
            setup.epsilon = setup.units.EPSILON0
        if setup.mu is None:
            setup.mu = setup.units.MU0
