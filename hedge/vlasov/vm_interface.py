import pytools
import numpy
import numpy.linalg as la




def default_get_eh(discr, vlasov_op, densities):
    return None, None




class VlasovMaxwellCPyUserInterface(pytools.CPyUserInterface):
    def __init__(self):
        from math import sqrt, exp, pi, sin, cos
        from p_discr import MomentumDiscretization
        constants = {
                "numpy": numpy,
                "la": la,
                "sqrt": sqrt,
                "exp": exp,
                "pi": pi,
                "sin": sin,
                "cos": cos,
                "PDiscr": MomentumDiscretization,
                }

        variables = {
                "final_time": None,
                "dt_scale": 1,
                "multirate_dt_scale": 1,

                "units": None,

                "epsilon": None,
                "mu": None,

                "x_element_count": None,
                "x_mesh": None,
                "x_dg_order": 3,
                "discr_debug_flags": [],
                "use_fft": False,

                "p_discrs": None,

                "species_mass": None,
                "species_charge": None,

                "get_densities": None,
                "get_eh": default_get_eh,

                "vis_interval": 40,

                "filter_interval": 0,
                }

        doc = {}

        pytools.CPyUserInterface.__init__(self, variables, constants, doc)

    def validate(self, setup):
        if setup.x_mesh is None:
            from hedge.mesh import make_uniform_1d_mesh
            from math import pi
            setup.x_mesh = make_uniform_1d_mesh(-pi, pi, 
                    setup.x_element_count, periodic=True)

        must_specify = ["species_mass", "species_charge", "p_discrs"]

        for name in must_specify:
            if getattr(setup, name) is None:
                raise ValueError("must specify %s" % name)

        if setup.epsilon is None:
            setup.epsilon = setup.units.EPSILON0
        if setup.mu is None:
            setup.mu = setup.units.MU0
