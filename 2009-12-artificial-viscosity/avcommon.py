from math import sqrt
import numpy




def make_ui(cases):
    from smoother import TriBlobSmoother

    variables = {
        "vis_interval": 2,
        "vis_interval_steps": 0,

        "order": 4,
        "vis_order": None,
        "quad_min_degree": None,

        "stab_coefficient": 10,

        "n_elements": 20,

        "case": cases[0](),

        "smoother": TriBlobSmoother(use_max=False),
        "sensor": "decay_gating skyline",

        "viscosity_scale": 1,

        "extra_vis": False,
        }

    constants = {
            }
    for cl in cases + [
            TriBlobSmoother
            ]:
        constants[cl.__name__] = cl

    from pytools import CPyUserInterface
    return CPyUserInterface(variables, constants)




def sensor_from_string(sensor_str, discr, setup, vis_proj):
    mesh_a, mesh_b = discr.mesh.bounding_box()
    from pytools import product
    area = product(mesh_b[i] - mesh_a[i] for i in range(discr.mesh.dimensions))
    h = sqrt(area/len(discr.mesh.elements))

    sensor_words = sensor_str.split()
    if sensor_words[0] == "decay_gating":
        from hedge.bad_cell import (DecayGatingDiscontinuitySensorBase,
                SkylineModeProcessor, AveragingModeProcessor)

        correct_for_fit_error = False
        mode_processor = None
        ignored_modes = 1
        weight_exponent = 0

        sensor_words.pop(0)

        if "fit_correction" in sensor_words:
            sensor_words.remove("fit_correction")
            correct_for_fit_error = True

        for i, sw in enumerate(sensor_words):
            if sw.startswith("weight_exponent="):
                sensor_words.pop(i)
                weight_exponent = float(sw.split("=")[1])
                ignored_modes = 0
                break

        if "skyline" in sensor_words:
            sensor_words.remove("skyline")
            mode_processor = SkylineModeProcessor()
        elif "averaging" in sensor_words:
            sensor_words.remove("averaging")
            mode_processor = AveragingModeProcessor()

        if sensor_words:
            raise RuntimeError("didn't understand sensor spec: %s" % ",".join(sensor_words))

        sensor = DecayGatingDiscontinuitySensorBase(
                mode_processor=mode_processor,
                weight_exponent=weight_exponent,
                ignored_modes=ignored_modes,
                correct_for_fit_error=correct_for_fit_error,
                max_viscosity=setup.viscosity_scale*h/setup.order)

        decay_expt = sensor.bind_quantity(discr, "decay_expt")
        decay_lmc = sensor.bind_quantity(discr, "log_modal_coeffs")
        decay_wlmc = sensor.bind_quantity(discr, "weighted_log_modal_coeffs")
        decay_estimated_lmc = sensor.bind_quantity(discr, "estimated_log_modal_coeffs")

        def get_extra_vis_vectors(u):
            return [
                ("expt_u_dg", vis_proj(decay_expt(u))), 
                ("lmc_u_dg", vis_proj(decay_lmc(u))), 
                ("w_lmc_u_dg", vis_proj(decay_wlmc(u))), 
                ("est_lmc_u_dg", vis_proj(decay_estimated_lmc(u))), 
                ]

    elif sensor_words[0] == "persson_peraire":
        from hedge.bad_cell import PerssonPeraireDiscontinuitySensor
        sensor = PerssonPeraireDiscontinuitySensor(kappa=2,
            eps0=setup.viscosity_scale*h/setup.order, s_0=numpy.log10(1/setup.order**4))

        def get_extra_vis_vectors(u):
            return []

    else:
        raise RuntimeError("invalid sensor")

    return sensor, get_extra_vis_vectors

