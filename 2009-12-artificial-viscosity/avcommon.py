from __future__ import division
from math import sqrt
import numpy




def make_ui(cases):
    from smoother import TriBlobSmoother, VertexwiseMaxSmoother

    variables = {
        "use_2d": False,

        "vis_interval": None,
        "vis_interval_steps": 0,

        "order": 4,
        "vis_order": None,
        "quad_min_degree": None,

        "stab_coefficient": 1,

        "n_elements": 20,

        "case": cases[0](),

        "smoother": VertexwiseMaxSmoother(),
        "sensor": "decay_gating skyline",

        "viscosity_scale": 1,

        "vis_exact": True,
        "extra_vis": False,
        "log_l1_error": True,
        }

    constants = {
            }
    for cl in cases + [
            TriBlobSmoother,
            VertexwiseMaxSmoother
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
        weight_mode = None

        sensor_words.pop(0)

        if "fit_correction" in sensor_words:
            sensor_words.remove("fit_correction")
            correct_for_fit_error = True

        for i, sw in enumerate(sensor_words):
            if sw.startswith("weight_exponent="):
                sensor_words.pop(i)
                weight_exponent = float(sw.split("=")[1])
                weight_mode = ("exponential", weight_exponent)
                ignored_modes = 0
                break

        if "nd_weight" in sensor_words:
            sensor_words.remove("nd_weight")
            weight_mode = "nd_weight"

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
                weight_mode=weight_mode,
                ignored_modes=ignored_modes,
                correct_for_fit_error=correct_for_fit_error)

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




def make_discr(setup):
    from hedge.backends import guess_run_context
    rcon = guess_run_context()

    if rcon.is_head_rank:
        if hasattr(setup.case, "make_mesh"):
            mesh = setup.case.make_mesh()
        elif setup.use_2d:
            extent_x = setup.case.b-setup.case.a
            extent_y = extent_x*0.5
            dx = extent_x/setup.n_elements
            subdiv = (setup.n_elements, int(1+extent_y//dx))
            from pytools import product

            from hedge.mesh.generator import make_rect_mesh
            mesh = make_rect_mesh((setup.case.a, 0), (setup.case.b, extent_y), 
                    periodicity=(setup.case.is_periodic, True), 
                    subdivisions=subdiv,
                    max_area=extent_x*extent_y/(2*product(subdiv))
                    )
        else:
            from hedge.mesh.generator import make_uniform_1d_mesh
            mesh = make_uniform_1d_mesh(
                    setup.case.a, setup.case.b, 
                    setup.n_elements, 
                    periodic=setup.case.is_periodic)

    if rcon.is_head_rank:
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    if setup.quad_min_degree is None:
        quad_min_degrees = {
                "gasdyn_vol": 3*setup.order,
                "gasdyn_face": 3*setup.order,
                }
    elif setup.quad_min_degree == 0:
        quad_min_degrees = {}
    else:
        quad_min_degrees = {
                "quad": setup.quad_min_degree,
                "gasdyn_vol": setup.quad_min_degree,
                "gasdyn_face": setup.quad_min_degree,
                }

    discr = rcon.make_discretization(mesh_data, order=setup.order,
            quad_min_degrees=quad_min_degrees,
            debug=[
            #"dump_optemplate_stages",
            #"dump_op_code"
            ]
            )

    return rcon, mesh_data, discr




def l1_norm(discr, field):
    if discr.dimensions > 0:
        return discr.norm(field, 1)
    else:
        from scipy.integrate import trapz
        from pytools.obj_array import with_object_array_or_scalar
        return numpy.sum(with_object_array_or_scalar(
                lambda f: trapz(numpy.abs(f), discr.nodes[:, 0]),
                field))
