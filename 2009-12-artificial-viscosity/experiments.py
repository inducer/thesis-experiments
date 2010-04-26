from pytools.batchjob import (
        guess_job_class,
        ConstructorPlaceholder,
        get_timestamp)

BatchJob = guess_job_class()

def cn_with_args(placeholder):
    if placeholder is None:
        return "None"

    result = placeholder.classname
    if placeholder.args:
        result += "-" + ("-".join(str(a) for a in args))

    def mogrify_kwarg(k, v):
        result = k[0]

        i = 1
        while i < len(k):
            if k[i] == "_":
                result += k[i+1]
                i += 2
            else:
                i += 1

        return result+str(v)

    if placeholder.kwargs:
        result += "-" + ("-".join(
            mogrify_kwarg(k, v)
            for k, v in placeholder.kwargs.iteritems()))

    return result


def burgers_survey():
    O = ConstructorPlaceholder

    timestamp = get_timestamp()
    for order in [3, 4, 5, 10]:
        for n_elements in [20, 40]:
            for case in [
                    #O("CenteredStationaryTestCase"),
                    #O("OffCenterStationaryTestCase"),
                    #O("OffCenterMigratingTestCase"),
                    O("LeaningTriangleTestCase"),
                    ]:
                for smoother in [
                        None,
                        O("TriBlobSmoother", use_max=False),
                        O("TriBlobSmoother", use_max=True),
                        ]:
                    for sensor in [
                            #"persson_peraire",
                            "decay_gating",
                            "decay_gating skyline",
                            "decay_gating averaging",
                            #"decay_gating fit_correction",
                            ]:
                        job = BatchJob(
                                "burgers-$DATE/N%d-K%d-%s-sm%s-%s" % (
                                    order,
                                    n_elements,
                                    case.classname,
                                    cn_with_args(smoother),
                                    sensor.replace(" ", "."),
                                    ),
                                "burgers.py",
                                timestamp=timestamp,
                                aux_files=["smoother.py"])

                        job.write_setup([
                            "order = %d" % order,
                            "n_elements = %d" % n_elements,
                            "smoother = %s" % smoother,
                            "vis_interval = 5",
                            "case = %s" % case,
                            "sensor = %r" % sensor,
                            ])
                        job.submit()

                        #raw_input()




def euler_sod_convergence():
    O = ConstructorPlaceholder
    timestamp = get_timestamp()

    for order in [4, 5, 7, 9]:
        for n_elements in [20, 40, 80, 160, 320, 640]:
            if n_elements >= 500 and order > 7:
                continue

            n_elements += 1

            for viscosity_scale in [0.2, 0.4, 0.8]:
                for smoother in [
                        O("TriBlobSmoother", use_max=False),
                        O("VertexwiseMaxSmoother"),
                        ]:
                    job = BatchJob(
                            "euler-$DATE/N%d-K%d-v%f-%s" % (
                                order,
                                n_elements,
                                viscosity_scale,
                                cn_with_args(smoother),
                                ),
                            "euler.py",
                            timestamp=timestamp,
                            aux_files=["smoother.py", "avcommon.py", "sod.py",
                                "euler_airplane.py"])

                    job.write_setup([
                        "order = %d" % order,
                        "n_elements = %d" % n_elements,
                        "viscosity_scale = %r" % viscosity_scale,
                        "vis_interval = 0.05",
                        "case = SodProblem()",
                        "smoother = %s" % smoother,
                        "vis_order = %d" % (2*order),
                        ])
                    job.submit()


def euler_sod_convergence_in_k():
    O = ConstructorPlaceholder
    timestamp = get_timestamp()

    for order in [5]:
        for n_elements in [20, 40, 80, 160, 320]:

            for viscosity_scale in [0.4]:
                for smoother in [
                        #O("TriBlobSmoother", use_max=False),
                        O("VertexwiseMaxSmoother"),
                        ]:
                    job = BatchJob(
                            "sod-kconv-$DATE/N%d-K%d-v%f-%s" % (
                                order,
                                n_elements,
                                viscosity_scale,
                                cn_with_args(smoother),
                                ),
                            "euler.py",
                            timestamp=timestamp,
                            aux_files=["smoother.py", "avcommon.py", "sod.py",
                                "euler_airplane.py"])

                    job.write_setup([
                        "order = %d" % order,
                        "n_elements = %d" % n_elements,
                        "viscosity_scale = %r" % viscosity_scale,
                        "vis_interval = 0.002",
                        "case = SodProblem()",
                        "smoother = %s" % smoother,
                        #"vis_exact = False",
                        ])
                    job.submit()




def advection_convergence_in_k():
    O = ConstructorPlaceholder
    timestamp = get_timestamp()

    for order in [4]:
        for n_elements in [20, 40, 80, 160, 320, 640, 1280]:
        #for n_elements in [40]:
            for viscosity_scale in [0.2]:
                for smoother in [
                        O("VertexwiseMaxSmoother"),
                        ]:
                    job = BatchJob(
                            "adv-kconv-$DATE/N%d-K%d-v%f-%s" % (
                                order,
                                n_elements,
                                viscosity_scale,
                                cn_with_args(smoother),
                                ),
                            "advection.py",
                            timestamp=timestamp,
                            aux_files=["smoother.py", "avcommon.py"])

                    job.write_setup([
                        "order = %d" % order,
                        "n_elements = %d" % n_elements,
                        "viscosity_scale = %r" % viscosity_scale,
                        "vis_interval = 0.2",
                        "case = FunnySineTestCase()",
                        "smoother = %s" % smoother,
                        #"vis_exact = False",
                        ])
                    job.submit()

def wave_convergence_in_k():
    O = ConstructorPlaceholder
    timestamp = get_timestamp()

    for case, vis_interval in [
            (O("TwoJumpTestCase"), 0.2),
            (O("LaxerSineTestCase"), 0.005),
            (O("PureSineTestCase"), 0.005),
            ]:
        for order in [5]:
            for n_elements in [20, 40, 80, 160, 320, 640]:
                for viscosity_scale in [0, 1]:
                    for smoother in [
                            O("VertexwiseMaxSmoother"),
                            ]:
                        job = BatchJob(
                                "wave-kconv-$DATE/N%d-K%d-v%f-%s-%s" % (
                                    order,
                                    n_elements,
                                    viscosity_scale,
                                    cn_with_args(smoother),
                                    cn_with_args(case),
                                    ),
                                "wave.py",
                                timestamp=timestamp,
                                aux_files=["smoother.py", "avcommon.py"])

                        job.write_setup([
                            "order = %d" % order,
                            "n_elements = %d" % n_elements,
                            "viscosity_scale = %f" % viscosity_scale,
                            "vis_interval = %f" % vis_interval,
                            "case = %s" % case,
                            "smoother = %s" % smoother,
                            #"vis_exact = False",
                            ])
                        job.submit()


import sys
exec sys.argv[1]
