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


def survey():
    O = ConstructorPlaceholder

    timestamp = get_timestamp()
    for order in [3, 4, 5, 10]:
        for n_elements in [20, 40]:
            for case in [
                    #O("CenteredStationaryTestCase"),
                    #O("OffCenterStationaryTestCase"),
                    #O("OffCenterMigratingTestCase"),
                    O("ExactTestCase"),
                    ]:
                for smoother in [
                        None,
                        O("TriBlobSmoother", use_max=False),
                        O("TriBlobSmoother", use_max=True),
                        ]:
                    for sensor in [
                            "persson_peraire",
                            "decay_gating",
                            "decay_gating skyline",
                            "decay_gating averaging",
                            "decay_gating fit_correction",
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
                            "case = %s" % case,
                            "sensor = %r" % sensor,
                            ])
                        job.submit()

                        #raw_input()





import sys
exec sys.argv[1]
