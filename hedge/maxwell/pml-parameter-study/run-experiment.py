from pytools.batchjob import \
        guess_job_class, \
        ConstructorPlaceholder, \
        get_timestamp

BatchJob = guess_job_class()

def pml_experiment():
    """Submit jobs to compare reconstruction/pushing methods."""

    O = ConstructorPlaceholder

    timestamp = get_timestamp()

    pml_mags = [1, 3, 10, 30, 100]
    for absorb_back in [True, False]:
        for pml_mag in pml_mags:
            for tau_mag in [0, 0.5, 1, 3, 10]:
                for pml_width in [0.1, 0.25, 0.5]:
                    for pml_exp in [2,3,4]:
                        for max_vol in [0.01, 0.03]:
                            job = BatchJob(
                                    "pmlstudy-$DATE/abs%s-mag%.1g-tau%.1g-exp%d-pmlw%.1g-vol%.1g"
                                    % (absorb_back, pml_mag, tau_mag, pml_exp, pml_width, max_vol),
                                    "maxwell-pml.py",
                                    timestamp=timestamp)
                            job.write_setup([
                                "absorb_back = %s" % absorb_back,
                                "pml_mag = %r" % pml_mag,
                                "tau_mag = %r" % tau_mag,
                                "pml_exp = %d" % pml_exp,
                                "pml_width = %r" % pml_width,
                                "max_vol = %r" % max_vol,
                                ])

                            job.submit()
                            raw_input()

                            from time import sleep
                            sleep(0.1)

if __name__ == "__main__":
    pml_experiment()

