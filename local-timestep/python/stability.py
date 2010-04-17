from __future__ import division
import numpy
from cmath import pi




prec = 1e-5
origin = -0.1

def is_stable(stepper, k):
    y = 1
    for i in range(100):
        if abs(y) > 2:
            return False
        y = stepper(y, i, 1, lambda t, y: k*y)
    return True

def make_k(angle, mag):
    from cmath import exp
    return origin+mag*exp(1j*angle)

def refine(stepper_maker, angle, stable, unstable):
    assert is_stable(stepper_maker(), make_k(angle, stable))
    assert not is_stable(stepper_maker(), make_k(angle, unstable))
    while abs(stable-unstable) > prec:
        mid = (stable+unstable)/2
        if is_stable(stepper_maker(), make_k(angle, mid)):
            stable = mid
        else:
            unstable = mid
    else:
        return stable

def find_stable_k(stepper_maker, angle):
    mag = 1

    if is_stable(stepper_maker(), make_k(angle, mag)):
        # try to grow
        mag *= 2
        while is_stable(stepper_maker(), make_k(angle, mag)):
            mag *= 2

            if mag > 2**8:
                return mag
        return refine(stepper_maker, angle, mag/2, mag)
    else:
        mag /= 2
        while not is_stable(stepper_maker(), make_k(angle, mag)):
            mag /= 2

            if mag < prec:
                return mag
        return refine(stepper_maker, angle, mag, mag*2)

class StabPointFinder:
    def __init__(self, stepper_maker):
        self.stepper_maker = stepper_maker

    def __call__(self, angle):
        return make_k(angle, find_stable_k(self.stepper_maker, angle))

def plot_stability_region(stepper_maker, **kwargs):
    points = []
    angles = numpy.arange(0, 2*pi, 2*pi/200)
    from multiprocessing import Pool
    points = Pool().map(StabPointFinder(stepper_maker), angles)
    #points = map(StabPointFinder(stepper_maker), angles)

    points = numpy.array(points)

    from matplotlib.pyplot import fill
    fill(points.real, points.imag, **kwargs)

class ABMaker:
    def __init__(self, order):
        self.order = order

    def __call__(self):
        from hedge.timestep import AdamsBashforthTimeStepper
        return AdamsBashforthTimeStepper(self.order)

class DumkaMaker:
    def __init__(self, pol_index):
        self.pol_index = pol_index

    def __call__(self):
        from hedge.timestep.dumka3 import Dumka3TimeStepper
        return Dumka3TimeStepper(self.pol_index, dtype=numpy.complex128)

class LSRK4Maker:
    def __call__(self):
        from hedge.timestep.runge_kutta import LSRK4TimeStepper
        return LSRK4TimeStepper(dtype=numpy.complex128)

class ODE23Maker:
    def __call__(self):
        from hedge.timestep.runge_kutta import ODE23TimeStepper
        return ODE23TimeStepper(dtype=numpy.complex128)

class ODE45Maker:
    def __call__(self):
        from hedge.timestep.runge_kutta import ODE45TimeStepper
        return ODE45TimeStepper(dtype=numpy.complex128)

if __name__ == "__main__":
    #sm = RK4TimeStepper
    #sm = ABMaker(5)
    #sm = SSPRK3TimeStepper
    sm = DumkaMaker(1)
    from matplotlib.pyplot import *

    rc("font", size=20)
    #title("Stability Region")
    xlabel(r"Re $\lambda$")
    ylabel(r"Im $\lambda$")
    grid()

    for pol_i in range(3, 0, -1):
        plot_stability_region(DumkaMaker(pol_i), 
                label="Dumka3(%d)" % pol_i)
    plot_stability_region(LSRK4Maker(), label="C/K RK4")
    #plot_stability_region(ODE23Maker(), label="ODE23", alpha=0.3)
    #plot_stability_region(ODE45Maker(), label="ODE45", alpha=0.3)

    legend(labelspacing=0.1, borderpad=0.3, loc="best")
    savefig("stab-regions.pdf")
    xmin, xmax = xlim()
    xsize = xmax-xmin

    gca().set_aspect("equal")
    ylim([-xsize/2*0.75, xsize/2*0.75])
    savefig("stab-regions-eq-aspect.pdf")
    #show()


