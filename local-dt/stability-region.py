from __future__ import division
import numpy
from cmath import pi




prec = 1e-5

points = []

def is_stable(stepper, k):
    y = 1
    for i in range(20):
        if abs(y) > 2:
            return False
        y = stepper(y, i, 1, lambda t, y: k*y)
    return True

def make_k(angle, mag):
    from cmath import exp
    return -prec+mag*exp(1j*angle)

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

def make_ab():
    from hedge.timestep import AdamsBashforthTimeStepper
    return AdamsBashforthTimeStepper(3)

from hedge.timestep import RK4TimeStepper

sm = RK4TimeStepper
#sm = make_ab

for angle in numpy.arange(0, 2*pi, 2*pi/200):
    print angle
    points.append(make_k(angle, find_stable_k(sm, angle)))

points = numpy.array(points)

from matplotlib.pyplot import *

title("Stability Region")
xlabel("Re $k$")
ylabel("Im $k$")
fill(points.real, points.imag)
grid()
show()
