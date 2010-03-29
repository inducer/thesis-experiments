from __future__ import division
from pytools import Record
import numpy
import numpy.linalg as la




class FactoryWithParameters(Record):
    __slots__ = []

    def get_parameter_dict(self):
        result = {}
        for f in self.__class__.fields:
            try:
                result[intern(f)] = getattr(self, f)
            except AttributeError:
                pass
        return result

class MethodFactory(FactoryWithParameters):
    __slots__ = ["method", "substep_count", "meth_order"]

    def __call__(self, dt):
        from hedge.timestep.multirate_ab import \
                TwoRateAdamsBashforthTimeStepper
        return TwoRateAdamsBashforthTimeStepper(
                method=self.method,
                large_dt=dt,
                substep_count=self.substep_count,
                order=self.meth_order)





def generate_method_factories():
    from hedge.timestep.multirate_ab.methods import methods
    from hedge.timestep.stability import approximate_imag_stability_region

    for method in methods.keys()[:1]:
        for order in [3]:
            for substep_count in [2]:
                yield MethodFactory(method=method, meth_order=order, 
                        substep_count=substep_count)




class RealMatrixFactory(FactoryWithParameters):
    __slots__ = ["ratio", "angle", "offset"]

    def get_parameter_dict(self):
        res = FactoryWithParameters.get_parameter_dict(self)
        res["mat_type"] = "realvalued"
        return res

    def get_eigvec_mat(self):
        from math import cos, sin
        return numpy.array([
            [cos(self.angle), cos(self.angle+self.offset)],
            [sin(self.angle), sin(self.angle+self.offset)],
            ])

    def __call__(self):
        mat = numpy.diag([-1, -1*self.ratio])
        evmat = self.get_eigvec_mat()
        return numpy.dot(la.solve(evmat, mat), evmat)




class ComplexConjugateMatrixFactory(FactoryWithParameters):
    __slots__ = ["complex_arg", "angle", "ev_arg"]

    def get_parameter_dict(self):
        res = FactoryWithParameters.get_parameter_dict(self)
        res["mat_type"] = "conjugate"
        return res

    def get_eigvec_mat(self):
        from math import cos, sin
        from cmath import exp
        b = self.angle
        c = self.ev_arg
        return numpy.array([
            [cos(b)*exp(1j*c), cos(b)*exp(-1j*c)],
            [sin(b)*exp(-1j*c), sin(b)*exp(1j*c)],
            ])

    def __call__(self):
        from math import cos, sin

        a = self.complex_arg
        mat = numpy.diag([cos(a) + 1j *sin(a), cos(a) - 1j*sin(a)])
        evmat = self.get_eigvec_mat()
        return numpy.real_if_close(
                numpy.dot(la.solve(evmat, mat), evmat))




def generate_matrix_factories():
    from math import pi

    complex_arg_steps = 20
    angle_steps = 20
    ev_arg_steps = 20
    for complex_arg in numpy.linspace(0, pi, complex_arg_steps):
        for angle in numpy.linspace(pi/angle_steps, pi, angle_steps):
            for ev_arg in numpy.linspace(
                    pi/ev_arg_steps, pi, ev_arg_steps):

                yield ComplexConjugateMatrixFactory(
                        complex_arg=complex_arg, 
                        angle=angle, 
                        ev_arg=ev_arg)

    if False:
        angle_steps = 20
        offset_steps = 40
        for ratio in numpy.linspace(0.1, 1, 10):
            for angle in numpy.linspace(0, pi, angle_steps, endpoint=False):
                for offset in numpy.linspace(
                        2*pi/offset_steps, 
                        2*pi, offset_steps, endpoint=False):
                    yield RealMatrixFactory(ratio=ratio, angle=angle, offset=offset)




class MRABJob(object):
    def __init__(self, method_fac, matrix_fac):
        self.method_fac = method_fac
        self.matrix_fac = matrix_fac

    def get_parameter_dict(self):
        result = {}
        result.update(self.method_fac.get_parameter_dict())
        result.update(self.matrix_fac.get_parameter_dict())
        return result

    def __call__(self):
        mat = self.matrix_fac()

        def is_stable(dt):
            stepper = self.method_fac(dt)

            y = numpy.array([1,1], dtype=numpy.float64)
            y /= la.norm(y)

            def f2f_rhs(t, yf, ys): return mat[0,0] * yf()
            def s2f_rhs(t, yf, ys): return mat[0,1] * ys()
            def f2s_rhs(t, yf, ys): return mat[1,0] * yf()
            def s2s_rhs(t, yf, ys): return mat[1,1] * ys()

            for i in range(30):
                y = stepper(y, i*dt, 
                        (f2f_rhs, s2f_rhs, f2s_rhs, s2s_rhs))
                if la.norm(y) > 2:
                    return False

            return True

        prec = 1e-4

        def refine(stable, unstable):
            assert is_stable(stable)
            assert not is_stable(unstable)
            while abs(stable-unstable) > prec:
                mid = (stable+unstable)/2
                if is_stable(mid):
                    stable = mid
                else:
                    unstable = mid
            else:
                return stable

        def find_stable_dt():
            dt = 0.1

            if is_stable(dt):
                dt *= 2
                while is_stable(dt):
                    dt *= 2

                    if dt > 2**8:
                        return dt
                return refine(dt/2, dt)
            else:
                dt /= 2
                while not is_stable(dt):
                    dt /= 2

                    if dt < prec:
                        return dt
                return refine(dt, dt*2)

        stable_dt = find_stable_dt()
        return {
                "dt": stable_dt
                }



def generate_mrab_jobs():
    for method_fac in generate_method_factories():
        for matrix_fac in generate_matrix_factories():
            yield MRABJob(method_fac, matrix_fac)




if __name__ == "__main__":
    from mpi_queue import enter_queue_manager
    #enter_queue_manager(generate_sleep_jobs, "output.dat")
    enter_queue_manager(generate_mrab_jobs, "output.dat")
