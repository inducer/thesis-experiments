from __future__ import division
from pytools import Record
import numpy
import numpy.linalg as la




# {{{ utilities ---------------------------------------------------------------
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

class StabilityTester(object):
    def __init__(self, method_fac, matrix_fac, stable_steps):
        self.method_fac = method_fac
        self.matrix_fac = matrix_fac
        self.matrix = matrix_fac()
        self.stable_steps = stable_steps

    def get_parameter_dict(self):
        result = {"stable_steps": self.stable_steps}
        result.update(self.method_fac.get_parameter_dict())
        result.update(self.matrix_fac.get_parameter_dict())
        return result

    def refine(self, stable, unstable):
        assert self.is_stable(stable)
        assert not self.is_stable(unstable)
        while abs(stable-unstable) > self.prec:
            mid = (stable+unstable)/2
            if self.is_stable(mid):
                stable = mid
            else:
                unstable = mid
        else:
            return stable

    def find_stable_dt(self):
        dt = 0.1

        if self.is_stable(dt):
            dt *= 2
            while self.is_stable(dt):
                dt *= 2

                if dt > 2**8:
                    return dt
            return self.refine(dt/2, dt)
        else:
            dt /= 2
            while not self.is_stable(dt):
                dt /= 2

                if dt < self.prec:
                    return dt
            return self.refine(dt, dt*2)

    def __call__(self):
        return { "dt": self.find_stable_dt() }




# }}}

# {{{ matrices ----------------------------------------------------------------
class MatrixFactory(FactoryWithParameters):
    __slots__ = ["ratio", "angle", "offset"]

    def get_parameter_dict(self):
        res = FactoryWithParameters.get_parameter_dict(self)
        res["mat_type"] = type(self).__name__
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




class DecayMatrixFactory(MatrixFactory):
    __slots__ = []

    def __call__(self):
        vec = numpy.array([-1, -1*self.ratio])
        mat = numpy.diag(vec)
        evmat = self.get_eigvec_mat()
        return numpy.dot(la.solve(evmat, mat), evmat)

class DecayOscillationMatrixFactory(MatrixFactory):
    __slots__ = []

    def __call__(self):
        vec = numpy.array([-1, 1j*self.ratio])
        mat = numpy.diag(vec)
        evmat = self.get_eigvec_mat()
        return numpy.dot(la.solve(evmat, mat), evmat)

class OscillationDecayMatrixFactory(MatrixFactory):
    __slots__ = []

    def __call__(self):
        vec = numpy.array([1j, -1*self.ratio])
        mat = numpy.diag(vec)
        evmat = self.get_eigvec_mat()
        return numpy.dot(la.solve(evmat, mat), evmat)

class OscillationMatrixFactory(MatrixFactory):
    __slots__ = []

    def __call__(self):
        vec = numpy.array([1j, 1j*self.ratio])
        mat = numpy.diag(vec)
        evmat = self.get_eigvec_mat()
        from hedge.tools.linalg import leftsolve
        return numpy.dot(evmat, leftsolve(evmat, mat))



def generate_matrix_factories():
    from math import pi

    angle_steps = 20
    offset_steps = 20
    for angle in numpy.linspace(0, pi, angle_steps, endpoint=False):
        for offset in numpy.linspace(
                pi/offset_steps, 
                pi, offset_steps, endpoint=False):
            for ratio in numpy.linspace(0.1, 1, 10):
                yield DecayMatrixFactory(ratio=ratio, angle=angle, offset=offset)
                yield OscillationMatrixFactory(ratio=ratio, angle=angle, offset=offset)
                yield OscillationDecayMatrixFactory(ratio=ratio, angle=angle, offset=offset)
                yield DecayOscillationMatrixFactory(ratio=ratio, angle=angle, offset=offset)





def generate_matrix_factories_hires():
    from math import pi

    offset_steps = 100
    for angle in [0, 0.05*pi, 0.1*pi]:
        for offset in numpy.linspace(
                pi/offset_steps, 
                pi, offset_steps, endpoint=False):
            for ratio in numpy.linspace(0.1, 1, 100):
                yield DecayMatrixFactory(ratio=ratio, angle=angle, offset=offset)
                yield OscillationMatrixFactory(ratio=ratio, angle=angle, offset=offset)
                yield OscillationDecayMatrixFactory(ratio=ratio, angle=angle, offset=offset)
                yield DecayOscillationMatrixFactory(ratio=ratio, angle=angle, offset=offset)

# }}}

# {{{ MRAB --------------------------------------------------------------------

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

    for method in methods.keys():
        for order in [3]:
            for substep_count in [2, 3, 4]:
                yield MethodFactory(method=method, meth_order=order, 
                        substep_count=substep_count)




def generate_method_factories_hires():
    from hedge.timestep.multirate_ab.methods import methods

    for method in ["Fq", "Ssf", "Sr"]:
        for order in [3]:
            for substep_count in [2, 5, 10]:
                yield MethodFactory(method=method, meth_order=order, 
                        substep_count=substep_count)






class MRABJob(StabilityTester):
    prec = 1e-4

    def is_stable(self, dt):
        stepper = self.method_fac(dt)
        mat = self.matrix

        y = numpy.array([1,1], dtype=numpy.float64)
        y /= la.norm(y)

        def f2f_rhs(t, yf, ys): return mat[0,0] * yf()
        def s2f_rhs(t, yf, ys): return mat[0,1] * ys()
        def f2s_rhs(t, yf, ys): return mat[1,0] * yf()
        def s2s_rhs(t, yf, ys): return mat[1,1] * ys()

        for i in range(self.stable_steps):
            y = stepper(y, i*dt, 
                    (f2f_rhs, s2f_rhs, f2s_rhs, s2s_rhs))
            if la.norm(y) > 10:
                return False

        return True



def generate_mrab_jobs():
    for method_fac in generate_method_factories():
        for matrix_fac in generate_matrix_factories():
            yield MRABJob(method_fac, matrix_fac, 120)




def generate_mrab_jobs_hires():
    for method_fac in generate_method_factories_hires():
        for matrix_fac in generate_matrix_factories_hires():
            yield MRABJob(method_fac, matrix_fac, 120)




def generate_mrab_jobs_step_verify():
    from math import pi

    def my_generate_matrix_factories():

        offset_steps = 20
        for angle in [0.05*pi]:
            for offset in numpy.linspace(
                    pi/offset_steps, 
                    pi, offset_steps, endpoint=False):
                for ratio in numpy.linspace(0.1, 1, 10):
                    yield DecayMatrixFactory(ratio=ratio, angle=angle, offset=offset)
                    yield OscillationMatrixFactory(ratio=ratio, angle=angle, offset=offset)
                    yield OscillationDecayMatrixFactory(ratio=ratio, angle=angle, offset=offset)
                    yield DecayOscillationMatrixFactory(ratio=ratio, angle=angle, offset=offset)

    for method_fac in list(generate_method_factories())[:1]:
        for matrix_fac in my_generate_matrix_factories():
            for stable_steps in [40, 80, 120]:
                yield MRABJob(method_fac, matrix_fac, stable_steps)

# }}}

# {{{ single-rate reference ---------------------------------------------------
class SRABMethodFactory(FactoryWithParameters):
    __slots__ = ["method", "substep_count", "meth_order"]

    def __call__(self):
        from hedge.timestep.ab import AdamsBashforthTimeStepper
        return AdamsBashforthTimeStepper(order=self.meth_order,
                dtype=numpy.complex128)




class SRABJob(StabilityTester):
    prec = 1e-4

    def is_stable(self, dt):
        stepper = self.method_fac()

        y = numpy.array([1,1], dtype=numpy.complex128)
        y /= la.norm(y)

        def rhs(t, y):
            return numpy.dot(self.matrix, y)

        for i in range(self.stable_steps):
            y = stepper(y, i*dt, dt, rhs)
            if la.norm(y) > 10:
                return False

        return True




def generate_srab_jobs():
    for method_fac in [SRABMethodFactory(method="SRAB", substep_count=1, meth_order=3)]:
        for matrix_fac in generate_matrix_factories():
            yield SRABJob(method_fac, matrix_fac, 120)

# }}}

if __name__ == "__main__":
    from pytools.prefork import enable_prefork
    enable_prefork()

    from mpi_queue import enter_queue_manager
    enter_queue_manager(generate_srab_jobs, "output-srab.dat")
    enter_queue_manager(generate_mrab_jobs, "output.dat")
    enter_queue_manager(generate_mrab_jobs_hires, "output-hires.dat")
    #enter_queue_manager(generate_mrab_jobs_step_verify, "output-step.dat")

# vim: foldmethod=marker
