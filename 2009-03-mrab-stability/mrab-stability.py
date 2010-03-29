from __future__ import division
from pytools import Record
import mpi4py.MPI as mpi
import sqlite3 as sqlite
import numpy
import numpy.linalg as la

comm = mpi.COMM_WORLD




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

    for method in methods.keys()[:2]:
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

    def exact_map(self, t):
        mat = numpy.exp(numpy.diag([-1, -1*self.ratio])*t)
        evmat = self.get_eigvec_mat()
        return numpy.dot(la.solve(evmat, mat), evmat)

    def __call__(self):
        mat = numpy.diag([-1, -1*self.ratio])
        evmat = self.get_eigvec_mat()
        return numpy.dot(la.solve(evmat, mat), evmat)




def generate_matrix_factories():
    from hedge.timestep.multirate_ab.methods import methods
    from hedge.timestep.stability import approximate_imag_stability_region

    angle_steps = 5
    offset_steps = 5
    from math import pi
    for ratio in numpy.linspace(0.1, 1, 10):
        for angle in numpy.linspace(0, 2*pi, angle_steps, endpoint=False):
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
        print "leave", stable_dt
        return {
                "dt": stable_dt
                }



def generate_mrab_jobs():
    for method_fac in generate_method_factories():
        for matrix_fac in generate_matrix_factories():
            yield MRABJob(method_fac, matrix_fac)




# {{{ sample job --------------------------------------------------------------
class SleepJob(Record):
    def __init__(self, sleep_duration):
        Record.__init__(self,
                sleep_duration=sleep_duration)

    def get_parameter_dict(self):
        return {"sleep_duration": self.sleep_duration}

    def __call__(self):
        from time import time, sleep
        from socket import gethostname
        start = time()
        sleep(self.sleep_duration)
        stop = time()
        return {
                "start": start,
                "stop": stop,
                "host": gethostname()
                }

def generate_sleep_jobs():
    from random import random
    for i in range(100):
        yield SleepJob(random())

# }}}

# {{{ queue management --------------------------------------------------------

def manage_queue(job_generator, outfile):
    try:
        from os.path import exists
        if exists(outfile):
            raise RuntimeError("output file '%s' exists" % outfile)

        parameter_cols = {}
        result_cols = {}
        rows = []

        free_ranks = set(range(1, comm.Get_size()))

        job_it = iter(job_generator())
        no_more_jobs = False

        while True:
            if not no_more_jobs:
                while free_ranks:
                    try:
                        next_job = job_it.next()
                    except StopIteration:
                        no_more_jobs = True
                        break
                    else:
                        assigned_rank = free_ranks.pop()
                        #print "job -> rank %d" % assigned_rank
                        comm.send(next_job, dest=assigned_rank)

            if no_more_jobs:
                if len(free_ranks) == comm.Get_size() - 1:
                    break

            rank, job, result = comm.recv(source=mpi.ANY_SOURCE)
            free_ranks.add(rank)

            for k, v in job.get_parameter_dict().iteritems():
                parameter_cols[k] = type(v)
            for k, v in result.iteritems():
                result_cols[k] = type(v)
            rows.append((job.get_parameter_dict(), result))

        print "storing"
        db_conn = sqlite.connect(outfile, timeout=30)

        # done
        parameter_cols = list(parameter_cols.iteritems())
        result_cols = list(result_cols.iteritems())
        columns = parameter_cols + result_cols

        def get_sql_type(tp):
            if tp == str:
                return "text"
            elif issubclass(tp, int):
                return "integer"
            elif issubclass(tp, (float, numpy.floating)):
                return "real"
            else:
                raise TypeError("No SQL type for %s" % tp)

        db_conn.execute("create table data (%s)"
                % ",".join("%s %s" % (name, get_sql_type(tp))
                    for name, tp in columns))

        insert_stmt = "insert into data values (%s)" % (
                ",".join(["?"]*len(columns)))

        par_col_count = len(parameter_cols)

        for parameter_data, result_data in rows:
            data = [None] * len(columns)
            for i, (col_name, col_tp) in enumerate(parameter_cols):
                data[i] = parameter_data[col_name]
            for i, (col_name, col_tp) in enumerate(result_cols):
                data[i+par_col_count] = result_data[col_name]

            db_conn.execute(insert_stmt, data)
        db_conn.commit()
        db_conn.close()

    finally:
        print "quitting"
        for rank in range(1, comm.Get_size()):
            comm.send("quit", dest=rank)

def serve_queue():
    rank = comm.Get_rank()
    while True:
        job = comm.recv(source=0)
        if isinstance(job, str) and job =="quit":
            return

        result = job()
        comm.send((rank, job, result), dest=0)

def main(*args, **kwargs):
    rank = comm.Get_rank()
    if rank == 0:
        manage_queue(*args, **kwargs)
    else:
        serve_queue()

# }}}




if __name__ == "__main__":
    #main(generate_sleep_jobs, "output.dat")
    main(generate_mrab_jobs, "output.dat")
