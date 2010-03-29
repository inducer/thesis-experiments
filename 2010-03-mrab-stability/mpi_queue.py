from __future__ import division
from pytools import Record
import numpy
import sqlite3 as sqlite
import mpi4py.MPI as mpi

comm = mpi.COMM_WORLD



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
    job_count = 0

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

            job_count += 1
            rank, job, result = comm.recv(source=mpi.ANY_SOURCE)
            free_ranks.add(rank)

            if job_count % 50 == 0:
                print job_count, "done"

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

def enter_queue_manager(*args, **kwargs):
    rank = comm.Get_rank()
    if rank == 0:
        manage_queue(*args, **kwargs)
    else:
        serve_queue()

# }}}




if __name__ == "__main__":
    enter_queue_manager(generate_sleep_jobs, "output.dat")




# vim: foldmethod=marker
