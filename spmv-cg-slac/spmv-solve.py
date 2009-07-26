from __future__ import division
from __future__ import with_statement
from pytools import memoize_method
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import numpy
import numpy.linalg as la
import pyublas




def main_coo():
    from optparse import OptionParser

    parser = OptionParser(
            usage="%prog [options] MATRIX-MARKET-FILE")
    options, args = parser.parse_args()

    from scipy.io import mmread
    mat = mmread(args[0])

    from coordinate import CoordinateSpMV
    spmv = CoordinateSpMV(mat, numpy.float32)

    x = numpy.random.randn(spmv.shape[0]).astype(spmv.dtype)

    x_gpu = gpuarray.to_gpu(x)
    y_gpu = spmv(x_gpu)
    y = y_gpu.get()

    y2 = mat*x

    print la.norm(y-y2)/la.norm(y)




def main_pkt():
    from optparse import OptionParser

    parser = OptionParser(
            usage="%prog [options] MATRIX-MARKET-FILE")
    parser.add_option("-s", "--is-symmetric", action="store_true",
            help="Specify that the input matrix is already symmetric")
    options, args = parser.parse_args()

    from scipy.io import mmread
    print "loading..."
    mat = mmread(args[0])
    print "done"

    from packeted import PacketedSpMV
    spmv = PacketedSpMV(mat, options.is_symmetric, numpy.float32)

    x = numpy.random.randn(spmv.shape[0]).astype(spmv.dtype)

    x_gpu = spmv.permute(gpuarray.to_gpu(x))

    print "warmup"
    # warmup
    y_gpu = gpuarray.zeros(spmv.shape[0], dtype=spmv.dtype)

    for i in range(5):
        y_gpu = spmv(x_gpu, y_gpu)

    y_gpu = gpuarray.zeros(spmv.shape[0], dtype=spmv.dtype)

    print "benchmark"

    start = drv.Event()
    stop = drv.Event()
    
    drv.Context.synchronize()
    rep_count = 20
    start.record()
    for i in range(rep_count):
        y_gpu = spmv(x_gpu, y_gpu)
    stop.record()
    stop.synchronize()

    elapsed = stop.time_since(start)*1e-3
    est_flops = (mat.nnz*2*rep_count)
    print "%d nnz, shape: %s, %g s/mult, %g gflops/s" % (
            mat.nnz, mat.shape, elapsed/rep_count, est_flops/elapsed/1e9)

    y = spmv.unpermute(y_gpu).get()

    y2 = mat*x

    print la.norm(y/rep_count-y2)/la.norm(y)

def solve_pkt_with_cg(pkt_spmv, b, x=None, tol=1e-7, max_iterations=None,
        debug=False):
    if x is None:
        x = gpuarray.zeros(pkt_spmv.shape[0], dtype=pkt_spmv.dtype)
    else:
        x = pkt_spmv.permute(x)

    def gpu_dot(x, y):
        return gpuarray.dot(x, y).get()

    from hedge.tools import CGStateContainer
    cg = CGStateContainer(pkt_spmv, dot=gpu_dot)
    cg.reset(pkt_spmv.permute(b), x)

    op_it_count = [0]
    total_it_count = [0]
    def debug_callback(what, it_count, x, resid, d, delta):
        if what == "it":
            op_it_count[0] += 1
            total_it_count[0] += 1
        elif what == "it-noop":
            total_it_count[0] += 1

    result = cg.run(max_iterations, tol, debug=debug, 
            debug_callback=debug_callback)

    return pkt_spmv.unpermute(result), op_it_count[0], total_it_count[0]


def main_cg():
    from optparse import OptionParser

    parser = OptionParser(
            usage="%prog [options] MATRIX-MARKET-FILE")
    parser.add_option("-s", "--is-symmetric", action="store_true",
            help="Specify that the input matrix is already symmetric")
    options, args = parser.parse_args()

    from scipy.io import mmread
    mat = mmread(args[0])

    from packeted import PacketedSpMV
    spmv = PacketedSpMV(mat, options.is_symmetric, numpy.float64)
    rhs = numpy.random.rand(spmv.shape[0]).astype(spmv.dtype)

    for i in range(4):
        print "start solve"
        start = drv.Event()
        stop = drv.Event()
        start.record()

        rhs_gpu = gpuarray.to_gpu(rhs)
        res_gpu, op_it_count, total_it_count = \
                solve_pkt_with_cg(spmv, rhs_gpu, 
                        tol=1e-7 if spmv.dtype == numpy.float64 else 5e-5)
        res = res_gpu.get()

        stop.record()
        stop.synchronize()

        elapsed = stop.time_since(start)*1e-3
        est_flops = (mat.nnz*2*op_it_count
            + mat.shape[0]*(2+2+2+2+2)*total_it_count)
                    
        print "residual norm: %g" % (la.norm(mat*res - rhs)/la.norm(rhs))
        print ("size: %d, elapsed: %g s, %d op it, %d total it, it/second: %g, "
                "%g gflops/s" % (
                    mat.shape[0],
                    elapsed, op_it_count, total_it_count, total_it_count/elapsed,
                    est_flops/elapsed/1e9))

    # TODO: deferred convergence check
    # TODO: mixed precision
    # TODO: use linear combination kernels?
    # TODO: benchmark





if __name__ == "__main__":
    print "starting..."
    main_pkt()
