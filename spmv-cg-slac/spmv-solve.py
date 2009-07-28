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




STREAM_POOL = []




def get_stream():
    if STREAM_POOL:
        return STREAM_POOL.pop()
    else:
        return drv.Stream()
        




class AsyncInnerProduct:
    def __init__(self, a, b, pagelocked_allocator):
        self.gpu_result = gpuarray.dot(a, b)
        self.gpu_finished_evt = drv.Event()
        self.gpu_finished_evt.record()
        self.gpu_finished = False

        self.pagelocked_allocator = pagelocked_allocator

    def get_host_result(self):
        if not self.gpu_finished:
            if self.gpu_finished_evt.query():
                self.gpu_finished = True
                self.copy_stream = get_stream()
                self.host_dest = self.pagelocked_allocator(
                        self.gpu_result.shape, self.gpu_result.dtype,
                        self.copy_stream)
                drv.memcpy_dtoh_async(self.host_dest, 
                        self.gpu_result.gpudata,
                        self.copy_stream)
                self.copy_finished_evt = drv.Event()
                self.copy_finished_evt.record()
        else:
            if self.copy_finished_evt.query():
                STREAM_POOL.append(self.copy_stream)
                return self.host_dest




class GPUCGStateContainer:
    def __init__(self, operator, precon=None, pagelocked_allocator=None):
        if precon is None:
            from hedge.iterative import IdentityOperator
            precon = IdentityOperator(operator.dtype, operator.shape[0])

        self.operator = operator
        self.precon = precon

        self.pagelocked_allocator = pagelocked_allocator

    @memoize_method
    def make_lc2_kernel(self, dtype, a_is_gpu, b_is_gpu):
        from pycuda.elementwise import get_linear_combination_kernel
        return get_linear_combination_kernel((
                (a_is_gpu, dtype, dtype),
                (b_is_gpu, dtype, dtype)
                ), dtype)

    def lc2(self, a, x, b, y, out=None):
        if out is None:
            out = gpuarray.empty(x.shape, dtype=x.dtype,
                    allocator=x.allocator)

        assert x.dtype == y.dtype == out.dtype
        a_is_gpu = isinstance(a, gpuarray.GPUArray)
        b_is_gpu = isinstance(b, gpuarray.GPUArray)
        assert x.shape == y.shape == out.shape

        kernel, texrefs = self.make_lc2_kernel(
                x.dtype, a_is_gpu, b_is_gpu)

        texrefs = texrefs[:]

        args = []

        if a_is_gpu:
            assert a.dtype == x.dtype
            assert a.shape == ()
            a.bind_to_texref_ext(texrefs.pop(0), allow_double_hack=True)
        else:
            args.append(a)
        args.append(x.gpudata)

        if b_is_gpu:
            assert b.dtype == y.dtype
            assert b.shape == ()
            b.bind_to_texref_ext(texrefs.pop(0), allow_double_hack=True)
        else:
            args.append(b)
        args.append(y.gpudata)
        args.append(out.gpudata)
        args.append(x.mem_size)

        kernel.set_block_shape(*x._block)
        kernel.prepared_call(x._grid, *args)

        return out

    def reset(self, rhs, x=None):
        self.rhs = rhs

        if x is None:
            x = numpy.zeros((self.operator.shape[0],))
        self.x = x

        self.residual = rhs - self.operator(x)

        self.d = self.precon(self.residual)

        # grows at the end
        delta = AsyncInnerProduct(self.residual, self.d,
                self.pagelocked_allocator)
        self.real_delta_queue = [delta]
        self.delta = delta.gpu_result

    def one_iteration(self, compute_real_residual=False):
        # typed up from J.R. Shewchuk,
        # An Introduction to the Conjugate Gradient Method
        # Without the Agonizing Pain, Edition 1 1/4 [8/1994]
        # Appendix B3

        q = self.operator(self.d)
        myip = gpuarray.dot(self.d, q)
        alpha = self.delta / myip

        self.lc2(1, self.x, alpha, self.d, out=self.x)

        if compute_real_residual:
            self.residual = self.lc2(
                    1, self.rhs, -1, self.operator(self.x))
        else:
            self.lc2(1, self.residual, -alpha, q, out=self.residual)

        s = self.precon(self.residual)
        delta_old = self.delta
        delta = AsyncInnerProduct(self.residual, s,
                self.pagelocked_allocator)
        self.delta = delta.gpu_result 
        beta = self.delta / delta_old;

        self.lc2(1, s, beta, self.d, out=self.d)

        if compute_real_residual:
            self.real_delta_queue.append(delta)

    def run(self, max_iterations=None, tol=1e-7, debug_callback=None):
        if max_iterations is None:
            max_iterations = 10 * self.operator.shape[0]

        iterations = 0
        delta_0 = None
        while iterations < max_iterations:
            compute_real_residual = \
                    iterations % 50 == 0

            self.one_iteration(
                    compute_real_residual=compute_real_residual)

            if debug_callback is not None:
                if compute_real_residual:
                    what = "it+residual"
                else:
                    what = "it"

                debug_callback(what, iterations, self.x, 
                        self.residual, self.d, self.delta)

            # do often enough to allow AsyncInnerProduct
            # to progress through (polled) event chain
            rdq = self.real_delta_queue
            if iterations % 20 == 0:
                if delta_0 is None:
                    delta_0 = rdq[0].get_host_result()
                    if delta_0 is not None:
                        rdq.pop(0)

                if delta_0 is not None:
                    i = 0
                    while i < len(rdq):
                        delta = rdq[i].get_host_result()
                        if delta is not None:
                            print abs(delta) / abs(delta_0)
                            if abs(delta) < tol*tol * abs(delta_0):
                                if debug_callback is not None:
                                    debug_callback("end", iterations, 
                                            self.x, self.residual, 
                                            self.d, self.delta)
                                return self.x
                            rdq.pop(i)
                        else:
                            i += 1

            iterations += 1

        from hedge.iterative import ConvergenceError
        raise ConvergenceError("cg failed to converge")




def solve_pkt_with_cg(pkt_spmv, b, x=None, tol=1e-7, max_iterations=None,
        debug=False, pagelocked_allocator=None):
    if x is None:
        x = gpuarray.zeros(pkt_spmv.shape[0], dtype=pkt_spmv.dtype,
                allocator=b.allocator)
    else:
        x = pkt_spmv.permute(x)

    if False:
        def inner(a, b):
            return gpuarray.dot(a,b).get()

        from hedge.iterative import CGStateContainer
        cg = CGStateContainer(pkt_spmv, dot=inner)
    else:
        if pagelocked_allocator is None:
            pagelocked_allocator = drv.pagelocked_empty

        cg = GPUCGStateContainer(pkt_spmv, 
                pagelocked_allocator=pagelocked_allocator)

    cg.reset(pkt_spmv.permute(b), x)

    it_count = [0]
    res_count = [0]
    def debug_callback(what, it_number, x, resid, d, delta):
        if what == "it":
            it_count[0] += 1
        elif what == "it+residual":
            res_count[0] += 1
            it_count[0] += 1

    result = cg.run(max_iterations, tol,
            debug_callback=debug_callback)

    return pkt_spmv.unpermute(result), it_count[0], res_count[0]




def main_cg():
    from optparse import OptionParser

    parser = OptionParser(
            usage="%prog [options] MATRIX-MARKET-FILE")
    parser.add_option("-s", "--is-symmetric", action="store_true",
            help="Specify that the input matrix is already symmetric")
    options, args = parser.parse_args()

    from scipy.io import mmread
    mat = mmread(args[0])

    print "building..."
    from packeted import PacketedSpMV
    spmv = PacketedSpMV(mat, options.is_symmetric, numpy.float32)
    rhs = numpy.random.rand(spmv.shape[0]).astype(spmv.dtype)

    from pycuda.tools import DeviceMemoryPool, PageLockedMemoryPool
    dev_pool = DeviceMemoryPool()
    pagelocked_pool = PageLockedMemoryPool()

    print "start solve"
    for i in range(4):
        start = drv.Event()
        stop = drv.Event()
        start.record()

        rhs_gpu = gpuarray.to_gpu(rhs, dev_pool.allocate)
        res_gpu, it_count, res_count = \
                solve_pkt_with_cg(spmv, rhs_gpu, 
                        tol=1e-7 if spmv.dtype == numpy.float64 else 5e-5,
                        pagelocked_allocator=pagelocked_pool.allocate)
        res = res_gpu.get()

        stop.record()
        stop.synchronize()

        elapsed = stop.time_since(start)*1e-3
        est_flops = (mat.nnz*2*(it_count+res_count)
            + mat.shape[0]*(2+2+2+2+2)*it_count)
                    
        print "residual norm: %g" % (la.norm(mat*res - rhs)/la.norm(rhs))
        print ("size: %d, elapsed: %g s, %d it, %d residual, it/second: %g, "
                "%g gflops/s" % (
                    mat.shape[0],
                    elapsed, it_count, res_count, it_count/elapsed,
                    est_flops/elapsed/1e9))

    # TODO: mixed precision
    # TODO: benchmark
    pagelocked_pool.stop_holding()
    dev_pool.stop_holding()





if __name__ == "__main__":
    print "starting..."
    main_cg()

    STREAM_POOL[:] = []
