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




def main():
    from optparse import OptionParser

    parser = OptionParser(
            usage="%prog [options] MATRIX-MARKET-FILE")
    parser.add_option("-s", "--is-symmetric", action="store_true",
            help="Specify that the input matrix is already symmetric")
    options, args = parser.parse_args()

    from scipy.io import mmread
    print "reading"
    mat = mmread(args[0])
    print "done"

    from packeted import PacketedSpMV
    spmv = PacketedSpMV(mat, options.is_symmetric, numpy.float32)

    from hedge.tools import unit_vector
    #x = numpy.random.randn(spmv.shape[0]).astype(spmv.dtype)
    x = unit_vector(spmv.shape[0], 1).astype(spmv.dtype)

    x_gpu = spmv.unpermute(gpuarray.to_gpu(x))
    y_gpu = spmv(x_gpu)
    coo_contrib = spmv.remaining_coo*x_gpu.get()
    y = spmv.permute(y_gpu + gpuarray.to_gpu(coo_contrib)).get()

    y2 = mat*x

    print la.norm(y-y2)/la.norm(y)




if __name__ == "__main__":
    main()
