import pycublas
import numpy
import numpy.linalg as la
from hedge.element import TetrahedralElement as Tet 

do_align = False
el_count = 20000

from pytools import Table

tbl = Table()

tbl.add_row(("Order", "Vol.nodes", "Lift nodes", "GFlops/s"))
for order in range(1, 10):
    t = Tet(order)
    vol_nodes = t.node_count()
    lift_nodes = t.face_node_count()*t.face_count()

    lift_shape = vol_nodes, lift_nodes
    vec_shape = lift_nodes, el_count
    lift_mat = numpy.mat(
            numpy.array(numpy.random.rand(*lift_shape), numpy.float32, order="F"))
    face_vec = numpy.mat(
            numpy.array(numpy.random.rand(*vec_shape), numpy.float32, order="F"))

    lift_mat_gpu = pycublas.CUBLASMatrix(lift_mat, mem_align=do_align)
    face_vec_gpu = pycublas.CUBLASMatrix(face_vec, mem_align=do_align)

    res_gpu = (lift_mat_gpu*face_vec_gpu).np_matrix()
    res = lift_mat*face_vec

    err = la.norm(res_gpu-res)/la.norm(res)
    assert err < 3e-7

    count = 10

    from cudart import cudaEvent_t,cudaEventCreate, cudaEventDestroy, \
            cudaEventRecord, cudaEventSynchronize, cudaEventElapsedTime
    evt_start = cudaEvent_t()
    evt_stop = cudaEvent_t()
    cudaEventCreate(evt_start)
    cudaEventCreate(evt_stop)
    cudaEventRecord(evt_start, 0)

    for i in range(count):
        lift_mat_gpu*face_vec_gpu

    cudaEventRecord(evt_stop, 0)
    cudaEventSynchronize(evt_stop)

    from ctypes import c_float
    tm = c_float()

    cudaEventElapsedTime(tm, evt_start, evt_stop)

    cudaEventDestroy(evt_start)
    cudaEventDestroy(evt_stop)

    matmul_time = tm.value*1e-3/count
    flops = 2*vol_nodes*el_count*lift_nodes

    tbl.add_row((order, vol_nodes, lift_nodes, flops/matmul_time/1e9))

print "%d elements\n" % el_count
print tbl
