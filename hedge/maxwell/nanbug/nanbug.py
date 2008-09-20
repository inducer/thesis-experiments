import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import numpy
import numpy.linalg as la




def test():
    from pycuda.tools import fill_shared_with
    from cPickle import load

    fill_shared_with(float('nan'))

    gathermod = cuda.SourceModule(open("flux.cu").read())
    gather = gathermod.get_function("apply_flux")

    fdata = load(open("fdata.dt"))
    print len(fdata)
    fdata = cuda.to_device(fdata)

    gather.prepare("PP", block=(15, 15, 1))

    liftmod = cuda.SourceModule(open("kernel.cu").read())
    lift = liftmod.get_function("apply_lift_mat")

    fof = gpuarray.to_gpu(load(open("fof.dt")))
    ij = gpuarray.to_gpu(load(open("ij.dt")))
    lm = cuda.to_device(load(open("ij.dt")))

    fluxes_on_faces_texref = liftmod.get_texref("fluxes_on_faces_tex")
    texrefs = [fluxes_on_faces_texref]

    inverse_jacobians_texref = liftmod.get_texref("inverse_jacobians_tex")
    ij.bind_to_texref(inverse_jacobians_texref)
    texrefs.append(inverse_jacobians_texref)

    lift.prepare("PPP", block=(16, 31, 1), texrefs=texrefs)

    for i in range(200):
        print "YO"
        debugbuf = gpuarray.zeros((1024,), dtype=numpy.float32)
        gather.prepared_timed_call((3435,1), 
                debugbuf.gpudata, fdata)

        fof.bind_to_texref(fluxes_on_faces_texref)
        debugbuf = gpuarray.zeros((1024,), dtype=numpy.float32)
        flux = gpuarray.empty((999936,), dtype=numpy.float32)
        lift.prepared_timed_call(
                (9,28), flux.gpudata, 
                lm, debugbuf.gpudata)

        copied_debugbuf = debugbuf.get()[:144*7].reshape((144,7))
        numpy.set_printoptions(linewidth=100)
        copied_debugbuf.shape = (144,7)
        numpy.set_printoptions(threshold=3000)

        assert not (copied_debugbuf[:, 0] > 26800).any()

        #print copied_debugbuf
        #raw_input()

        #cuda.Context.synchronize()
        #print "NANCHECK"
        #copied_flux = discr.volume_from_gpu(flux)
        #contains_nans = numpy.isnan(copied_flux).any()

        #assert not contains_nans, "Resulting flux contains NaNs."


if __name__ == "__main__":
    test()


