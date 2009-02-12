import numpy
import numpy.linalg as la
import pycuda.autoinit
from pytools import Record

class ParticleInfoBlock(Record):
    pass

def make_pib(particle_count):
    MM = 1e-3
    if False:
        from pyrticle.geometry import make_cylinder_with_fine_core
        mesh = make_cylinder_with_fine_core(
            r=25*MM, inner_r=2.5*MM,
            min_z=-50*MM, max_z=50*MM,
            max_volume_inner=10*MM**3,
            max_volume_outer=100*MM**3,
            radial_subdiv=10)
    else:
        from hedge.mesh import make_cylinder_mesh
        mesh = make_cylinder_mesh(radius=25*MM, height=50*MM,
                max_volume=300*MM**3)

    xdim = 3
    vdim = 3
    xdim_align = 4 
    vdim_align = 4 

    mesh_min, mesh_max = mesh.bounding_box()
    center = (mesh_min+mesh_max)*0.5

    x_particle = 0.03*numpy.random.randn(particle_count, xdim)
    x_particle[:,0] *= 0.01
    x_particle[:,1] *= 0.01

    for i in range(xdim):
        x_particle[:,i] += center[i]

    v_particle = numpy.zeros((particle_count, vdim))
    v_particle[:,0] = 1

    from hedge.backends.jit import Discretization
    discr = Discretization(mesh, order=4)

    return ParticleInfoBlock(
        mesh=mesh,
        discr=discr,
        xdim=xdim,
        vdim=vdim,
        x_particle=x_particle,
        v_particle=v_particle,
        charge=1,
        #radius=0.000556605732511,
        radius=50*MM,
        )

def main():
    #pcounts = [100*1000]
    pcounts = [1]


    def obj_array_norm(oa):
        return sum(la.norm(oa_i)**2 for oa_i in oa)**0.5

    from brute import time_brute
    from sift import time_sift
    for test_nr, particle_count in enumerate(pcounts):
        print "Making PIB"
        pib = make_pib(particle_count)
        print "done"

        vis_data = []

        computed_j_fields = []
        for alg, func in [
            ("brute", time_brute),
            ("sift", time_sift),
            ]:
            rate, j = func(pib)
            print alg, particle_count, rate
            computed_j_fields.append(j)
            vis_data.append(("%s_%d" % (alg, particle_count), j))

        ref_j = computed_j_fields[0]
        print "---------------------------------------"
        print "ref norm", obj_array_norm(ref_j)
        for comp_j in computed_j_fields[1:]:
            from hedge.tools import relative_error
            print "rel error", relative_error(
                    obj_array_norm(ref_j - comp_j),
                    obj_array_norm(ref_j))

        from hedge.visualization import SiloVisualizer
        vis = SiloVisualizer(pib.discr)
        visf = vis.make_file("particles")
        vis.add_data(visf, vis_data)
        visf.close()

if __name__ == "__main__":
    main()
