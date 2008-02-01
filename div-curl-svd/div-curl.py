from __future__ import division
import pylinear.array as num
import pylinear.operator as op
import pylinear.computation as comp




def main():
    from hedge.element import TetrahedralElement
    from hedge.mesh import make_ball_mesh, make_cylinder_mesh, make_box_mesh
    from hedge.mesh import make_ball_mesh, make_cylinder_mesh, make_box_mesh
    from hedge.visualization import \
            VtkVisualizer, \
            SiloVisualizer, \
            get_rank_partition
    from math import sqrt, pi, sin, cos
    from hedge.parallel import guess_parallelization_context
    from hedge.tools import dot, cross, BlockMatrix
    from hedge.flux import make_normal, FluxVectorPlaceholder
    from pytools.arithmetic_container import ArithmeticList

    pcon = guess_parallelization_context()

    if pcon.is_head_rank:
        mesh = make_box_mesh(max_volume=0.1, periodicity=(True, True, True))

    if pcon.is_head_rank:
        mesh_data = pcon.distribute_mesh(mesh)
    else:
        mesh_data = pcon.receive_mesh()

    order = 3
    discr = pcon.make_discretization(mesh_data, TetrahedralElement(order))
    vis = SiloVisualizer(discr, pcon)

    def l2_norm(field):
        return sqrt(dot(field, discr.mass_operator*field))

    normal = make_normal(discr.dimensions)
    w = FluxVectorPlaceholder(discr.dimensions)

    def premultiply_minv_in_block(bmat):
        minv = discr.inverse_mass_operator.matrix()
        return BlockMatrix(
                (i, j, minv*chunk)
                for i, j, chunk in bmat.chunks)

    nabla = discr.nabla
    m_inv = discr.inverse_mass_operator

    flen = len(discr.volume_zeros())
    print "#dof = %d" % flen

    curl_flux_op = discr.get_flux_operator(
                1/2*cross(normal, w.int-w.ext), 
                #-upwind_alpha/(2)*cross(normal, cross(normal, e.int-e.ext)),
                direct=False)

    if True:
        print "nabmat"
        nabla_mat = [ddx.matrix() for ddx in discr.nabla]
        curl_matrix = BlockMatrix([
            (0     , 2*flen,  nabla_mat[1]),
            (0     , 1*flen, -nabla_mat[2]),
            (1*flen, 0*flen,  nabla_mat[2]),
            (1*flen, 2*flen, -nabla_mat[0]),
            (2*flen, 1*flen,  nabla_mat[0]),
            (2*flen, 0*flen, -nabla_mat[1]),
            ])

        print "curl fluxmat"
        curl_flux_mat = premultiply_minv_in_block(curl_flux_op.matrix_inner())

        div_matrix = BlockMatrix([
            (0, i*flen, nabla_mat[i])
            for i in range(discr.dimensions)])

        print "div fluxmat"
        div_flux_mat = premultiply_minv_in_block(discr.get_flux_operator(
                dot(w.avg, normal),
                direct=False).matrix_inner())

        div = div_matrix - div_flux_mat
        curl = curl_matrix - curl_flux_mat
        div_t = div.T
        curl_t = curl.T

    def chop(vec):
        components = int(len(vec) / flen)
        return ArithmeticList(
                vec[i*flen:(i+1)*flen].real for i in range(components))

    def unchop(vec):
        return num.vstack(vec)

    class DivCurl2Operator(op.Operator(num.Float64)):
        def size1(self):
            return flen*3

        def size2(self):
            return flen*3

        def apply(self, before, after):
            after[:] = curl_t*(div_t*(div*(curl*before)))

        def is_ata(self):
            return False

        def apply_only_a(self, x):
            return div*(curl*x)

    class CurlOperator1(op.Operator(num.Float64)):
        def size1(self):
            return flen*3

        def size2(self):
            return flen*3

        def apply(self, before, after):
            after[:] = (curl*before)

        @property
        def is_ata(self):
            return False

    class CurlOperator2(op.Operator(num.Float64)):
        def size1(self):
            return flen*3

        def size2(self):
            return flen*3

        def apply(self, before, after):
            vec = chop(before)
            after[:] = unchop(cross(nabla, vec) - m_inv*(curl_flux_op*vec))

        @property
        def is_ata(self):
            return False

    operator = DivCurl2Operator()
    #operator = CurlOperator2()

    print "finding eigenvalues"
    results = comp.operator_eigenvectors(operator, 50)

    def add_vis(vis, visf, vec, name="ev", write_coarse_mesh=False):
        if len(vec) > flen:
            components = int(len(vector) / flen)
            vis.add_data(visf, vectors=[
                (name, chop(vec)),
                ],
                write_coarse_mesh=write_coarse_mesh)
        else:
            vis.add_data(visf, scalars=[
                (name, vec),
                ])

    if operator.is_ata:
        for i, (value,vector) in enumerate(results):
            assert abs(value.imag) < 1e-14
            assert comp.norm_2(vector.imaginary) < 1e-12
            sigma = sqrt(abs(value))
            print i, sigma
            vector = vector.real
            visf = vis.make_file("eigenvec-%04d" % i)
            add_vis(vis, visf, vector, "v", True)
            add_vis(vis, visf, 1/sigma*operator.apply_only_a(vector), "u")
            add_vis(vis, visf, curl*vector, "curl_v")
            visf.close()
        pass
    else:
        for i, (value,vector) in enumerate(results):
            print i, value
            visf = vis.make_file("eigenvec-%04d" % i)
            add_vis(vis, visf, vector.real)
            visf.close()




if __name__ == "__main__":
    main()

