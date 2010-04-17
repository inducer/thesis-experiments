from smoother import *




def make_mesh():
    if True:
        from hedge.mesh.generator import make_disk_mesh
        return make_disk_mesh(r=0.5, max_area=1e-2)
    elif False:
        from hedge.mesh.generator import make_regular_rect_mesh
        return make_regular_rect_mesh()
    else:
        from hedge.mesh.generator import make_uniform_1d_mesh
        return make_uniform_1d_mesh(-3, 3, 20, periodic=True)




def main():
    from hedge.backends import guess_run_context
    rcon = guess_run_context()

    mesh = make_mesh()

    el_values = {}

    from random import uniform, seed

    seed(1)

    def bumpy_f(x, el):
        try:
            return el_values[el.id]
        except KeyError:
            if False:
                x = uniform(0, 1)
                if x < 0.15:
                    result = 1
                else:
                    result = 0
            else:
                result = uniform(0, 1)
                if uniform(0,1) > 0.05:
                    result = 0
            el_values[el.id] = result
            return result

    discr = rcon.make_discretization(mesh, order=4)

    bumpy = discr.interpolate_volume_function(bumpy_f)

    if False:
        smoother = TemplatedSmoother(discr)
        smoothed = smoother(bumpy)
    elif False:
        p1_discr = rcon.make_discretization(discr.mesh, order=1)

        from hedge.discretization import Projector
        down_proj = Projector(discr, p1_discr)
        up_proj = Projector(p1_discr, discr)

        from hedge.mesh import TAG_NONE, TAG_ALL
        from hedge.models.diffusion import DiffusionOperator
        from hedge.tools.second_order import IPDGSecondDerivative
        op = DiffusionOperator(p1_discr.dimensions,
                dirichlet_tag=TAG_NONE, neumann_tag=TAG_ALL,
                scheme=IPDGSecondDerivative(10000))
        bound_op = op.bind(p1_discr)

        p1_accu = down_proj(bumpy)
        for i in range(10):
            laplace_result = bound_op(0, p1_accu)
            p1_accu += 0.3*la.norm(p1_accu)/la.norm(laplace_result)*laplace_result
        smoothed = up_proj(p1_accu)
    elif True:
        smoother = VertexwiseMaxSmoother().bind(discr)
        smoothed = smoother(bumpy)
    else:
        smoother = TriBlobSmoother().bind(discr)
        smoothed = smoother(bumpy)

    from hedge.tools.convergence import relative_error
    print relative_error(discr.norm(bumpy-smoothed), discr.norm(bumpy))

    #vis_discr = discr
    vis_discr = rcon.make_discretization(mesh, order=30)

    from hedge.discretization import Projector
    vis_proj = Projector(discr, vis_discr)

    from hedge.visualization import VtkVisualizer
    vis = VtkVisualizer(vis_discr, rcon)
    visf = vis.make_file("bumpy")
    vis.add_data(visf, [ 
        ("bumpy", vis_proj(bumpy)),
        ("smoothed", vis_proj(smoothed)),
        ])
    visf.close()



def pstudy_triblob():
    from hedge.backends import guess_run_context
    rcon = guess_run_context()

    mesh = make_mesh()

    discr = rcon.make_discretization(mesh, order=4)

    #vis_discr = discr
    vis_discr = rcon.make_discretization(mesh, order=30)

    from hedge.discretization import Projector
    vis_proj = Projector(discr, vis_discr)

    el_values = {}

    from random import uniform

    def bumpy_f(x, el):
        try:
            return el_values[el.id]
        except KeyError:
            x = uniform(0, 1)
            if x < 0.2:
                result = 1
            else:
                result = 0
            el_values[el.id] = result
            return result

    bumpy = discr.interpolate_volume_function(bumpy_f)
    bumpy_norm = discr.norm(bumpy, 1)

    vis_data = [("bumpy", vis_proj(bumpy))]

    for ramp in [ramp_sin, ramp_bruno]:
        ramp_name = ramp.__name__[5:]

        outf = open(ramp_name+".dat", "w")

        for exponent in numpy.arange(0.5, 1.5, 0.05):
            for scaling in numpy.arange(0.8, 3, 0.1):
                smoother = TriBlobSmoother(discr, ramp, exponent, scaling)
                smoothed = smoother(bumpy)

                name = "%s_%.2f_%.2f" % (ramp_name, exponent, scaling)
                from hedge.tools.convergence import relative_error
                rel_err = relative_error(discr.norm(bumpy-smoothed, 1), bumpy_norm)
                print name, rel_err
                outf.write("%g %g %g\n" % (exponent, scaling, rel_err))

                vis_data.append((name.replace(".", "_"), vis_proj(smoothed)))
            outf.write("\n")
            outf.flush()

    from hedge.visualization import VtkVisualizer
    vis = VtkVisualizer(vis_discr, rcon)
    visf = vis.make_file("bumpy")
    vis.add_data(visf, vis_data)
    visf.close()




if __name__ == "__main__":
    main()
