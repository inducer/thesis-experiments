from __future__ import division, with_statement


plot_loc_errors = False
plot_eoc = False
plot_xt_eoc = False
plot_xt_eoc_mpl = False
plot_xt_eoc_mpl_pcolor = True

from pylo import SiloFile, DB_READ, DB_CLOBBER, DB_NODECENT
from glob import glob
from os.path import join, basename
import re
import numpy
import numpy.linalg as la
from pytools import memoize_method
from matplotlib.pyplot import (
    clf, semilogy, legend, legend, title, savefig,
    plot, show, grid, rc, xlabel, ylabel, pcolor,
    colorbar, imshow)





def find_step_no(filename):
    bn = basename(filename)
    mtch = re.search(r".*-([0-9]+)\.silo$", filename)
    return int(mtch.group(1))



def get_point_evaluator(from_discr, to_discr, to_node_idx):
    # makes 1D uniform assumptions
    from_eg, = from_discr.element_groups
    to_eg, = to_discr.element_groups

    from_ldis = from_eg.local_discretization
    to_ldis = to_eg.local_discretization

    to_el_idx = to_node_idx//to_ldis.node_count()
    to_el = to_eg.members[to_el_idx]
    to_el_rng = to_eg.ranges[to_el_idx]
    assert to_el_rng.start <= to_node_idx < to_el_rng.stop

    pt = to_discr.nodes[to_node_idx]

    el_ratio = len(to_discr.mesh.elements) / len(from_discr.mesh.elements)
    assert el_ratio == int(el_ratio)
    el_ratio = int(el_ratio)

    from_el_idx = to_el_idx // el_ratio
    from_el = from_eg.members[from_el_idx]
    assert from_el.contains_point(pt, 1e-10)
    from_el_rng = from_eg.ranges[from_el_idx]

    from hedge.discretization import _PointEvaluator

    basis_values = numpy.array([
            phi(from_el.inverse_map(pt))
            for phi in from_ldis.basis_functions()])
    vdm_t = from_ldis.vandermonde().T
    return _PointEvaluator(
            discr=from_discr,
            el_range=from_el_rng,
            interp_coeff=la.solve(vdm_t, basis_values))




class RunInfo:
    def __init__(self, path, conv_var, suffix):
        self.path = path

        self.vis_steps = sorted(
                (find_step_no(filename), filename)
                    for filename in glob(join(path, "*-[0-9]*.silo")))

        log_filename, = glob(join(path, "*.dat"))

        from pytools.log import LogManager
        logmgr = LogManager(log_filename, "r")
        (steps, _, _), (times, _, _) = logmgr.get_plot_data("step", "t_sim")
        self.element_count = logmgr.constants["element_count"]
        self.dg_order = logmgr.constants["dg_order"]
        logmgr.close()

        self.steps = numpy.array(steps, dtype=numpy.float64)
        self.times = numpy.array(times, dtype=numpy.float64)

        self.discr, self.a, self.b = self.get_discr(conv_var, suffix)

        self.h = (self.b-self.a)/self.element_count

    def find_step(self, t):
        return numpy.interp(t, self.times, self.steps)

    def find_time(self, t):
        return numpy.interp(t, self.steps, self.times)

    def get_discr(self, conv_var, suffix):
        with SiloFile(self.vis_steps[0][1], create=False, mode=DB_READ) as db:
            crv = db.get_curve(conv_var+suffix)
            self.a = a = crv.x[0]
            self.b = b = crv.x[-1]

            from hedge.mesh.generator import make_uniform_1d_mesh
            mesh = make_uniform_1d_mesh(a, b, self.element_count)
            from hedge.backends.jit import Discretization
            discr = Discretization(mesh, order=self.dg_order)
            assert numpy.allclose(discr.nodes[:,0], crv.x)

            return discr, a, b

    @memoize_method
    def get_point_evaluator(self, to_discr, to_node_idx):
        return get_point_evaluator(self.discr, to_discr, to_node_idx)


def data_from_silo(fname, var_names, discr=None):
    data = []
    with SiloFile(fname, create=False, mode=DB_READ) as db:
        for var_name in var_names:
            crv = db.get_curve(var_name)
            if discr is not None:
                assert numpy.allclose(discr.nodes[:,0], crv.x)

            data.append(crv.y.copy())

    return data

def interpolate_to_ref_grid(ref_ri, ri, var):
    result = ref_ri.discr.volume_zeros()
    for i in xrange(len(ref_ri.discr.nodes)):
        intp = ri.get_point_evaluator(ref_ri.discr, i)
        result[i] = intp(var)

    return result







def make_spatial_eoc_plot(
        basedir,
        refsoln,
        conv_var,
        suffix="",
        outfile=None,
        what="",
        glob_pattern="*",
        plot_loc_errors = False,
        plot_eoc = False,
        plot_xt_eoc = False,
        plot_xt_eoc_mpl = False,
        plot_xt_eoc_mpl_pcolor = True):
    clf()

    ref_ri = RunInfo(join(basedir, refsoln), conv_var, suffix)

    run_infos = []
    for dirname in glob(join(basedir, glob_pattern)):
        ri = RunInfo(dirname, conv_var, suffix)
        if ri.element_count <= ref_ri.element_count:
            run_infos.append((ri.element_count, ri))
    run_infos.sort()

    h_values = numpy.array([ri.h for k, ri in run_infos])

    eoc_data = []
    vis_times = []
    #for vis_idx, (ref_step, ref_silo) in list(enumerate(ref_ri.vis_steps)):
    #for vis_idx, (ref_step, ref_silo) in list(enumerate(ref_ri.vis_steps))[1:]:
    for vis_idx, (ref_step, ref_silo) in list(enumerate(ref_ri.vis_steps))[:-1]:
        t = ref_ri.find_time(ref_step)
        print t

        ref_var, = data_from_silo(ref_silo, [conv_var+"_exact"])
        #ref_var, = data_from_silo(ref_silo, [conv_var+suffix])

        error_lists = numpy.zeros((len(ref_ri.discr), len(run_infos)))
        for ri_idx, (el_count, run_info) in enumerate(run_infos):
            print "el_cnt", el_count
            native_var, = data_from_silo(
                    run_info.vis_steps[vis_idx][1], 
                    [conv_var+suffix], 
                    run_info.discr)
            var = interpolate_to_ref_grid(ref_ri, run_info, native_var)
            error_lists[:,ri_idx] = numpy.abs(var-ref_var)

        eoc_values = ref_ri.discr.volume_zeros()
        for i, error_list in enumerate(error_lists):
            from hedge.tools.convergence import estimate_order_of_convergence
            el = numpy.maximum(1e-15, numpy.array(error_list))

            _, eoc_values[i] = estimate_order_of_convergence(h_values, el)

        eoc_values *= -1

        vis_times.append(t)
        eoc_data.append(eoc_values)

        if plot_loc_errors:
            clf()
            for (el_count, ri), err in zip(run_infos, error_lists.T):
                semilogy(ref_ri.discr.nodes[:,0], err, 
                        label="$K=%d$" % ri.element_count)
            #plot(ref_ri.discr.nodes[:,0], eoc_values)
            if vis_idx <= 3:
                legend()
            title("Error plots at $t=%f$" % t)
            savefig("loc-errors-%03d.png" % vis_idx)

        if plot_eoc:
            title("Local EOC at $t=%f$" % t)

            plot(ref_ri.discr.nodes[:,0], eoc_values)
            savefig("eoc-%03d.png" % vis_idx)
            show()

    if plot_xt_eoc:
        with SiloFile("spatial-eoc.silo", mode=DB_CLOBBER) as f:
            f.put_quadmesh("xtmesh", [
                ref_ri.discr.nodes[:,0].copy(),
                numpy.array(vis_times, dtype=numpy.float64)*10
                ])

            ary_data = numpy.asarray(eoc_data)
            f.put_quadvar1(conv_var+"_eoc", "xtmesh",
                    ary_data, ary_data.shape,
                    centering=DB_NODECENT)

    if plot_xt_eoc_mpl:
        clf()
        s = len(vis_times)
        for i in [0, int(s/2), s-1]:
            plot(ref_ri.discr.nodes[:,0], eoc_data[i],
                    label="$t=%.2f$" % vis_times[i])
        xlabel("$x$")
        ylabel("Experimental Order of Convergence")
        legend()
        grid()

        show()

    if plot_xt_eoc_mpl_pcolor:
        clf()
        x = ref_ri.discr.nodes[:,0]
        t = numpy.array(vis_times)
        eoc_data = numpy.array(eoc_data)

        #x = x[:, numpy.newaxis] * numpy.ones((len(x), len(t)))
        #t = t[numpy.newaxis, :] * numpy.ones((len(x), len(t)))
        #pcolor(x, t, eoc_data.T)
        imshow(eoc_data[::-1,:], extent=(x[0], x[-1], t[0], t[-1]),
                aspect="auto")
        xlabel("$x$")
        ylabel("$t$")
        title("Spatial EOC: %s" % what)

        colorbar()

        if outfile:
            savefig("%s-xt-eoc.pdf" % outfile)
        else:
            show()




if __name__ == "__main__":
    rc("font", size=20)
    if False:
        make_spatial_eoc_plot(
                basedir = "sod-kconv-2010-04-12-185516",
                refsoln = "N4-K640-v0.400000-VertexwiseMaxSmoother/",
                conv_var = "rho",
                outfile="spatial-eoc-pics/euler-sod",
                what="Euler Sod $N=4$")
        make_spatial_eoc_plot(
                basedir = "adv-kconv-2010-04-12-185521",
                refsoln = "N4-K640-v0.200000-VertexwiseMaxSmoother",
                conv_var = "u",
                suffix = "_dg",
                outfile="spatial-eoc-pics/advection",
                what="Advection Sines $N=4$")
    if False:
        make_spatial_eoc_plot(
                basedir = "wave-kconv-2010-04-17-145456",
                glob_pattern="*-v0*TwoJumpTestCase",
                refsoln = "N5-K640-v1.000000-VertexwiseMaxSmoother-TwoJumpTestCase",
                conv_var = "u",
                outfile="spatial-eoc-pics/wave-no-visc",
                what=r"Wave Heavisides $N=5$ $\nu=0$")
        make_spatial_eoc_plot(
                basedir = "wave-kconv-2010-04-17-145456",
                glob_pattern="*-v1*TwoJumpTestCase",
                refsoln = "N5-K640-v1.000000-VertexwiseMaxSmoother-TwoJumpTestCase",
                conv_var = "u",
                outfile="spatial-eoc-pics/wave",
                what="Wave Heavisides $N=5$")

    make_spatial_eoc_plot(
            basedir = "wave-kconv-2010-04-14-120341",
            glob_pattern="*-v0*-PureSineTestCase",
            refsoln = "N5-K640-v0.000000-VertexwiseMaxSmoother-PureSineTestCase",
            conv_var = "u",
            outfile="spatial-eoc-pics/wave-sin",
            what="Wave Sine $N=5$",
            plot_loc_errors = True,
            #plot_xt_eoc_mpl_pcolor = False,
            )
