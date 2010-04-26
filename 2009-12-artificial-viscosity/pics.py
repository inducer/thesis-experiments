from __future__ import division
from matplotlib.pyplot import (
        plot, show, grid, xlabel, ylabel, rc, legend, title,
        savefig, xlim, ylim, clf, gcf, bar, xticks, yticks, gca)
from pylo import SiloFile, DB_READ
import numpy
import numpy.linalg as la
import sqlite3
from glob import glob

font_size = 20

def make_shock_wrinkles_pic():
    clf()
    db = SiloFile(
            "kept-data/euler-2010-03-04-141612/N7-K641-v0.400000-VertexwiseMaxSmoother/euler-003823.silo",
            create=False, mode=DB_READ)
    rho_n7 = db.get_curve("rho")
    db = SiloFile(
            "kept-data/euler-2010-03-04-141612/N5-K81-v0.400000-VertexwiseMaxSmoother/euler-000570.silo",
            create=False, mode=DB_READ)
    rho_n5 = db.get_curve("rho")

    plot(rho_n7.x, rho_n7.y, label="$N=7$ $K=641$")
    plot(rho_n5.x, rho_n5.y, "o-", markersize=4, label="$N=5$ $K=81$")
    xlabel("$x$")
    ylabel(r"$\rho$")
    legend(loc="best")
    grid()
    xlim([0.45,0.73])
    ylim([0.23,0.45])
    savefig("pics/shock-wrinkles-cd.pdf")
    xlim([0.63, 0.69])
    ylim([0.413, 0.430])
    savefig("pics/shock-wrinkles-big.pdf")


def make_shu_osher_pic():
    clf()
    db = SiloFile(
            sorted(glob("kept-data/shu-osher-n5-k80/euler*silo"))[-1],
            create=False, mode=DB_READ)
    rho = db.get_curve("rho")
    p = db.get_curve("p")

    title("Shock-Wave Interaction Problem\nwith $N=5$ and $K=80$")
    plot(rho.x, rho.y, label=r"$\rho$")
    plot(p.x, p.y, label=r"$p$")
    xlabel("$x$")
    ylabel(r"$\rho$, $p$")
    legend(loc="best")
    grid()
    gcf().subplots_adjust(top=0.85)
    gcf().subplots_adjust(left=0.15)
    savefig("pics/shu-osher-n5-k80.pdf")





def make_lax_pic():
    clf()
    db = SiloFile(
            glob("kept-data/lax-n5-k80/euler*silo")[-1],
            create=False, mode=DB_READ)
    rho = db.get_curve("rho")
    p = db.get_curve("p")

    t = title("Lax's Problem with $N=5$ and $K=80$")
    t.set_position([0.5, 1.05])
    plot(rho.x, rho.y, label=r"$\rho$")
    plot(p.x, p.y, label=r"$p$")
    xlim([-0.1,4.1])
    xlabel("$x$")
    ylabel(r"$\rho$, $p$")
    legend(loc="best")
    grid()
    gcf().subplots_adjust(left=0.15)
    savefig("pics/lax-n5-k80.pdf")




def make_sod_pic():
    clf()
    from glob import glob
    db = SiloFile(
            "kept-data/euler-2010-03-04-141612/N5-K81-v0.400000-VertexwiseMaxSmoother/euler-000570.silo",
            create=False, mode=DB_READ)

    rho = db.get_curve("rho")
    p = db.get_curve("p")
    rho_ex = db.get_curve("rho_exact")
    p_ex = db.get_curve("p_exact")

    plot(rho.x, rho.y, label=r"$\rho$")
    plot(p.x, p.y, label=r"$p$")
    plot(rho_ex.x, rho_ex.y, label=r"$\rho$ (exact, $L^2$ proj.)")
    plot(p_ex.x, p_ex.y, label=r"$p$ (exact, $L^2$ proj.)")

    t = title("Sod's Problem with $N=5$ and $K=80$")
    t.set_position([0.5, 1.05])
    xlabel("$x$")
    ylabel(r"$\rho$, $p$")
    legend(loc="best", prop=dict(size=18))
    xlim([-0.05,1.05])
    grid()
    gcf().subplots_adjust(left=0.15)
    savefig("pics/sod-n5-k80.pdf")




def make_skyline_pics():

    def make_plots(func, what, draw_baseline=False, order=10):
        from hedge.discretization.local import IntervalDiscretization
        el = IntervalDiscretization(order)
        nodes = numpy.array(el.unit_nodes()).reshape(-1)
        fine_x_nodes = numpy.linspace(nodes[0], nodes[-1], 100)

        values = numpy.array([func(el, node) for node in nodes])
        modes = la.solve(el.vandermonde(), values)

        mode_nums = numpy.arange(0, len(modes), dtype=numpy.float64)
        log_mode_nums = numpy.log10(mode_nums[1:])

        clf()
        plot(nodes, values, "o", label="Data")
        plot(fine_x_nodes, 
                [sum(mode*basis([x]) for mode, basis in zip(modes, el.basis_functions()))
                    for x in fine_x_nodes],
                "-", label=r"Interpolant")
        xlabel("$x$")
        ylabel("$q(x)$")
        legend(loc="best")
        gcf().subplots_adjust(left=0.2)
        savefig("pics/modeproc-%s-spatial.pdf" % what)

        log_modes = numpy.log10(modes**2)/2
        log_modes[modes==0] = -15

        norm_2 = numpy.sum(modes**2)

        baseline_decay = norm_2*mode_nums**(-el.order)
        bl_log_modes = numpy.log10(modes**2 + baseline_decay**2)/2

        width = 0.6

        def draw_estimate(log_modes, label=None, **kwargs):
            s, log10_c = numpy.polyfit(log_mode_nums, log_modes[1:], 1)
            c = 10**log10_c

            min_x, max_x = xlim()
            est_x = numpy.linspace(min_x, max_x, 30)
            est_values = numpy.log10(c * est_x**s)

            if "%" in label:
                label = label % -s

            result, = plot(est_x, est_values, label=label, **kwargs)

            # restore x limits
            xlim([min_x, max_x])

            return result

        def do_skyline(log_modes):
            skyline_log_modes = []
            log_skyline = []
            skyline_max = max(log_modes[-2:])
            for i, log_coeff in list(enumerate(log_modes))[::-1]:
                skyline_max = max(log_coeff, skyline_max)
                skyline_log_modes.append(skyline_max)
                log_skyline.append((i+width, skyline_max))
                log_skyline.append((i-width, skyline_max))

            skyline_x, skyline_y = zip(*log_skyline)
            return numpy.array(skyline_log_modes[::-1]), skyline_x, skyline_y

        skyline_bl_log_modes, skyline_x, skyline_y = do_skyline(bl_log_modes)
        skyline_no_bl_log_modes, _, _ = do_skyline(log_modes)


        clf()
        i = numpy.arange(len(modes), dtype=numpy.float64)
        #bar(i-width/2, skyline_no_bl_log_modes,
                    #width=width*0.5, label="SL",
                    #color="yellow", zorder=12)
        #bar(i, sky_log_modes,
                    #width=width*.5, label="BD",
                    #color="g", zorder=9, alpha=0.9)
        if draw_baseline:
            bar(i-0.1, numpy.log10(baseline_decay),
                        width=0.2, label=r"$||q_N||_2 \hat b_n$",
                        color="yellow", zorder=13)

        plot(skyline_x, skyline_y, "-",
                label="SL cutoff", zorder=20)
        bar(i-width/2, log_modes,
                    width=width, label=r"$\hat q_n$",
                    color="r", zorder=9)

        leg_1 = legend(
                loc="lower left", 
                prop={"size": 20})

        xlabel("Mode number $n$", va="bottom")
        ylabel(r"$\log_{10} |\hat q_n|$")
        grid()
        gca().get_xaxis().set_ticks_position("top")
        gca().get_xaxis().set_label_position("top")

        xlim(-width, len(modes)-1+width)
        ymin, ymax = ylim()

        leg_2_curves = [
                draw_estimate(log_modes, "Raw: $s=%.1f$",
                    color="r", zorder=15, dashes=(10,10)),
                ]
        skyline_fit_curve = draw_estimate(
                skyline_no_bl_log_modes, "SL: $s=%.2f$",
                color="g", zorder=15)
        leg_2_curves.append(skyline_fit_curve)

        baseline_fit_curve = draw_estimate(skyline_bl_log_modes, "BD+SL: $s=%.2f$", 
                    color="g", zorder=15)
        leg_2_curves.append(baseline_fit_curve)
        baseline_fit_curve.set_visible(draw_baseline)

        leg_2 = legend(leg_2_curves, 
                [crv.get_label() for crv in leg_2_curves],
                loc="lower right",
                prop={"size": 20})
        gca().add_artist(leg_1)

        leg_1.set_zorder(20)
        leg_2.set_zorder(20)

        if draw_baseline:
            ylim([ymin*1.5, 1])
        else:
            ylim([ymin*1.5, 0])

        gcf().subplots_adjust(left=0.15)
        savefig("pics/modeproc-%s-modes-pre.pdf" % what)

    from random import random, seed
    seed(30)
    make_plots(lambda el, x: el.basis_functions()[-1]([x]), "last-mode")

    make_plots(lambda el, x: numpy.cos(3+numpy.sin(1.3*x)), "exp-sin")
    make_plots(lambda el, x: numpy.sin(numpy.pi*x), "sin")
    make_plots(lambda el, x: 1 if x >= 0 else 0, "jump")
    make_plots(lambda el, x: 1 if x >= 0.9 else 0, "offset-jump", order=20)
    make_plots(lambda el, x: (x-0.8) if x >= 0.8 else 0, "offset-kink", order=20)
    make_plots(lambda el, x: (x-0.8)**2 if x >= 0.8 else 0, "offset-c1", order=20)
    make_plots(lambda el, x: x if x >= 0 else 0, "kink")
    make_plots(lambda el, x: x**2 if x >= 0 else 0, "c1")
    make_plots(lambda el, x: 1+(random()-0.5)*1e-3, "noisy-one",
            draw_baseline=True)




def make_adv_el_trans_pic():
    db = sqlite3.connect("kept-data/advection-N10-K20/advection.dat")
    t, sens = zip(*list(db.execute("select t_sim.value,max_sensor.value"
            " from t_sim inner join max_sensor using (step)"
            " order by step")))
    plot(t, sens)
    gca().spines["left"].set_position(("outward", 10))
    gca().spines["bottom"].set_position(("outward", 10))
    gca().spines["top"].set_visible(False)
    gca().spines["right"].set_visible(False)
    xlabel("$t$")
    ylabel(r"$||\nu||_{L^\infty}$")
    grid()
    rc("path", simplify=False)
    gcf().subplots_adjust(left=0.15)
    gcf().subplots_adjust(bottom=0.12)
    savefig("pics/advection-element-transitions.pdf")




def make_adv_dt_pic():
    clf()
    db = sqlite3.connect("kept-data/advection-N10-K20/advection.dat")
    step, dt = zip(*list(db.execute("select step,value from dt"
            " order by step")))
    plot(step, dt)
    gca().spines["left"].set_position(("outward", 10))
    gca().spines["bottom"].set_position(("outward", 10))
    gca().spines["top"].set_visible(False)
    gca().spines["right"].set_visible(False)
    xlabel("Step number")
    ylabel(r"$\Delta t$")
    grid()
    rc("path", simplify=False)
    gcf().subplots_adjust(left=0.18)
    gcf().subplots_adjust(bottom=0.15)
    savefig("pics/advection-delta-t.pdf")


def make_adv_smooth_pics(base_dir, out_pic, long_term=False, el_bdry=False):
    from os.path import join
    clf()
    db = sqlite3.connect(join(base_dir, "advection.dat"))
    step_to_time = dict(db.execute("select step, value from t_sim"))

    def get_par(name):
        from cPickle import loads
        from pytools import one
        return loads(str(one(one(
            db.execute("select value from constants where name=?", (name, ))))))

    import re
    import os.path
    fname_list = sorted(glob(join(base_dir, "fld*.silo")))
    if long_term:
        fname_list = fname_list[::14]
    else:
        fname_list = fname_list[:3]

    for i, fname in enumerate(fname_list):
        step = int(re.search("[0-9]+", os.path.basename(fname)).group(0))

        try:
            t = step_to_time[step]
        except KeyError:
            continue

        sil = SiloFile(fname, create=False, mode=DB_READ)
        u = sil.get_curve("u_dg")
        plot(u.x, u.y+0.1*i, "o-", markersize=3, label="$t=%0.2f$" % t)

    if el_bdry:
        xmin, xmax = xlim()
        el_cnt = get_par("element_count")
        el_starts = numpy.linspace(xmin, xmax, el_cnt+1)

        from matplotlib.ticker import FixedLocator
        gca().xaxis.set_minor_locator(FixedLocator(el_starts))

        grid(which="minor")

    yticks([])
    xlabel("$x$")
    ylabel("$u(x)$")
    legend(prop=dict(size=22), loc="best")
    gcf().subplots_adjust(left=0.08)

    savefig(out_pic)










if __name__ == "__main__":
    make_skyline_pics()
    import sys; sys.exit()

    rc("font", size=22)
    make_adv_el_trans_pic()
    make_adv_dt_pic()
    make_adv_smooth_pics("kept-data/advection-N10-K20", 
            "pics/advection-with-visc.pdf", el_bdry=True)
    make_adv_smooth_pics("kept-data/advection-N10-K20-no-visc", 
            "pics/advection-without-visc.pdf", el_bdry=True)
    make_adv_smooth_pics("kept-data/advection-N10-K20", 
            "pics/advection-long-term.pdf", 
            long_term=True, el_bdry=True)

    rc("font", size=font_size)
    make_shock_wrinkles_pic()
    make_shu_osher_pic()
    make_lax_pic()
    make_sod_pic()
