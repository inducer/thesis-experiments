#! /usr/bin/env python
from __future__ import division
import numpy
import numpy.linalg as la
from matplotlib.pyplot import plot, ion, draw, clf, ylim, legend, show, savefig




a = -1
b = 1


def find_jump_errors(degree, do_plot, thresh=0.1):
    from hedge.quadrature import legendre_gauss_lobatto_points
    points = legendre_gauss_lobatto_points(degree)
    #points = numpy.linspace(a, b, degree+1)

    def f(x):
        if x < 0:
            return 0
        else:
            return 1

    from hedge.polynomial import monomial_vdm
    coeffs = la.solve(monomial_vdm(points), numpy.array(
        [f(x) for x in points]))

    from pymbolic.polynomial import Polynomial, differentiate
    from pymbolic import var, evaluate_kw
    u = Polynomial(var("x"), list(enumerate(coeffs)))

    vis_points = numpy.linspace(a, b, 300)

    true_soln = numpy.array([f(x) for x in vis_points])

    for i in range(1000):
        ux = differentiate(u)
        uxx = differentiate(ux)

        u_values = numpy.array([evaluate_kw(u, x=vp) for vp in vis_points])
        ux_values = numpy.array([evaluate_kw(ux, x=vp) for vp in vis_points])

        ok_indices_left = numpy.nonzero(numpy.abs(ux_values) < thresh)[0]
        ok_indices_right = numpy.nonzero(numpy.abs(ux_values) < thresh)[0]

        have_ok = len(ok_indices_left) and len(ok_indices_right)
        if have_ok:
            ok_start = ok_indices_left[0]
            ok_end = ok_indices_right[-1]

        if do_plot:
            clf()
            plot(vis_points, u_values, label="$u$")
            plot(vis_points, ux_values, label="$u_x$")
            if have_ok:
                plot(
                        [vis_points[ok_start], vis_points[ok_end]],
                        [u_values[ok_start], u_values[ok_end]],
                        "vr")

        if have_ok:
            nojump_bools = ux_values[ok_start:ok_end] < -thresh
            if do_plot:
                plot(
                        vis_points[ok_start:ok_end][nojump_bools], 
                        ux_values[ok_start:ok_end][nojump_bools], "or")

            if (ux_values[ok_start:ok_end] > -thresh).all():
                l1_err = (numpy.sum(numpy.abs(u_values-true_soln)[ok_start:ok_end])
                        /(ok_end-ok_start)) * (b-a)
                l2_err = numpy.sqrt((numpy.sum(((u_values-true_soln)**2)[ok_start:ok_end])
                        /(ok_end-ok_start)) * (b-a))
                return l1_err, l2_err

        if do_plot:
            legend()
            ylim([-1, 2])
            #savefig("art-visc-%05d.png" % i)
            draw()

        u = u + 1e-4 * uxx


def main():
    ion()
    find_jump_errors(20, do_plot=True)
    return

    degrees = numpy.arange(13, 42)
    l1s = []
    l2s = []
    from hedge.tools.convergence import EOCRecorder
    eoc_rec_l1 = EOCRecorder()
    eoc_rec_l2 = EOCRecorder()

    for degree in degrees:
        l1, l2 = find_jump_errors(degree, do_plot=False)
        l1s.append(l1)
        l2s.append(l2)
        print degree, l1, l1*degree, l2, l2*degree

        eoc_rec_l1.add_data_point(degree, l1)
        eoc_rec_l2.add_data_point(degree, l2)

    print eoc_rec_l1.pretty_print(error_label="L1")
    print eoc_rec_l2.pretty_print(error_label="L2")

    l1s = numpy.array(l1s)
    l2s = numpy.array(l2s)
    plot(degrees[0::2], 1/l2s[0::2])
    plot(degrees[1::2], 1/l2s[1::2])
    show()



if __name__ == "__main__":
    main()
