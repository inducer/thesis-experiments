from __future__ import division
import sqlite3 as sqlite
import numpy
from plot_tools import auto_xy_reshape, unwrap_list
from matplotlib.pyplot import (contour, clabel, show, plot, subplot,
        text, rc, pcolormesh, colorbar, gray, title, gcf)
#from frame import FrameAxes


def get_data(db_conn, mat_type, angle, method, substep_count):
    qry = db_conn.execute(
            "select ratio, offset, dt from data"
            " where method=? and angle=?"
            " and mat_type=? and substep_count=?"
            " order by ratio, offset",
            (method, angle, mat_type, substep_count))
    ratio, offset, dt = auto_xy_reshape(qry)

    ratio = numpy.array(ratio)
    offset = numpy.array(offset)

    x = numpy.cos(offset)*ratio[:,numpy.newaxis]
    y = numpy.sin(offset)*ratio[:,numpy.newaxis]

    return ratio, offset, x, y, dt

def make_stabplot(db_conn, mat_type, angle, method, substep_count):

    ratio, offset, x, y, dt = get_data(
            db_conn, mat_type, angle, method, substep_count)

    rc("font", size=20)
    #gray()
    gcf().subplots_adjust(top=0.85)

    ax = subplot(111)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    from numpy import sin, cos, pi
    plot(
            ratio[-1]*cos(offset),
            ratio[-1]*sin(offset), 'k')
    plot(
            ratio[0]*cos(offset),
            ratio[0]*sin(offset), 'k')
    plot(ratio*cos(offset[-1]), ratio*sin(offset[-1]), 'k')
    plot(ratio*cos(offset[0]), ratio*sin(offset[0]), 'k')

    text(0, 1.051*ratio[-1], r"$\pi/2$")
    text(0, -1.051*ratio[-1], r"-$\pi/2$",
            va="top")
    text(-1.05*ratio[-1], 0, r" $\pi$",
            ha="right", va="center")

    text(ratio[-1]*1.05*cos(3*pi/4), ratio[-1]*1.05*sin(3*pi/4),
            "Angle between eigenvectors",
            rotation=45, va="center", ha="center")
    text(ratio[len(ratio)//2], 0, "Ratio of eigenvalues",
            va="center", ha="center")
    text(0, 0, "%.1f" % ratio[0],
            va="center", ha="center")
    text(ratio[-1], 0, "%.1f" % ratio[-1],
            va="center", ha="center")

    pcolormesh(x, y, dt)
    cb = colorbar()
    cb.ax.set_xlabel(r"stable $H$")

    cs = contour(x, y, dt, 20, colors="k")
    #clabel(cs, inline_spacing=2)

    text(0, 1.2, "%s MRAB(%dx) %.3f $\pi$"
            % (method, substep_count, angle/pi),
            fontsize=25, ha="center")





if __name__ == "__main__":
    db_conn = sqlite.connect("output.dat", timeout=30)

    all_angles = unwrap_list(
            db_conn.execute("select distinct angle from data"))
    all_methods = unwrap_list(
            db_conn.execute("select distinct method from data"))
    print all_methods
    #all_mat_types = unwrap_list(
            #db_conn.execute("select distinct mat_type from data"))
    #print all_mat_types
    #all_substep_counts = unwrap_list(
            #db_conn.execute("select distinct substep_count from data"))

    make_stabplot(db_conn,
            "OscillationMatrixFactory",
            all_angles[0],
            "Fsr", 2)

    show()
