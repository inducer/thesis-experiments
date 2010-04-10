from __future__ import division
import sqlite3 as sqlite
import numpy
from plot_tools import auto_xy_reshape, unwrap_list
from matplotlib.pyplot import (contour, clabel, show, plot, subplot,
        text, rc, pcolormesh, colorbar, gray, title, gcf,
        savefig, clf)
#from frame import FrameAxes

# -----------------------------------------------------------------------------
# manual clabel monkey patch
# http://www.mail-archive.com/matplotlib-users@lists.sourceforge.net/msg15284.html
import matplotlib.blocking_input as blocking_input
def mouse_event_stop(self, event ):
    blocking_input.BlockingInput.pop(self,-1)
    self.fig.canvas.stop_event_loop()
def add_click(self, event):
    self.button1(event)
def pop_click(self, event, index=-1):
    if self.inline:
        pass
    else:
        self.cs.pop_label()
        self.cs.ax.figure.canvas.draw()

blocking_input.BlockingMouseInput.mouse_event_stop = mouse_event_stop
blocking_input.BlockingMouseInput.add_click = add_click
blocking_input.BlockingMouseInput.pop_click = pop_click
# -----------------------------------------------------------------------------


def get_data(db_conn, mat_type, angle, method, substep_count):
    print mat_type
    qry = db_conn.execute(
            "select ratio, offset, dt from data"
            " where method=? and angle=?"
            " and mat_type=? and substep_count=?"
            " and offset <= ?+1e-10"
            " order by ratio, offset",
            (method, angle, mat_type+"MatrixFactory", substep_count,
                numpy.pi))
    ratio, offset, dt = auto_xy_reshape(qry)

    ratio = numpy.array(ratio)
    offset = numpy.array(offset)

    x = numpy.cos(offset)*ratio[:,numpy.newaxis]
    y = numpy.sin(offset)*ratio[:,numpy.newaxis]

    return ratio, offset, x, y, dt

def make_stabplot(db_conn, mat_type, angle, method, substep_count):

    ratio, offset, x, y, dt = get_data(
            db_conn, mat_type, angle, method, substep_count)

    font_size = 40
    rc("font", size=font_size)
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
    #text(0, -1.051*ratio[-1], r"-$\pi/2$", va="top")
    text(-1.05*ratio[-1], 0, r" $\pi$", ha="right", va="center")

    text(ratio[-1]*1.1*cos(3*pi/4), ratio[-1]*1.1*sin(3*pi/4),
            r"$\beta$",
            va="center", ha="center")
    #text(ratio[len(ratio)//2], 0, r"$\mu$",
            #va="center", ha="center")
    text(ratio[0], -0.1, "%.1f" % ratio[0], va="baseline", ha="center")
    text(ratio[-1], -0.1, "%.1f" % ratio[-1], va="baseline", ha="center")
    text((ratio[-1]+ratio[0])/2, -0.1, "$\mu$" % ratio[-1],
            va="baseline", ha="left")

    pcolormesh(x, y, dt)
    cb = colorbar()
    cb.ax.set_position([0.85, 0.1, 0.9, 0.85])
    cb.ax.set_xlabel(r"stable $H$", labelpad=15)

    if mat_type == "Decay": mat_eigval = "-1,-1"
    elif mat_type == "OscillationDecay": mat_eigval = "i,-1"
    elif mat_type == "DecayOscillation": mat_eigval = "-1,i"
    elif mat_type == "Oscillation": mat_eigval = "i,i"
    else: raise ValueError("matrix type not understood")

    text(0, 1.5, r"%s MRAB on $(\lambda_1, \lambda_2)=(%s)$ $k=%d$ $\alpha=%.3f \pi$"
            % (method, mat_eigval, substep_count, angle/pi), ha="center")

    cs = contour(x, y, dt, 20, colors="k")
    clabel(cs, fontsize=font_size*0.6, inline_spacing=0,
            manual=True)




if __name__ == "__main__":
    import sys
    db_conn = sqlite.connect(sys.argv[1], timeout=30)

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

    if True:
        for mat_type in [
                "Oscillation",
                "Decay",
                "OscillationDecay",
                "DecayOscillation",
                ]:
            clf()
            make_stabplot(db_conn,
                    mat_type,
                    all_angles[1],
                    "Fq", 2)

            savefig("contour-plots/stab-countour-%s.pdf" % mat_type.lower())

    clf()
    make_stabplot(db_conn, "Oscillation", all_angles[0], "Fq", 2)
    savefig("contour-plots/stab-countour-triangular.pdf")

    for substep_count in [2, 5]:
        clf()
        make_stabplot(db_conn, "Oscillation", all_angles[1], "Fq", substep_count)
        savefig("contour-plots/stab-countour-kscale-%d.pdf" % substep_count)

    for method in ["Fq", "Ssf"]:
        clf()
        make_stabplot(db_conn, "Oscillation", all_angles[1], method, 5)
        savefig("contour-plots/stab-countour-methdep-%s.pdf" % method)

