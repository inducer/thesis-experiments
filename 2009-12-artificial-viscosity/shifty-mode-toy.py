import pygtk
pygtk.require('2.0')
import gtk

import numpy
import numpy.linalg as la

class InteractivePlot:
    def delete_event(self, widget, event, data=None):
        return False

    def destroy(self, widget, data=None):
        gtk.main_quit()

    def __init__(self, order, f=lambda x: 0):
        from hedge.discretization.local import IntervalDiscretization
        self.ldis = IntervalDiscretization(order)

        self.func = f

        self.window = gtk.Window(gtk.WINDOW_TOPLEVEL)

        self.window.connect("delete_event", self.delete_event)
        self.window.connect("destroy", self.destroy)
        self.window.set_border_width(10)
        self.vbox = gtk.VBox()

        self.shift_adj = gtk.Adjustment(0, -1, 1)
        self.shift_adj.connect(
                "value-changed", lambda adj, user_arg: self.update_plot(), None)

        hscale = gtk.HScale(self.shift_adj)
        hscale.set_digits(4)
        hscale.set_value_pos(gtk.POS_RIGHT)
        hscale.set_increments(0.01, 0.1)
        self.vbox.pack_start(hscale, expand=False)
        hscale.show()

        from matplotlib.backends.backend_gtk import FigureCanvasGTK
        from matplotlib.figure import Figure

        self.figure = Figure(figsize=(6,4), dpi=72)
        self.axis_spatial = self.figure.add_subplot(211)
        self.axis_modal = self.figure.add_subplot(212)

        self.canvas = FigureCanvasGTK(self.figure)
        self.vbox.pack_start(self.canvas)
        self.canvas.show()

        self.update_plot()

        self.vbox.show()
        self.window.add(self.vbox)
        self.window.show()

    def update_plot(self):
        shift = self.shift_adj.get_value()

        self.axis_spatial.cla()
        self.axis_spatial.set_xlim([-1.05, 1.05])
        self.axis_spatial.grid()

        nodal_coeffs = numpy.array([
            self.func(x[0]-shift) for x in self.ldis.unit_nodes()])
        modal_coeffs = la.solve(self.ldis.vandermonde(), nodal_coeffs)

        def f(x):
            return sum(coeff*bf([x]) 
                    for coeff, bf in zip(modal_coeffs, self.ldis.basis_functions()))
        xpoints = numpy.linspace(-1, 1, 100, endpoint=True)
        self.axis_spatial.plot(xpoints, [f(x) for x in xpoints])
        self.axis_spatial.plot(xpoints, [
            self.func(x) for x in xpoints-shift])

        nodes = [x[0] for x in self.ldis.unit_nodes()]
        self.axis_spatial.plot(nodes, [f(x) for x in nodes], "og")

        self.axis_modal.cla()
        self.axis_modal.bar(
                numpy.arange(len(modal_coeffs)),
                modal_coeffs)

        self.canvas.draw()

    def main(self):
        gtk.main()

if __name__ == "__main__":
    order = 15
    from hedge.discretization.local import IntervalDiscretization
    ldis = IntervalDiscretization(order)
    nodal_coeffs = numpy.array([
        1 if x[0] > 0 else 0 for x in ldis.unit_nodes()])
    modal_coeffs = la.solve(ldis.vandermonde(), nodal_coeffs)
    def f(x):
        return sum(coeff*bf([x]) 
                for coeff, bf in zip(modal_coeffs, ldis.basis_functions()))

    ip = InteractivePlot(order, lambda x:1 if x > 0 else 0)
    #ip = InteractivePlot(order, f)
    ip.main()

