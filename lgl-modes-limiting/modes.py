import pygtk
pygtk.require('2.0')
import gtk

import numpy
import numpy.linalg as la

class OrderController:
    def delete_event(self, widget, event, data=None):
        return False

    def destroy(self, widget, data=None):
        gtk.main_quit()

    def __init__(self, order, f=lambda x: 0, l2_project=True):
        from hedge.element import IntervalElement
        self.ldis = IntervalElement(order)

        self.window = gtk.Window(gtk.WINDOW_TOPLEVEL)

        from matplotlib.pyplot import ion

        #ion()

        self.window.connect("delete_event", self.delete_event)
        self.window.connect("destroy", self.destroy)
        self.window.set_border_width(10)
        self.vbox = gtk.VBox()

        self.adjustments = []

        if l2_project:
            from hedge.quadrature import LegendreGaussQuadrature

            quad = LegendreGaussQuadrature(order*5)
            modal_coeff = [
                    quad(lambda x: f(x)*bf([x]))
                    for bf in self.ldis.basis_functions()
                    ]

        else:
            nodal_coeff = numpy.array([
                f(x[0]) for x in self.ldis.unit_nodes()])
            modal_coeff = la.solve(self.ldis.vandermonde(), nodal_coeff)

        for i in range(self.ldis.node_count()):
            adj = gtk.Adjustment(modal_coeff[i], -5, 5)
            self.adjustments.append(adj)
            adj.connect("value-changed", lambda adj, user_arg: self.update_plot(), None)

            hscale = gtk.HScale(adj)
            hscale.set_digits(4)
            hscale.set_value_pos(gtk.POS_RIGHT)
            hscale.set_increments(0.01, 0.1)
            self.vbox.pack_start(hscale, expand=False)
            hscale.show()

        from matplotlib.backends.backend_gtk import FigureCanvasGTK
        from matplotlib.figure import Figure

        self.figure = Figure(figsize=(6,4), dpi=72)
        self.axis = self.figure.add_subplot(111)

        self.canvas = FigureCanvasGTK(self.figure)
        self.vbox.pack_start(self.canvas)
        self.canvas.show()

        self.update_plot()

        self.vbox.show()
        self.window.add(self.vbox)
        self.window.show()

    def update_plot(self):
        self.axis.cla()
        self.axis.set_xlim([-1.05, 1.05])
        self.axis.grid()

        modal_coeffs = [adj.get_value() for adj in self.adjustments]

        def f(x):
            return sum(coeff*bf([x]) 
                    for coeff, bf in zip(modal_coeffs, self.ldis.basis_functions()))
        xpoints = numpy.linspace(-1, 1, 100, endpoint=True)
        self.axis.plot(xpoints, [f(x) for x in xpoints])

        nodes = [x[0] for x in self.ldis.unit_nodes()]
        self.axis.plot(nodes, [f(x) for x in nodes], "og")

        self.canvas.draw()

    def main(self):
        gtk.main()

if __name__ == "__main__":
    oc = OrderController(15, lambda x:1 if x > 0 else 0)
    oc
    oc.main()

