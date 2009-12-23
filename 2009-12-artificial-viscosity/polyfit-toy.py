x_points = []
y_points = []

import numpy
from matplotlib.pyplot import (clf, plot, show, xlim, ylim, 
        get_current_fig_manager, gca, draw, connect)

deg = 1

def update_plot():
    clf()
    xlim([0, 10])
    ylim([0, 10])
    gca().set_autoscale_on(False)
    plot(x_points, y_points, 'o')

    if len(x_points) >= deg+1:
        eval_points = numpy.linspace(0, 10, 100)
        poly = numpy.poly1d(numpy.polyfit(
                    numpy.array(x_points), 
                    numpy.array(y_points), deg))
        plot(eval_points, poly(eval_points), "-")


def click(event):
   """If the left mouse button is pressed: draw a little square. """
   tb = get_current_fig_manager().toolbar
   if event.button==1 and event.inaxes and tb.mode == '':
       x_points.append(event.xdata)
       y_points.append(event.ydata)
       update_plot()
       draw()

update_plot()
connect('button_press_event', click)
show()
