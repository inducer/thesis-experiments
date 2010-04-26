from __future__ import division
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la


from hedge.discretization.local import TriangleDiscretization
t = TriangleDiscretization(5)

o = np.array([-0.3,-0.3])




def get_tri_mesh():
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    x, y = np.meshgrid(x, y)

    x = (x+1)/2 * (1-(y+1)/2) * 2 - 1

    return x, y




def draw_interp():
    from pyrticle._internal import PolynomialShapeFunction
    sf = PolynomialShapeFunction(0.5, 2, 2)

    fig = plt.figure()
    ax = Axes3D(fig)

    x, y = get_tri_mesh()
    nodal_values = np.array([sf(n-o) for n in t.unit_nodes()])
    modal_values = la.solve(t.vandermonde(), nodal_values)

    z = np.array([ 
        sum(m*bf(np.array([xi,yi])) for m, bf in 
            zip(modal_values, t.basis_functions()))
        for xi, yi in zip(x.flat, y.flat)]).reshape(x.shape)

    z_real = np.array([ 
        sf(np.array([xi,yi])-o)
        for xi, yi in zip(x.flat, y.flat)]).reshape(x.shape)

    ax.view_init(30, 230)

    #nodes = np.array(t.unit_nodes())
    #ax.scatter(nodes[:,0], nodes[:,1], nodal_values)

    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.jet)
    plt.savefig("shape-interp.pdf")

    ax.cla()
    ax.plot_surface(x, y, z_real, rstride=1, cstride=1, cmap=cm.jet)
    plt.savefig("shape-real.pdf")





def draw_delta():
    fig = plt.figure()
    ax = Axes3D(fig)

    x, y = get_tri_mesh()
    modal_values = np.array([bf(o) for bf in t.basis_functions()])

    z = np.array([ 
        sum(m*bf(np.array([xi,yi])) for m, bf in 
            zip(modal_values, t.basis_functions()))
        for xi, yi in zip(x.flat, y.flat)]).reshape(x.shape)

    ax.view_init(30, 230)

    #nodes = np.array(t.unit_nodes())
    #ax.scatter(nodes[:,0], nodes[:,1], nodal_values)

    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.jet)
    plt.savefig("delta-proj.pdf")




if __name__ == "__main__":
    draw_interp()
    draw_delta()
