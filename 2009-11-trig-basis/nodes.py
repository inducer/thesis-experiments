from __future__ import division
import numpy
import numpy.linalg as la




def make_node_count_table():
    from pytools import Table
    tbl = Table()
    tbl.add_row(["n", "linear", "pk", "tri", "tri volume"])

    for n in range(0, 5):
        linear_node_count = 2*n+1
        pk_node_count = (n+2)*(n+3)/2
        node_count = 2*pk_node_count - 1
        tbl.add_row([n, linear_node_count, pk_node_count, 
            node_count, node_count - 3*pk_node_count])
    print tbl




def make_tri_nodes(n):
    sqrt = numpy.sqrt
    from hedge.tools import AffineMap

    equilateral_to_unit = AffineMap(
            numpy.array([[1, -1 / sqrt(3)], [0, 2 / sqrt(3)]]),
            numpy.array([-1/3, -1/3]))

    unit_to_equilateral = equilateral_to_unit.inverted()

    # n: max nu in sin(nu*x)

    linear_node_count = 2*n+1
    pk_node_count = (n+2)*(n+3)/2
    node_count = 2*pk_node_count - 1

    # for square nodes
    #node_count = (2*n)**2 + 1

    fixed_u_nodes = (
        [numpy.array((t,-1), dtype=numpy.float64)
            for t in numpy.linspace(-1, 1, linear_node_count-1, endpoint=False)]
        + [numpy.array((-t,t), dtype=numpy.float64)
            for t in numpy.linspace(-1, 1, linear_node_count-1, endpoint=False)]
        + [numpy.array((-1,-t), dtype=numpy.float64)
            for t in numpy.linspace(-1, 1, linear_node_count-1, endpoint=False)]
        )

    variable_u_nodes = []

    while len(variable_u_nodes) + len(fixed_u_nodes) < node_count:
        pt = numpy.random.uniform(-1, 1, 2)
        if numpy.sum(pt) >= 0:
            continue
        variable_u_nodes.append(pt)

    var_node_count = len(variable_u_nodes)

    fixed_e_nodes = numpy.array([unit_to_equilateral(p) for p in fixed_u_nodes])
    variable_e_nodes = numpy.array([unit_to_equilateral(p) for p in variable_u_nodes])

    from matplotlib.pyplot import plot, show, xlim, ylim, grid, ion, draw, clf

    # variable_e_nodes.flatten().reshape(var_node_count, 2)
    flat_nodes = variable_e_nodes.flatten()

    ion()

    def plot_point_set(var_nodes):
        clf()
        plot(var_nodes[:,0], var_nodes[:,1], "o")
        plot(fixed_e_nodes[:,0], fixed_e_nodes[:,1], "o")

        xlim([-1.2, 1.2])
        ylim([-1.2, 1.2])
        grid()

    class Plane:
        def __init__(self, points):
            a, b = points
            x, y = b-a
            normal = numpy.array([y, -x])
            self.normal = normal/la.norm(normal)
            self.offset = numpy.dot(self.normal, a)

        def __call__(self, x):
            return numpy.dot(x, self.normal)-self.offset

    planes = [
            Plane([unit_to_equilateral(numpy.array(p, dtype=numpy.float64)) for p in pts]) 
            for pts in [
                [(-1, -1), (1, -1)],
                [(1, -1), (-1, 1)],
                [(-1, 1), (-1, -1)],
                ]]

    step = [0]
    def suckiness(flat_e_nodes):
        var_nodes = flat_e_nodes.reshape(var_node_count, 2)
        step[0] += 1
        if step[0] % 20 == 0:
            plot_point_set(var_nodes)
            draw()

        result = 0
        for i, p in enumerate(variable_e_nodes):
            for j, other_p in enumerate(variable_e_nodes):
                if i != j:
                    result += la.norm(p-other_p)**(-2)

            for pl in planes:
                z = pl(p)
                if z > -0.2:
                    result += (z+0.2)**2
        return result

    #import scipy.optimize as opt
    #z = opt.fmin(suckiness, variable_e_nodes.flatten())










    if True:
        from time import sleep

        dt = 0.1
        plane_onset = -0.5
        var_scale = 0.1*dt*1/len(variable_e_nodes)
        fixed_scale = 0.1*dt*1/len(fixed_e_nodes)
        plane_scale = 0.3
        step = 0
        while True:
            step = step + 1

            if step % 5 == 0:
                plot_point_set(variable_e_nodes)
                draw()

            for i, p in enumerate(variable_e_nodes):
                if True:
                    for j, other_p in enumerate(variable_e_nodes):
                        if i != j:
                            r = p-other_p
                            p += var_scale*r/la.norm(r)**3

                for j, other_p in enumerate(fixed_e_nodes):
                    r = p-other_p
                    p += fixed_scale*r/la.norm(r)**3

                for pl in planes:
                    z = pl(p)
                    if z > plane_onset:
                        p += plane_scale*(z-plane_onset)**2 * -pl.normal




if __name__ == "__main__":
    make_node_count_table()
    #make_tri_nodes(2)

