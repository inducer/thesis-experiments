#! /usr/bin/env/python
from enthought.mayavi.mlab import surf, points3d, show, axes
import sqlite3 as sqlite
import numpy
import numpy.linalg as la
from time import sleep

def group_by_first(cursor):
    last = None
    group = []
    for row in cursor:
        if last is None:
            last = row[0]
        elif last != row[0]:
            yield last, group
            group = []
            last = row[0]

        group.append(row[1:])

    if group:
        yield row[0], group

def auto_xy_reshape(cursor):
    x_values = []
    y_values = []
    z_values = []

    for x, y, v in cursor:
        z_values.append(v)
        if x not in x_values:
            x_values.append(x)

        if y not in y_values:
            y_values.append(y)

    z_values = numpy.array(z_values).reshape(
            (len(x_values), len(y_values)), order="C")

    return x_values, y_values, z_values


def plot_xyv_mayavi(cursor):
    x, y, z = zip(*list(cursor))
    points3d(x, y, z)

def plot_xyv2_mayavi(cursor):
    x, y, z, v = zip(*list(cursor))
    points3d(x, y, z, v, scale_mode='none')

def plot_xyv_surf_mayavi(cursor):
    x, y, z = auto_xy_reshape(cursor)
    surf(x,y, z)


def main():
    db_conn = sqlite.connect("output.dat", timeout=30)
    qry = db_conn.execute(
            "select offset, ratio, angle, dt from data"
            " where method='slowest_first_2w'"
            " order by offset, ratio, angle")

    obj = None
    for first, group in group_by_first(qry):
        print first
        x, y, z = auto_xy_reshape(group)

        if obj is None:
            obj = surf(x, y, z)
            axes()
            #show()
            sleep(1)
        else:
            obj.mlab_source.scalars = z
        sleep(0.5)

if __name__ == "__main__":
    main()
