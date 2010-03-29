#! /usr/bin/env/python
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




def main():
    def unwrap_list(iterable):
        return list(row[0] for row in iterable)
    db_conn = sqlite.connect("output.dat", timeout=30)
    all_offsets = unwrap_list(
            db_conn.execute("select distinct offset from data;"))
    all_methods = unwrap_list(
            db_conn.execute("select distinct method from data;"))

    from enthought.traits.api import HasTraits, Range, Instance, \
                        on_trait_change, Enum
    from enthought.traits.ui.api import View, Item, VGroup
    from enthought.tvtk.pyface.scene_editor import SceneEditor
    from enthought.mayavi.tools.mlab_scene_model import \
                        MlabSceneModel
    from enthought.mayavi.core.ui.mayavi_scene import MayaviScene

    class Visualization(HasTraits):
        scene      = Instance(MlabSceneModel, ())
        offset = Range(0, len(all_offsets)-1)
        method = Range(0, len(all_methods)-1)

        def __init__(self):
            HasTraits.__init__(self)

            x, y, z = self.get_data(
                    method=all_methods[0],
                    offset=all_offsets[0],
                    )

            x = numpy.array(x)
            y = numpy.array(y)
            xremap = numpy.empty(z.shape)
            xnew = numpy.cos(y)*x[:,numpy.newaxis]
            ynew = numpy.sin(y)*x[:,numpy.newaxis]

            self.plot = self.scene.mlab.mesh(xnew, ynew, z)
            self.first = True

        def get_data(self, method, offset):
            qry = db_conn.execute(
                    "select ratio, angle, dt from data"
                    " where method=? and offset=?"
                    " order by ratio, angle",
                    (method, offset))
            return  auto_xy_reshape(qry)

        @on_trait_change('offset,method')
        def update_plot(self):
            if self.first:
                self.scene.mlab.axes()
                #self.scene.mlab.xlabel("ratio")
                #self.scene.mlab.ylabel("angle")
                self.first = False

            method = all_methods[int(self.method)]
            offset = all_offsets[int(self.offset)]

            x, y, z = self.get_data(
                    method=method,
                    offset=offset,
                    )

            from mrab_stability import RealMatrixFactory
            print "------------------------------"
            ratio = x[0]
            print method, offset, ratio
            #for ratio in x[::4]:
            for angle in y[::2]:
                print RealMatrixFactory(
                        ratio=ratio,
                        angle=angle,
                        offset=offset)()

            self.plot.mlab_source.set(scalars=z, z=z)

        # the layout of the dialog created
        view = View(
                Item('scene', 
                    editor=SceneEditor(scene_class=MayaviScene),
                    show_label=False),
                VGroup('_', 'offset', 'method',),
                width=1024, height=768)

    visualization = Visualization()
    visualization.configure_traits()




if __name__ == "__main__":
    main()
