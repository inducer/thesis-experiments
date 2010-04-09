#! /usr/bin/env/python
import sqlite3 as sqlite
import numpy
import numpy.linalg as la
from plot_tools import auto_xy_reshape, unwrap_list




from enthought.traits.api import HasTraits, Range, Instance, \
        on_trait_change
from enthought.traits.ui.api import View, Item, VGroup
from enthought.tvtk.pyface.scene_editor import SceneEditor
from enthought.mayavi.tools.mlab_scene_model import \
        MlabSceneModel
from enthought.mayavi.core.ui.mayavi_scene import MayaviScene




def main():
    import sys
    db_conn = sqlite.connect(sys.argv[1], timeout=30)

    all_angles = unwrap_list(
            db_conn.execute("select distinct angle from data"))
    all_methods = unwrap_list(
            db_conn.execute("select distinct method from data"))
    all_mat_types = unwrap_list(
            db_conn.execute("select distinct mat_type from data"))
    all_mat_types[0:2] = all_mat_types[1::-1]
    all_substep_counts = unwrap_list(
            db_conn.execute("select distinct substep_count from data"))
    all_stable_steps = unwrap_list(
            db_conn.execute("select distinct stable_steps from data"))

    class Visualization(HasTraits):
        scene = Instance(MlabSceneModel, ())
        angle = Range(0, len(all_angles)-1)
        method = Range(0, len(all_methods)-1)
        mat_type = Range(0, len(all_mat_types)-1)
        substep_count = Range(0, len(all_substep_counts)-1)

        def __init__(self):
            HasTraits.__init__(self)

            x, y, z = self.get_data(
                    mat_type=all_mat_types[0],
                    substep_count=all_substep_counts[0],
                    method=all_methods[0],
                    angle=all_angles[0],
                    )

            self.plot = self.scene.mlab.mesh(x, y, z)
            self.first = True
            self.axes = None

        def get_data(self, mat_type, substep_count, method, angle):
            qry = db_conn.execute(
                    "select ratio, offset, dt from data"
                    " where method=? and angle=?"
                    " and mat_type=? and substep_count=?"
                    " and offset <= ?+1e-10"
                    " order by ratio, offset",
                    (method, angle, mat_type, substep_count, numpy.pi))
            x, y, z = auto_xy_reshape(qry)

            import mrab_stability
            factory = getattr(mrab_stability, mat_type)
            print "------------------------------"
            print mat_type, method, substep_count, angle/numpy.pi
            if x:
                ratio = x[0]
                print "matrices for ratio=%g" % ratio

                offset_step = max(len(y)//20, 1)
                for offset in y[::offset_step]:
                    print repr(factory(
                            ratio=ratio,
                            angle=angle,
                            offset=offset)()), offset/numpy.pi
            else:
                print "EMPTY"

            x = numpy.array(x)
            y = numpy.array(y)
            xremap = numpy.empty(z.shape)
            xnew = numpy.cos(y)*x[:,numpy.newaxis]
            ynew = numpy.sin(y)*x[:,numpy.newaxis]

            return xnew, ynew, z

        @on_trait_change('angle,method,mat_type,substep_count')
        def update_plot(self):

            mat_type = all_mat_types[int(self.mat_type)]
            substep_count = all_substep_counts[int(self.substep_count)]
            method = all_methods[int(self.method)]
            angle = all_angles[int(self.angle)]

            x, y, z = self.get_data(
                    mat_type=mat_type,
                    substep_count=substep_count,
                    method=method,
                    angle=angle,
                    )

            self.plot.mlab_source.set(x=x, y=y, scalars=z, z=z)

            if self.first:
                self.axes = self.scene.mlab.axes()
                self.axes.axes.use_data_bounds = True
                self.scene.mlab.scalarbar(
                        orientation="vertical", title="stable dt")
                self.first = False

        # the layout of the dialog created
        view = View(
                Item('scene', 
                    editor=SceneEditor(scene_class=MayaviScene),
                    show_label=False),
                VGroup('_', 'angle', 'method', 'mat_type', 'substep_count'),
                width=1024, height=768, resizable=True)

    visualization = Visualization()
    visualization.configure_traits()



if __name__ == "__main__":
    main()
