import numpy
import OCC.gp as gp
import OCC.BRepPrimAPI as brep
import OCC.BRepAlgoAPI as brep_alg
import OCC.Utils.DataExchange.STEP as step
import OCC.Utils.DataExchange.IGES as iges

dirs = [numpy.array([1, 0, 0]), numpy.array([0, 1, 0])]

cyls = []
for d in dirs:
    ax = gp.gp_Ax2()
    ax.SetLocation(gp.gp_Pnt(0,0,0))
    ax.SetDirection(gp.gp_Dir(*d))
    cyl = brep.BRepPrimAPI_MakeCylinder(ax, 5, 5)
    cyls.append(cyl)

#csg = brep_alg.BRepAlgoAPI_Fuse(cyls[0].Shape(), cyls[1].Shape())
#csg = brep_alg.BRepAlgoAPI_Cut(cyls[0].Shape(), cyls[1].Shape())
csg = brep_alg.BRepAlgoAPI_Common(cyls[0].Shape(), cyls[1].Shape())
csg_shape = csg.Shape()

step_writer = step.STEPExporter("two-cylinders.step")
step_writer.AddShape(csg_shape)
step_writer.WriteFile()

iges_writer = iges.IGESExporter("two-cylinders.igs")
iges_writer.AddShape(csg_shape)
iges_writer.WriteFile()

#display.DisplayShape(union_shape)
