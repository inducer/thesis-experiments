import OCC.gp as gp
import OCC.BRepPrimAPI as brep
import OCC.Utils.DataExchange.STEP as step
import OCC.Utils.DataExchange.IGES as iges

ax = gp.gp_Ax2()
ax.SetLocation(gp.gp_Pnt(0, 0, 0))
ax.SetDirection(gp.gp_Dir(1, 0, 0))
cyl = brep.BRepPrimAPI_MakeCylinder(ax, 50, 50)

step_writer = step.STEPExporter("one-cylinder.step")
step_writer.AddShape(cyl.Shape())
step_writer.WriteFile()

iges_writer = iges.IGESExporter("one-cylinder.igs")
iges_writer.AddShape(cyl.Shape())
iges_writer.WriteFile()
