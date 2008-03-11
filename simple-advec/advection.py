from __future__ import division
import pylinear.array as num
import pylinear.computation as comp




def main() :
    from math import sin, cos, pi, sqrt
    from math import floor

    def boundary_tagger(vertices, el, face_nr):
        if el.face_normals[face_nr] * v <= 0:
            return ["inflow"]
        else:
            return ["outflow"]

    v = num.array([1,0])
    from hedge.mesh import make_rect_mesh
    mesh = make_rect_mesh(
            (-0.5, -0.5),
            (-0.5+2, 0.5),
            max_area=0.02,
            boundary_tagger=boundary_tagger,
            periodicity=(True, False),
            subdivisions=(10,5),
            )

    from hedge.discretization import Discretization
    from hedge.element import TriangularElement
    discr = Discretization(mesh, TriangularElement(7))

    from hedge.visualization import SiloVisualizer
    vis = SiloVisualizer(discr)

    # operator setup ----------------------------------------------------------
    from hedge.data import \
            ConstantGivenFunction, \
            TimeConstantGivenFunction, \
            TimeDependentGivenFunction

    from hedge.operators import StrongAdvectionOperator
    op = StrongAdvectionOperator(discr, v, 
            inflow_u=TimeConstantGivenFunction(ConstantGivenFunction()),
            #inflow_u=TimeDependentGivenFunction(u_analytic)),
            flux_type="upwind")

    def gauss_hump(x):
        from math import exp
        rsquared = (x*x)/(0.1**2)
        return exp(-rsquared)

    #u = discr.interpolate_volume_function(lambda x: u_analytic(x, 0))
    u = discr.interpolate_volume_function(gauss_hump)
    #u /= integral(discr, u)

    # timestep loop -----------------------------------------------------------
    from hedge.timestep import RK4TimeStepper
    stepper = RK4TimeStepper()

    dt = discr.dt_factor(op.max_eigenvalue())
    nsteps = int(300/dt)

    print "%d elements, dt=%g, nsteps=%d" % (
            len(discr.mesh.elements),
            dt,
            nsteps)

    for step in range(nsteps):
        t = step*dt

        if step % 20 == 0:
            print "step %d" % step
            visf = vis.make_file("fld-%04d" % step)
            vis.add_data(visf, [
                        ("u", u), 
                        ], 
                        time=t, 
                        step=step
                        )
            visf.close()

        u = stepper(u, t, dt, op.rhs)

    vis.close()


if __name__ == "__main__":
    main()
