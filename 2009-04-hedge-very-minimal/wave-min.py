# Hedge - the Hybrid'n'Easy DG Environment
# Copyright (C) 2007 Andreas Kloeckner
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""This is an example of the very minimum amount of code that's
necessary to get a hedge solver going."""




from __future__ import division
import numpy
import numpy.linalg as la
from hedge.models import HyperbolicOperator
from numpy import dot




class StrongWaveOperator(HyperbolicOperator):
    def __init__(self, dimensions):
        assert isinstance(dimensions, int)

        self.dimensions = dimensions

    def flux(self):
#flux
        from hedge.flux import FluxVectorPlaceholder, make_normal

        w = FluxVectorPlaceholder(1+self.dimensions)
        u = w[0]
        v = w[1:]
        normal = make_normal(self.dimensions)

        from hedge.tools import join_fields
        return - join_fields(
                dot(v.avg, normal) - 0.5*(u.int-u.ext),

                u.avg * normal
                - 0.5*(normal * dot(normal, v.int-v.ext)))
#end

    def op_template(self):
#optemplate
        from hedge.optemplate import (make_vector_field,
                BoundaryPair, get_flux_operator, make_stiffness_t,
                InverseMassOperator, BoundarizeOperator,
                make_normal)

        d = self.dimensions

        w = make_vector_field("w", d+1)
        u = w[0]
        v = w[1:]

        # boundary conditions
        from hedge.tools import join_fields
        from hedge.mesh import TAG_ALL

        dir_normal = make_normal(TAG_ALL, d)

        dir_u = BoundarizeOperator(TAG_ALL)(u)
        dir_v = BoundarizeOperator(TAG_ALL)(v)
        dir_bc = join_fields(-dir_u, dir_v)

        # operator assembly
        flux_op = get_flux_operator(self.flux())

        return InverseMassOperator()(
                join_fields(
                    -dot(make_stiffness_t(d), v),
                    -(make_stiffness_t(d)*u)
                    )
                - (flux_op(w) 
                    + flux_op(BoundaryPair(w, dir_bc, TAG_ALL))))
#end

    def bind(self, discr):
        compiled_op_template = discr.compile(self.op_template())

        def rhs(t, w):
            return compiled_op_template(w=w)

        return rhs

    def max_eigenvalue(self, t, fields=None, discr=None):
        return 1




def main(write_output=True):
    from math import sin, exp, sqrt

    from hedge.mesh.generator import make_rect_mesh
    mesh = make_rect_mesh(a=(-0.5,-0.5),b=(0.5,0.5),max_area=0.008)

    from hedge.backends.jit import Discretization

    discr = Discretization(mesh, order=4, debug=["dump_dataflow_graph"])

    from hedge.visualization import VtkVisualizer
    vis = VtkVisualizer(discr, None, "fld")

    def ic_u(x, el):
        x = x - numpy.array([0.3,0.22])
        return exp(-numpy.dot(x, x)*128)


    op = StrongWaveOperator(discr.dimensions)

    from hedge.tools import join_fields
    fields = join_fields(discr.interpolate_volume_function(ic_u),
            [discr.volume_zeros() for i in range(discr.dimensions)])

    from hedge.timestep.runge_kutta import LSRK4TimeStepper
    stepper = LSRK4TimeStepper()
    dt = op.estimate_timestep(discr, stepper=stepper, fields=fields)

    nsteps = int(1/dt)
    print "dt=%g nsteps=%d" % (dt, nsteps)

    rhs = op.bind(discr)
    for step in range(nsteps):
        t = step*dt

        if step % 50 == 0 and write_output:
            print step, t, discr.norm(fields[0])
            visf = vis.make_file("fld-%04d" % step)

            vis.add_data(visf,
                    [ ("u", fields[0]), ("v", fields[1:]), ],
                    time=t, step=step)
            visf.close()

        fields = stepper(fields, t, dt, rhs)

    vis.close()




if __name__ == "__main__":
    main()
