# Hedge - the Hybrid'n'Easy DG Environment
# Copyright (C) 2008 Andreas Kloeckner
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




import numpy
import numpy.linalg as la




def main():
    from hedge.discretization.local import (
            TriangleDiscretization,
            TetrahedronDiscretization)

    t = TriangleDiscretization(3)
    d = t.differentiation_matrices()

    v = t.vandermonde()

    from hedge.tools import leftsolve
    def chop(m):
        result = m.copy()
        result[numpy.abs(m) < 1e-13] = 0
        return result

    d_modal = [
            chop(leftsolve(v, numpy.dot(di, v)))
            for di in d]

    if False:
        print list(t.generate_mode_identifiers())
        for dmi in d_modal:
            print dmi
            print la.svd(dmi)[1]
            print

    print list(t.generate_mode_identifiers())
    laplace = chop(sum(
            numpy.dot(dmi, dmi)
            for dmi in d_modal))

    print laplace

    print chop(la.svd(laplace)[1])



if __name__ == "__main__":
    main()
