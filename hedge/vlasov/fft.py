from __future__ import division
import numpy
import cmath
import numpy.linalg as la
from pytools import memoize





@memoize
def find_factors(N):
    from math import sqrt

    N1 = 2
    max_N1 = int(sqrt(N))+1
    while N % N1 != 0 and N1 <= max_N1:
        N1 += 1

    if N1 > max_N1:
        N1 = N

    N2 = N // N1

    return N1, N2




def fft(x, sign=1, wrap_intermediate=lambda x: x):
    """Computes the Fourier transform of x:

    F[x]_i = \sum_{j=0}^{n-1} z^{ij} x_j

    where z = exp(sign*-2j*pi/n) and n = len(x).
    """

    # http://en.wikipedia.org/wiki/Cooley-Tukey_FFT_algorithm
    # revision 293076305, http://is.gd/1c7PI

    from math import pi

    N = len(x)

    if N == 1:
        return x

    N1, N2 = find_factors(N)

    # do the transform
    sub_ffts = [
            wrap_intermediate(
                fft(x[n1::N1], sign, wrap_intermediate)
                * numpy.exp(numpy.linspace(0, sign*(-2j)*pi*n1/N1, N2,
                    endpoint=False)))
            for n1 in range(N1)]

    return numpy.hstack([
        sum(subvec * cmath.exp(sign*(-2j)*pi*n1*k1/N1)
            for n1, subvec in enumerate(sub_ffts))
        for k1 in range(N1)
        ])




def test_with_floats():
    for n in [2**i for i in range(4, 10)]+[17, 12, 948]:
        a = numpy.random.rand(n) + 1j*numpy.random.rand(n)
        f_a = fft(a)
        a2 = 1/n*fft(f_a, -1)
        assert la.norm(a-a2) < 1e-10

        f_a_numpy = numpy.fft.fft(a)
        assert la.norm(f_a-f_a_numpy) < 1e-10




from pymbolic.mapper import IdentityMapper
class NearZeroKiller(IdentityMapper):
    def map_constant(self, expr):
        if isinstance(expr, complex):
            r = expr.real
            i = expr.imag
            if abs(r) < 1e-15:
                r = 0
            if abs(i) < 1e-15:
                i = 0
            return complex(r, i)
        else:
            return expr





def test_with_pymbolic():
    from pymbolic import var
    vars = numpy.array([var(chr(97+i)) for i in range(16)], dtype=object)
    print vars

    def wrap_intermediate(x):
        if len(x) > 1:
            from hedge.optemplate import make_common_subexpression
            return make_common_subexpression(x)
        else:
            return x

    nzk = NearZeroKiller()
    print nzk(fft(vars))
    traced_fft = nzk(fft(vars, wrap_intermediate=wrap_intermediate))

    from pymbolic.mapper.stringifier import PREC_NONE
    from pymbolic.mapper.c_code import CCodeMapper
    ccm = CCodeMapper()

    code = [ccm(tfi, PREC_NONE) for tfi in traced_fft]

    for i, cse in enumerate(ccm.cses):
        print "_cse%d = %s" % (i, cse)

    for i, line in enumerate(code):
        print "result[%d] = %s" % (i, line)




if __name__ == "__main__":
    test_with_floats()
    test_with_pymbolic()
