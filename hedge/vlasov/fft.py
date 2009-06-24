from __future__ import division
import numpy
import numpy.linalg as la




def fft(x, sign=1, wrap_intermediate=lambda x: x):
    """Computes the Fourier transform of x:

    F[x]_i = \sum_{j=0}^{n-1} z^{ij} x_j

    where z = exp(sign*-2j*pi/n) and n = len(x) must be a power of 2.
    """

    if len(x) == 1:
        return x

    from math import pi
    ft_even = wrap_intermediate(fft(x[::2], sign, wrap_intermediate))
    ft_odd = wrap_intermediate(fft(x[1::2], sign, wrap_intermediate) \
            * numpy.exp(numpy.linspace(0, sign*(-1j)*pi, len(x)//2,
                endpoint=False)))

    return numpy.hstack([ft_even+ft_odd, ft_even-ft_odd])




def test_with_floats():
    for i in range(4, 9):
        n = 2**i
        a = numpy.random.rand(n) + 1j*numpy.random.rand(n)
        f_a = fft(a)
        a2 = 1/n*fft(f_a, -1)
        assert la.norm(a-a2) < 1e-12

        f_a_numpy = numpy.fft.fft(a)
        assert la.norm(f_a-f_a_numpy) < 1e-12




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

    traced_fft = fft(vars, wrap_intermediate=wrap_intermediate)

    traced_fft = nzk(traced_fft)

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
