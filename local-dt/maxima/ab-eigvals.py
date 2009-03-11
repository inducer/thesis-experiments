from __future__ import division
import numpy
import enthought.mayavi.mlab as mlab

def eigvals(a,b,c,d,h):
    from cmath import sqrt
    return [-(h*sqrt(a**4*h**2+(-8*a**2*d+16*a*b*c+10*a**3)*h+16*d**2-32*a*d+64*b*c+16*a**2)
              -a**2*h**2+(-4*d-4*a)*h-8)/8.0E+0,
            (h*sqrt(a**4*h**2+(-8*a**2*d+16*a*b*c+8*a**3)*h+16*d**2-32*a*d+64*b*c+16*a**2)
              +a**2*h**2+(4*d+4*a)*h+8)/8.0E0]

def rho(**kwargs):
    ev = eigvals(**kwargs)
    return max(abs(i) for i in ev)
    #return ev[1].imag

def main():
    ctx = dict(a=2j, d=1j, h=0.1)

    eps = 1e-5
    x, y = numpy.mgrid[-7:7+eps:0.1, -5:5+eps:0.1]

    def f(x, y):
        return rho(b=x, c=y, **ctx)

    d = numpy.vectorize(f)(x, y)
    s = mlab.surf(x, y, d, warp_scale=10)
    mlab.axes()
    for h in numpy.arange(0.1, 1, 0.1):
        print h

        ctx['h'] = h
        s.mlab_source.scalars = numpy.vectorize(f)(x, y)
        
        from time import sleep
        sleep(1)



if __name__ == "__main__":
    main()
