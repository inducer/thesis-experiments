from __future__ import division
from pytools import Record
from math import sqrt
import numpy



class SodData(Record):
    def __call__(self, x):
        xa = self.abscissae
        x = x[0]

        idx = numpy.searchsorted(xa, x, side="left")

        if idx == 0:
            def get(ary):
                return ary[0]
        else:
            if xa[idx] == xa[idx-1]:
                t = 0
            else:
                t = (x-xa[idx-1])/(xa[idx]-xa[idx-1])

            def get(ary):
                z = t*ary[idx-1] + (1-t)*ary[idx]
                if numpy.isnan(z) or numpy.isinf(z):
                    from pudb import set_trace;set_trace()
                return z

        return [get(self.rho), get(self.e),
                get(self.rho_u)] + [0]*(self.dim-1)





class SodProblem:
    """
    Computes the exact solution for the Sod's shock-tube problem.
    See J.D. Anderson, Modern Compressible Flow (1984) for details.

       --------------------------------------------
       |                     |                    |
       |    p4, r4, u4       |     p1, r1, u1     |   t = 0.0
       |                     |                    |
       --------------------------------------------
       xl                    xd                   xr
                  p4>p1


       --------------------------------------------
       |      |      |           | up       | W   |
       |   4  |<-----|    3    --|->  2   --|-->  |
       |      |      |           |          |     |
       --------------------------------------------
       xl    expansion          slip     shock    xr

    See
    http://www.grc.nasa.gov/WWW/wind/valid/stube/stube.html
    """
    # (via Alan Schiemenz)

    def __init__(self, p_l=1, p_r=0.1, rho_l=1, rho_r=0.125, 
            gamma=1.4, dim=1, xl=0, xr=1, tol=1e-5):
        self.p_l = p_l
        self.p_r = p_r
        self.rho_l = rho_l
        self.rho_r = rho_r
        self.gamma = gamma

        self.xl = xl
        self.xr = xr

        self.dim = dim
        self.shape = (2+dim,)

        self.tol = tol

    def get_data_for_time(self, t):
        gam = self.gamma
        gm1 = gam - 1.0
        gp1 = gam + 1.0

        # Set initial states (non-dimensional).

        p4 = self.p_l
        r4 = self.rho_l
        u4 = 0.0

        p1 = self.p_r
        r1 = self.rho_r
        u1 = 0.0

        # Set dimensions of shocktube.

        xl = self.xl
        xr = self.xr
        xd = (xl+xr)/2

        # Compute acoustic velocities.

        a1 = sqrt(gam * p1 / r1)
        a4 = sqrt(gam * p4 / r4)

        # Use a Newton-secant iteration to compute p2p1.

        p2p1 = self.sp2p1(a1, a4)

        t2t1 = p2p1 * ( gp1/gm1 + p2p1 ) / ( 1.0 + gp1 * p2p1 / gm1 )
        r2r1 = ( 1.0 + gp1 * p2p1 / gm1 ) / ( gp1 / gm1 + p2p1 )

        # W, shock-wave speed.

        wsp  = a1 * sqrt( gp1 * ( p2p1 - 1.0 ) / ( 2.0 * gam ) + 1.0 )

        # Shock location.

        xs = xd + wsp * t

        # State 2.

        p2 = p2p1 * p1
        r2 = r2r1 * r1

        # State 3.

        p3 = p2

        # Isentropic between 3 and 4.

        r3 = r4 * ( p3 / p4 )**(1.0/gam)

        a3 = sqrt( gam * p3 / r3 )

        # Speed of contact discontinuity.

        up = 2.0 * a4 * ( 1.0 - (p2/p4)**(0.5*gm1/gam) ) / gm1

        u2 = up
        u3 = up

        # Location of contact discontinuity.

        xc = xd + up * t

        # Location of expansion region.

        xhead = xd + ( u4 - a4 ) * t
        xtail = xd + ( u3 - a3 ) * t

        if t == 0:
            abscissae =  [xl, xc, xc, xr]
            p_arr = [p4, p4, p1, p1]
            rho_arr = [r4, r4, r1, r1]
            u_arr = [u4, u4, u1, u1]
        else:
            abscissae = [xl, xhead]
            p_arr = [p4, p4]
            rho_arr = [r4, r4]
            u_arr = [u4, u4]

            nxp = 89

            for n in range(1, nxp+1):
                xx = xhead + ( xtail - xhead )  * n / ( nxp + 1.0 )
                ux = u4 + u3 * ( xx - xhead ) / ( xtail - xhead )
                px = p4 * ( 1.0 - 0.5 * gm1 * ( ux / a4 ) )**( 2.0 * gam / gm1 )
                rx = r4 * ( 1.0 - 0.5 * gm1 * ( ux / a4 ) )**( 2.0 / gm1 )

                abscissae.append(xx)
                p_arr.append(px)
                rho_arr.append(rx)
                u_arr.append(ux)

            abscissae.extend( [xtail, xc, xc, xs, xs, xr])
            p_arr.extend(     [p3,    p3, p2, p2, p1, p1])
            rho_arr.extend(   [r3,    r3, r2, r2, r1, r1])
            u_arr.extend(     [u3,    u3, u2, u2, u1, u1])

        p_arr = numpy.array(p_arr)
        rho_arr = numpy.array(rho_arr)
        u_arr = numpy.array(u_arr)
        e_arr = p_arr / gm1 + rho_arr / 2 * u_arr**2

        return SodData(
                abscissae=numpy.array(abscissae),
                p=p_arr, e=e_arr, rho=rho_arr, u=u_arr,
                rho_u=rho_arr*u_arr, dim=self.dim)



    def sp2p1(self, a1, a4):
        # Uses Newton-secant method to iterate on eqn 7.94 (Anderson, 
        # 1984) to find p2p1 across moving shock wave.

        # Set some variables

        gam = self.gamma
        gm1 = gam - 1.0
        gp1 = gam + 1.0

        p4 = self.p_l
        p1 = self.p_r

        # Initialize p2p1 for starting guess

        p2p1m = 0.9 * p4 / p1

        t1 = - 2.0 * gam / gm1

        t2 = gm1 * ( a1 / a4 ) * ( p2p1m - 1.0 )
        t3 = 2.0 * gam * ( 2.0 * gam + gp1 * ( p2p1m - 1.0 ) )
        fm = p4 / p1 - p2p1m * ( 1.0 - t2 / sqrt(t3) )**t1

        # Perturb p2p1

        p2p1 = 0.95 * p2p1m

        # Begin iteration

        itcount  = 0
        itmax = 20

        while True:
            itcount = itcount + 1

            t2 = gm1 * ( a1 / a4 ) * ( p2p1 - 1.0 )
            t3 = 2.0 * gam * ( 2.0 * gam + gp1 * ( p2p1 - 1.0 ) )

            f  = p4 / p1 - p2p1 * ( 1.0 - t2 / sqrt(t3) )**t1 

            if abs(f) > self.tol and  itcount < itmax:
                p2p1n = p2p1 - f * ( p2p1 - p2p1m ) / ( f - fm )
                p2p1m = p2p1
                fm    = f
                p2p1  = p2p1n
            else:
                break

        if itcount > itmax:
            raise RuntimeError("secant method for Sod didn't converge")

        return p2p1




if __name__ == "__main__":
    sod = SodFunction()
    
    for t in numpy.linspace(0, 0.2, 4):
        data = sod.compute_points_for_t(t)
        from matplotlib.pyplot import plot, show
        plot(data.abscissae, data.u)
    show()

