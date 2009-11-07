import numpy
import numpy.linalg as la

def make_quad(a, b, c, d):
    return numpy.vstack([
        numpy.hstack([a,b]),
        numpy.hstack([c,d]),
        ])

def rho(mat):
    return max(numpy.abs(la.eigvals(mat)))

def get_norm_rho_row(mat, name=""):
    return (name, 
            "%g" % rho(mat),
            "%g" % la.norm(mat, 1),
            "%g" % la.norm(mat, 2),
            "%g" % la.norm(mat, numpy.inf),
            )

def get_norm_row(mat, name=""):
    return (name, 
            "%g" % la.norm(mat, 1),
            "%g" % la.norm(mat, 2),
            "%g" % la.norm(mat, numpy.inf),
            )
    
def show_2x1_example():
    a = numpy.random.randn(2,2)
    b = numpy.random.randn(2,2)

    f = numpy.hstack([a,b])

    from pytools import Table
    tb = Table()
    tb.add_row(("name", "l1", "l2", "linf"))
    tb.add_row(get_norm_row(a, "a"))
    tb.add_row(get_norm_row(b, "b"))
    tb.add_row(get_norm_row(f, "f"))
    print tb

def show_1x2_example():
    a = numpy.random.randn(2,2)
    b = numpy.random.randn(2,2)

    f = numpy.vstack([a,b])

    from pytools import Table
    tb = Table()
    tb.add_row(("name", "l1", "l2", "linf"))
    tb.add_row(get_norm_row(a, "a"))
    tb.add_row(get_norm_row(b, "b"))
    tb.add_row(get_norm_row(f, "f"))
    print tb




def show_example():
    a = numpy.random.randn(2,2)
    b = numpy.random.randn(2,2)
    c = numpy.random.randn(2,2)
    d = numpy.random.randn(2,2)

    f = make_quad(a, b, c, d)

    from pytools import Table
    tb = Table()
    tb.add_row(("name", "rho", "l1", "l2", "linf"))
    tb.add_row(get_norm_rho_row(a, "a"))
    tb.add_row(get_norm_rho_row(b, "b"))
    tb.add_row(get_norm_rho_row(c, "c"))
    tb.add_row(get_norm_rho_row(d, "d"))
    tb.add_row(get_norm_rho_row(f, "f"))
    print tb


def check():
    pass
def main():
    #show_1x2_example()
    show_2x1_example()



if __name__ == "__main__":
    main()
