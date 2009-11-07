from __future__ import division
import numpy
import numpy.linalg as la
import matplotlib.pyplot as plt

#def check(a, n_small, n_large):
def main():
    a = numpy.array([[2j,0],[0,1j]])
    a_dir1 = numpy.array([[0,2],[0,0]])
    a_dir2 = numpy.array([[0,2],[2j,1]])

    color = numpy.array([0,0,1])
    color_dir1 = numpy.array([1,0,-1])
    color_dir2 = numpy.array([0,1,0])

    plt.ion()
    plt.grid()

    ev = la.eigvals(a)
    plt.scatter(ev.real, ev.imag, color=color, s=100)

    plt.draw()
    for eps1 in numpy.arange(0, 1, 0.1):
        for eps2 in numpy.arange(0, 1, 0.1):
            ev = la.eigvals(a+eps1*a_dir1+eps2*a_dir2)
            plt.scatter(
                    ev.real, ev.imag, 
                    color=color+eps1*color_dir1+eps2*color_dir2)
        plt.draw()
        from time import sleep
        sleep(0.1)

    plt.show()




if __name__ == "__main__":
    main()
