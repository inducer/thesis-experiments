from matplotlib.pyplot import plot, show, grid, xlabel, ylabel, rc, legend, title
from pylo import SiloFile, DB_READ

font_size = 30

def make_shock_wrinkles_pic():
    rc("font", size=font_size)
    db = SiloFile(
            "euler-2010-03-04-141612/N7-K641-v0.400000-VertexwiseMaxSmoother/euler-003823.silo",
            create=False, mode=DB_READ)
    rho_n7 = db.get_curve("rho")
    db = SiloFile(
            "euler-2010-03-04-141612/N5-K81-v0.400000-VertexwiseMaxSmoother/euler-000570.silo",
            create=False, mode=DB_READ)
    rho_n5 = db.get_curve("rho")

    plot(rho_n7.x, rho_n7.y, label="$N=7$ $K=641$")
    plot(rho_n5.x, rho_n5.y, "o-", markersize=4, label="$N=5$ $K=81$")
    xlabel("$t$")
    ylabel(r"$\rho$")
    legend(loc="best")
    grid()
    show()

def make_lax_pic():
    rc("font", size=font_size)
    db = SiloFile(
            "euler-004622.silo",
            create=False, mode=DB_READ)
    rho = db.get_curve("rho")
    p = db.get_curve("p")

    title("Shock-Wave Interaction Problem with $N=5$ and $K=80$")
    #title("Lax's Problem with $N=5$ and $K=80$")
    plot(rho.x, rho.y, label=r"$\rho$")
    plot(p.x, p.y, label=r"$p$")
    xlabel("$t$")
    legend(loc="best")
    grid()
    show()




if __name__ == "__main__":
    #make_shock_wrinkles_pic()
    make_lax_pic()
