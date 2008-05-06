def main():
    order = 5

    from hedge.mesh import make_regular_rect_mesh
    mesh = make_regular_rect_mesh(n=(order+2,order+2))
    from hedge.element import TriangularElement
    ldis = TriangularElement(order, fancy_node_ordering=False)
    from hedge.discretization import Discretization
    discr = Discretization(mesh, ldis)

    u = discr.volume_zeros()
    node_idx = 0
    for i in range(order+1):
        for j in range(order+1):
            if j+i >= order+1:
                continue
                
            el_index = i*(order+1)+j
            print el_index, (i,j)
            el_start,el_end = discr.find_el_range(el_index*2)
            u[el_start+node_idx] = 1
            node_idx += 1 

    assert node_idx == ldis.node_count()

    from hedge.discretization import Filter, ExponentialFilterResponseFunction
    filt = Filter(discr, ExponentialFilterResponseFunction(
        0.01, 6))

    from hedge.visualization import SiloVisualizer
    vis = SiloVisualizer(discr)
    visf = vis.make_file("spikes")
    vis.add_data(visf, [ 
        ("u", u), 
        ("filt_u", filt(u)), 
        ] )
    visf.close()
    vis.close()




if __name__ == "__main__":
    main()
