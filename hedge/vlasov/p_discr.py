import spyctral
import numpy




class MomentumDiscretization:
    def __init__(self, grid_size, method="wiener", hard_scale=None,
            filter_type=None, filter_parameters={},
            bounded_fraction=0.8, use_fft=True):

        """ Filter types:
        exponential: {p:8, alpha:-log(eps), eta_cutoff=0.5}
           p: exponent of falloff
           alpha: attenuation at terminal mode
           eta_cutoff: fraction of preserved modes
        lanczos: {}
        cesaro: {}
        raised_cosine: {order:1}
          order: exponent on 0.5*(1+cos(k))

        bounded_fraction: (aka delta)
          fraction of points falling within hard_scale
        """
        from numpy import dot, ones

        self.grid_size = grid_size

        parameters = {"physical_scale": hard_scale,
                "physical_scale_ratio": bounded_fraction}
        if method == "wiener":
            self.basis = spyctral.WienerBasis(self.grid_size,**parameters)
        elif method == "hermite":
            self.basis = spyctral.HermiteBasis(self.grid_size,**parameters)
        else:
            raise ValueError("invalid v discretization method: %s" % method)

        if filter_type == "exponential":
            self.filter =\
                spyctral.ExponentialFilter(self.grid_size,**filter_parameters)
        elif filter_type == "lanczos":
            self.filter =\
                spyctral.LanczosFilter(self.grid_size,**filter_parameters)
        elif filter_type == "raised_cosine":
            self.filter =\
                spyctral.RaisedCosineFilter(self.grid_size,**filter_parameters)
        elif filter_type == "cesaro":
            self.filter =\
                spyctral.CesaroFilter(self.grid_size,**filter_parameters)
        else:
            raise ValueError("invalid filter specification %s" % filter_type)
        self.basis.initialize_filter(self.filter)

        self.quad_points_1d, self.quad_weights_1d = \
            self.basis.quadrature.nodes, self.basis.quadrature.weights

        self.quad_points = numpy.reshape(self.quad_points_1d,
                (len(self.quad_points_1d), 1))
        self.quad_weights = self.quad_weights_1d

        self.diffmat = self.basis.nodal_differentiation_matrix

        if self.basis.fftable and use_fft:
                self.basis.initialize_fft()
                self.diff_function = self.basis.fft_differentiation
        self.apply_filter = self.basis.apply_spectral_filter_to_nodes

        self.basis.make_spectral_filter_matrix_for_nodes()
        self.filter_matrix = self.basis.spectral_filter_matrix_for_nodes
        self.diffmat = dot(self.filter_matrix,self.diffmat)
        #self.filter(nodal_evaluations)
