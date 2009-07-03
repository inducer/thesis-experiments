import spyctral
import numpy




class MomentumDiscretization:
    def __init__(self, grid_size, method="wiener", hard_scale=None,
            filter_type=None, filter_parameters={},
            bounded_fraction=1.):

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

        params = {}
        if method == "wiener":
            params['s'] = 1
            quadpoints = spyctral.wiener.quad.pgq
            function_evaluation = spyctral.wiener.eval.weighted_wiener
            derivative_evaluation = spyctral.wiener.eval.dweighted_wiener
            indices = spyctral.common.indexing.integer_range(grid_size)
            etas = spyctral.common.indexing.integer_etas(grid_size)
            scaling = lambda N,L,**params: spyctral.wiener.nodes.scale_nodes(N,L,**params)

        elif method == "hermite":
            params['mu'] = 0
            quadpoints = spyctral.hermite.quad.pgq
            function_evaluation = spyctral.hermite.eval.hermite_function
            derivative_evaluation = spyctral.hermite.eval.dhermite_function
            indices = spyctral.common.indexing.whole_range(grid_size)
            etas = spyctral.common.indexing.whole_etas(grid_size)
            scaling = spyctral.hermite.nodes.scale_nodes

        elif method == "mapjpoly":
            params['s'] = 1.
            params['t'] = 1.
            quadpoints = spyctral.mapjpoly.quad.pgq
            function_evaluation = spyctral.mapjpoly.eval.weighted_jacobi_function
            derivative_evaluation = spyctral.mapjpoly.eval.dweighted_jacobi_function
            indices = spyctral.common.indexing.whole_range(grid_size)
            etas = spyctral.common.indexing.whole_etas(grid_size)
            scaling = spyctral.mapjpoly.nodes.scale_nodes

        else:
            raise ValueError("invalid v discretization method: %s" % method)

        if hard_scale is not None:
            params['scale'] = scaling(hard_scale,grid_size,
                    delta=bounded_fraction,**params)

        if filter_type:
            filter = spyctral.filter.filter_parse(filter_type)
            filter_coefficients = filter.modal_weights(etas,**filter_parameters)
        else:
            filter_coefficients = ones(grid_size)

        self.quad_points_1d, self.quad_weights_1d = \
            quadpoints(grid_size,**params)

        self.quad_points = numpy.reshape(self.quad_points_1d,
                (len(self.quad_points_1d), 1))
        self.quad_weights = self.quad_weights_1d

        ps = function_evaluation(self.quad_points_1d,indices,**params)
        dps = derivative_evaluation(self.quad_points_1d, indices,**params)
        self.diffmat = dot(dps*filter_coefficients,ps.T.conj()*self.quad_weights_1d)
        
        if method == "wiener":
            self.diff_function = lambda values: \
                    spyctral.wiener.operators.fft_nodal_differentiation(values, 
                            **params)
