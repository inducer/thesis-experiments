import spyctral
import numpy




class VelocityDiscretization:
    def __init__(self, grid_size, method="wiener", hard_scale=None):
        self.grid_size = grid_size
        params = {}
        if method is "wiener":
            params['s'] = 1
            quadpoints = spyctral.wiener.quad.pgq
            function_evaluation = spyctral.wiener.eval.weighted_wiener
            derivative_evaluation = spyctral.wiener.eval.dweighted_wiener
            indices = spyctral.common.indexing.integer_range(grid_size)
            scaling = lambda N,L,**params: spyctral.wiener.nodes.scale_nodes(N,L,s=wiener_s,**params)

        elif method is "hermite":
            params['mu'] = 0
            quadpoints = spyctral.hermite.quad.pgq
            function_evaluation = spyctral.hermite.eval.hermite_function
            derivative_evaluation = spyctral.hermite.eval.dhermite_function
            indices = spyctral.common.indexing.whole_range(grid_size)
            scaling = spyctral.hermite.nodes.scale_nodes

        elif method is "mapjpoly":
            params['s'] = 1.
            params['t'] = 1.
            quadpoints = spyctral.mapjpoly.quad.pgq
            function_evaluation = spyctral.mapjpoly.eval.weighted_jacobi_function
            derivative_evaluation = spyctral.mapjpoly.eval.dweighted_jacobi_function
            indices = spyctral.common.indexing.whole_range(grid_size)
            scaling = spyctral.mapjpoly.nodes.scale_nodes

        if hard_scale is not None:
            params['scale'] = scaling(hard_scale,grid_size,delta=1.0,**params)

        self.quad_points_1d, self.quad_weights_1d = \
            quadpoints(grid_size,**params)

        self.quad_points = numpy.reshape( self.quad_points_1d,
                (len(self.quad_points_1d), 1))
        self.quad_weights = self.quad_weights_1d

        from numpy import dot
        ps = function_evaluation(self.quad_points_1d,indices,**params)
        dps = derivative_evaluation(self.quad_points_1d, indices,**params)
        self.diffmat = dot(dps,ps.T.conj()*self.quad_weights_1d)
