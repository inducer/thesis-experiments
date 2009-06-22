import spyctral
v_method = "mapjpoly"
hard_scale = 5.
v_params = {}
if v_method is "wiener":
    v_params['s'] = 1
    v_quadpoints = spyctral.wiener.quad.pgq
    v_function_evaluation = spyctral.wiener.eval.weighted_wiener
    v_derivative_evaluation = spyctral.wiener.eval.dweighted_wiener
    v_indices = spyctral.common.indexing.integer_range(v_grid_size)
    v_scaling = lambda N,L,**params: spyctral.wiener.nodes.scale_nodes(N,L,s=wiener_s,**params)
    
elif v_method is "hermite":
    v_params['mu'] = 0
    v_quadpoints = spyctral.hermite.quad.pgq
    v_function_evaluation = spyctral.hermite.eval.hermite_function
    v_derivative_evaluation = spyctral.hermite.eval.dhermite_function
    v_indices = spyctral.common.indexing.whole_range(v_grid_size)
    v_scaling = spyctral.hermite.nodes.scale_nodes
    
elif v_method is "mapjpoly":
    v_params['s'] = 1.
    v_params['t'] = 1.
    v_quadpoints = spyctral.mapjpoly.quad.pgq
    v_function_evaluation = spyctral.mapjpoly.eval.weighted_jacobi_function
    v_derivative_evaluation = spyctral.mapjpoly.eval.dweighted_jacobi_function
    v_indices = spyctral.common.indexing.whole_range(v_grid_size)
    v_scaling = spyctral.mapjpoly.nodes.scale_nodes

if hard_scale is not None:
    v_params['scale'] = v_scaling(hard_scale,v_grid_size,delta=1.0,**v_params)

self.v_quad_points_1d, self.v_quad_weights_1d = \
    v_quadpoints(v_grid_size,**v_params)

self.v_quad_points = numpy.reshape( self.v_quad_points_1d,
        (len(self.v_quad_points_1d), 1)) 
self.v_quad_weights = self.v_quad_weights_1d

from numpy import dot 
ps = v_function_evaluation(self.v_quad_points_1d,v_indices,**v_params) 
dps = v_derivative_evaluation(self.v_quad_points_1d, v_indices,**v_params) 
self.v_diffmat = dot(dps,ps.T.conj()*self.v_quad_weights_1d)
