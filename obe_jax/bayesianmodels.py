from jax import jit,vmap

from .abstractbayesmodel import AbstractBayesianModel
from .utility_measures import entropy_change

class SimulatedModel(AbstractBayesianModel):
    """
    A type of BayesianModel where the likelihood
    function is derived from a simulation. 
    
    The precompute_function takes in an input and parameter
    specification and does a computation that will be passed
    into the simulation_likelihood function. The latest precomputation 
    results are cached for future re-use to minimize
    computational overhead.
    
    The simulation_likelihood is a likelihood function that
    takes in input, output, parameters, and precomputed_array. 
    
    Both precompute_function and simulation_likelihood must be 
    `jax.jit`-able. 

    precompute_function(oneinput_vec,oneparameter_vec)
    simulation_likelihood(oneinput_vec,oneoutput_vec,oneparameter_vec,precompute_data)

    
    """
    
    def __init__(self, key, particles, weights, 
                 precompute_function, 
                 simulation_likelihood,
                 **kwargs):
        
        self.sim_lower_kwargs = kwargs
        self.precompute_function = precompute_function
        self.precompute_oneinput_multiparams = jit(vmap(precompute_function,in_axes=(None,1),out_axes=(-1)))
        self.simulation_likelihood = simulation_likelihood
        self.sim_likelihood_oneinput_oneoutput_multiparams = jit(vmap(simulation_likelihood,in_axes=(None,None,1,-1)))
        self.latest_precomputed_data = None

        @jit
        def likelihood(oneinput_vec,oneoutput_vec,oneparameter_vec):
            precompute_data = precompute_function(oneinput_vec,oneparameter_vec)
            return simulation_likelihood(oneinput_vec,oneoutput_vec,oneparameter_vec,precompute_data)
        
        AbstractBayesianModel.__init__(self, key, particles, weights, 
                                       likelihood_function=likelihood,**kwargs)
        
        
    @jit
    def updated_weights_precomputes_from_experiment(self, oneinput, oneoutput):
        precomputes = self.precompute_oneinput_multiparams(oneinput,self.particles)
        ls = self.oneinput_oneoutput_multiparams(oneinput, oneoutput, self.particles, precomputes)
        weights = self.update_weights(ls)
        return weights, precomputes
        
    @jit
    def updated_weights_from_precompute(self, oneinput, oneoutput):
        ls = self.sim_likelihood_oneinput_oneoutput_multiparams(oneinput, oneoutput, self.particles, self.latest_precomputed_data)
        weights = self.update_weights(ls)
        return weights
        
    def bayesian_update(self, oneinput, oneoutput, use_latest_precompute=False):
        """
        Refines the parameters' probability distribution function given a
        measurement result.

        This is where measurement results are entered. An implementation of
        Bayesian inference, uses the model to calculate the likelihood of
        obtaining the measurement result as a function of
        parameter values, and uses that likelihood to generate a refined
        *posterior* (after-measurement) distribution from the *prior* (
        pre-measurement) parameter distribution.
        
        If use_latest_precompute=True and the ParticlePDF hasn't been resampled
        then the previously cached precompute results are input into the 
        simulation_likelihood function.

        """
        
        if use_latest_precompute and not self.just_resampled:
            self.weights = self.updated_weights_from_precompute(oneinput, oneoutput)
        else:
            self.weights, self.latest_precomputed_data = self.updated_weights_precomputes_from_experiment(oneinput, oneoutput)
        
        if self.tuning_parameters['auto_resample']:
            self.resample_test()

    def _tree_flatten(self):
        children = (self.key, self.particles, self.weights)  # arrays / dynamic values
        aux_data = {'precompute_function':self.precompute_function, 
                    'simulation_likelihood':self.simulation_likelihood, **self.sim_lower_kwargs
                   }
        return (children, aux_data)
    
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children,**aux_data)
    
from jax import tree_util
tree_util.register_pytree_node(SimulatedModel,
                               SimulatedModel._tree_flatten,
                               SimulatedModel._tree_unflatten)