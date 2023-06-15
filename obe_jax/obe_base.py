import jax.numpy as jnp
from jax import jit, vmap

from obe_jax import ParticlePDF
from obe_jax.utility_measures import entropy_change

class OBE_PDF(ParticlePDF):
    """An implementation of sequential Bayesian experiment design.

    OptBayesExpt is a manager that calculates strategies for efficient
    measurement runs. OptBayesExpt incorporates measurement data, and uses
    that information to select inputs for measurements with high
    predicted benefit / cost ratios.

    The use cases are situations where the goal is to find the parameters of
    a parametric model.

    The primary functions of this class are to interpret measurement data
    and to calculate effective inputs.

        \*\*kwargs: Keyword arguments passed to the parent ParticlePDF class.

    **Attributes:**
    """

    def __init__(self, key, particles, weights, 
                 likelihood_function=None, utility_measure=entropy_change, expected_outputs=None,
                 **kwargs):
        
        ParticlePDF.__init__(self, key, particles, weights, **kwargs)

        # Test if there is a precompute function and set a flag
        self.likelihood_function = likelihood_function # takes (oneinput_vec,oneoutput_vec,oneparameter_vec)
        oneinput_oneoutput_multiparams = jit(vmap(likelihood_function,in_axes=(None,None,1))) # takes (oneinput_vec, oneoutput_vec, multi_param_vec)   
        self.oneinput_oneoutput_multiparams = oneinput_oneoutput_multiparams
        self.oneinput_multioutput_multiparams = jit(vmap(oneinput_oneoutput_multiparams,in_axes = (None,1,None), out_axes=1))# takes (oneinput, multioutput, multiparameters)   
        self.utility_measure = utility_measure
        self.multioutput_utility = jit(vmap(self.utility_measure,in_axes=(None,None,1)))
        self.expected_outputs = jnp.asarray(expected_outputs)

    @jit
    def updated_weights_from_experiment(self, oneinput, oneoutput):
        ls = self.oneinput_oneoutput_multiparams(oneinput, oneoutput, self.particles)
        weights = self.update_weights(ls)
        return weights
    
    def bayesian_update(self, oneinput, oneoutput):
        """
        Refines the parameters' probability distribution function given a
        measurement result.

        This is where measurement results are entered. An implementation of
        Bayesian inference, uses the model to calculate the likelihood of
        obtaining the measurement result as a function of
        parameter values, and uses that likelihood to generate a refined
        *posterior* (after-measurement) distribution from the *prior* (
        pre-measurement) parameter distribution.

        """
        self.weights = self.updated_weights_from_experiment(oneinput, oneoutput)
        
        if self.tuning_parameters['auto_resample']:
            self.resample_test()
     
    @jit
    def expected_utility(self,oneinput):
        # Compute a matrix of likelihoods for various output/parameter combinations. 
        ls = self.oneinput_multioutput_multiparams(oneinput,self.expected_outputs,self.particles) # shape n_particles x n_outputs
        # This gives the probability of measuring various outputs/parameters for a single input
        us = self.multioutput_utility(self.particles,self.weights,ls)

        return jnp.sum(jnp.dot(ls,us))
    
    @jit
    def expected_utilities(self,inputs):
        umap = jit(vmap(self.expected_utility,in_axes=(1,)))
        return umap(inputs)
        
    def _tree_flatten(self):
        children = (self.key, self.particles, self.weights)  # arrays / dynamic values
        aux_data = {'likelihood_function':self.likelihood_function, 
                    'oneinput_oneoutput_multiparams':self.oneinput_oneoutput_multiparams,
                    'oneinput_multioutput_multiparams':self.oneinput_multioutput_multiparams, 
                    'utility_measure':self.utility_measure,
                    'multioutput_utility': self.multioutput_utility,
                    'expected_outputs':self.expected_outputs}
        return (children, aux_data)
    
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children,**aux_data)
    
from jax import tree_util
tree_util.register_pytree_node(OBE_PDF,
                               OBE_PDF._tree_flatten,
                               OBE_PDF._tree_unflatten)