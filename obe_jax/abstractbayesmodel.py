import jax.numpy as jnp
from jax import jit, vmap, random

from obe_jax import ParticlePDF
from obe_jax.utility_measures import entropy_change

class AbstractBayesianModel(ParticlePDF):
    """An abstract Bayesian probabilistic model for a system with 
    unknown parameters. This class defines a Bayesian model for a 
    system with oututs predictable from a likelihood function
    that is paramaterized by a set of underlying parameters 
    which are inteded to be estimated. 

   This abstract class implements the basic methods needed to 
   sequentially update the probabilistic model from new measurements
   and compute utilities of future experimental inputs.

    """

    def __init__(self, key, particles, weights, 
                 likelihood_function=None, expected_outputs=None, utility_measure=entropy_change,
                 **kwargs):
        
        ParticlePDF.__init__(self, key, particles, weights, **kwargs)

        self.likelihood_function = likelihood_function # takes (oneinput_vec,oneoutput_vec,oneparameter_vec)
        self.oneinput_multioutput_oneparam = jit(vmap(likelihood_function,in_axes=(None,1,None)))
        self.oneinput_oneoutput_multiparams = jit(vmap(likelihood_function,in_axes=(None,None,1))) # takes (oneinput_vec, oneoutput_vec, multi_param_vec)   
        self.oneinput_multioutput_multiparams = jit(vmap(self.oneinput_oneoutput_multiparams,in_axes = (None,1,None), out_axes=1))# takes (oneinput, multioutput, multiparameters)   
        self.utility_measure = utility_measure
        self.multioutput_utility = jit(vmap(self.utility_measure,in_axes=(None,None,1)))
        self.expected_outputs = jnp.asarray(expected_outputs)
        self.num_expected_outputs = expected_outputs.shape[1]

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
    
    def sample_output(self,oneinput,oneparam):
        ls = self.oneinput_multioutput_oneparam(oneinput,self.expected_outputs,oneparam)
        key, subkey = random.split(self.key)
        self.key = key
        output = random.choice(subkey,self.expected_outputs,p=ls[:,0],axis=1)
        return output
    
    def sample_outputs(self, inputs, oneparam):
        num_inputs = inputs.shape[1]
        return jnp.hstack([self.sample_output(inputs[:,i],oneparam) for i in range(num_inputs)])
        
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
tree_util.register_pytree_node(AbstractBayesianModel,
                               AbstractBayesianModel._tree_flatten,
                               AbstractBayesianModel._tree_unflatten)