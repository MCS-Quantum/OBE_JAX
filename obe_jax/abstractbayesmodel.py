import jax.numpy as jnp
from jax import jit, vmap, random, lax

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
        self.lower_kwargs = kwargs
        ParticlePDF.__init__(self, key, particles, weights, **kwargs)

        self.likelihood_function = likelihood_function # takes (oneinput_vec,oneoutput_vec,oneparameter_vec)
        self.oneinput_multioutput_oneparam = jit(vmap(likelihood_function,in_axes=(None,1,None)))
        self.oneinput_oneoutput_multiparams = jit(vmap(likelihood_function,in_axes=(None,None,1))) # takes (oneinput_vec, oneoutput_vec, multi_param_vec)   
        self.oneinput_multioutput_multiparams = jit(vmap(self.oneinput_oneoutput_multiparams,in_axes = (None,1,None), out_axes=1))# takes (oneinput, multioutput, multiparameters)   
        self.utility_measure = utility_measure
        self.multioutput_utility = jit(vmap(self.utility_measure,in_axes=(None,None,1)))
        self.expected_outputs = jnp.asarray(expected_outputs)
        try:
            self.num_expected_outputs = expected_outputs.shape[1]
        except:
            self.num_expected_outputs = 0

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
    def _expected_utility(self,oneinput,particles,weights):
        # Compute a matrix of likelihoods for various output/parameter combinations. 
        ls = self.oneinput_multioutput_multiparams(oneinput,self.expected_outputs,particles) # shape n_particles x n_outputs
        # This gives the probability of measuring various outputs/parameters for a single input
        us = self.multioutput_utility(particles,weights,ls)

        return jnp.sum(jnp.dot(ls,us))
    
    @jit
    def _expected_utilities(self,inputs,particles,weights):
        umap = jit(vmap(self._expected_utility,in_axes=(1,None,None)))
        return umap(inputs,particles,weights)
         
    @jit
    def expected_utility(self,oneinput):
        # Compute a matrix of likelihoods for various output/parameter combinations. 
        return self._expected_utility(oneinput,self.particles,self.weights)
    
    @jit
    def expected_utilities(self,inputs):
        umap = jit(vmap(self.expected_utility,in_axes=(1,)))
        return umap(inputs)

    def expected_utility_k_particles(self,oneinput,k=10):
        # Compute a matrix of likelihoods for various output/parameter combinations. 
        weights, inds = lax.top_k(self.weights,k)
        particles = self.particles[:,inds]
        return self._expected_utility(oneinput,particles,weights)
    
    def expected_utilities_k_particles(self,inputs,k=10):
        weights, inds = lax.top_k(self.weights,k)
        particles = self.particles[:,inds]
        return self._expected_utilities(inputs,particles,weights)
    
    def sample_output(self,oneinput,oneparam):
        # This can definitely be re-written to parallelize the computation of likelihoods up-front 
        # but creating a synthetic dataset doesn't really need to be fast at the moment.
        ls = self.oneinput_multioutput_oneparam(oneinput,self.expected_outputs,oneparam)
        key, subkey = random.split(self.key)
        self.key = key
        output = random.choice(subkey,self.expected_outputs,p=ls,axis=1)
        return output
    
    def sample_outputs(self, inputs, oneparam):
        # This can definitely be re-written to parallelize the computation of likelihoods up-front 
        # but creating a synthetic dataset doesn't really need to be fast at the moment.
        num_inputs = inputs.shape[1]
        return jnp.hstack([self.sample_output(inputs[:,i],oneparam) for i in range(num_inputs)])
        
    def _tree_flatten(self):
        children = (self.key, self.particles, self.weights)  # arrays / dynamic values
        aux_data = {'likelihood_function':self.likelihood_function,
                    'utility_measure':self.utility_measure,
                    'expected_outputs':self.expected_outputs,**self.lower_kwargs}
        return (children, aux_data)
    
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children,**aux_data)
    
from jax import tree_util
tree_util.register_pytree_node(AbstractBayesianModel,
                               AbstractBayesianModel._tree_flatten,
                               AbstractBayesianModel._tree_unflatten)