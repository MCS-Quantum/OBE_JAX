import jax.numpy as jnp
from jax import jit

@jit
def entropy_change(current_particles,current_weights,likelihoods):
    new_weights = current_weights*likelihoods
    new_weights = new_weights/jnp.sum(new_weights)
    H_old = jnp.nansum(current_weights*jnp.log2(current_weights))
    H_new = jnp.nansum(new_weights*jnp.log2(new_weights))
    return H_new-H_old

@jit
def relative_entropy(current_particles,current_weights,likelihoods):
    new_weights = current_weights*likelihoods
    new_weights = new_weights/jnp.sum(new_weights)
    log_rel = jnp.log2(jnp.divide(current_weights,new_weights))
    return -jnp.nansum(new_weights*log_rel)

@jit
def posterior_variance(current_particles,current_weights,likelihoods):
    new_weights = current_weights*likelihoods
    new_weights = new_weights/jnp.sum(new_weights)
    raw_covariance = jnp.cov(current_particles, aweights=new_weights)
    return jnp.trace(raw_covariance)