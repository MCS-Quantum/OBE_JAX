import jax.numpy as jnp
from jax import jit, vmap

@jit
def diffable_plogp(p):
    lp = jnp.log2(jnp.where(p>0,p,1))
    return lp*p

diffable_plogp_vec = jit(vmap(diffable_plogp,in_axes=(0,)))

@jit
def entropy_change(current_particles,current_weights,likelihoods):
    new_weights = current_weights*likelihoods
    H_old = jnp.sum(diffable_plogp_vec(current_weights))
    H_new = jnp.sum(diffable_plogp_vec(new_weights))
    return H_new-H_old