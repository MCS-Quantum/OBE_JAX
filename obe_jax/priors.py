from jax import random
import jax.numpy as jnp

def uniform_prior_particles(key, minimums, maximums, N):
    newkey, subkey = random.split(key)
    n_params = len(minimums)
    return random.uniform(subkey,(n_params,N),
                          minval=jnp.asarray(minimums).reshape(n_params,1),
                          maxval=jnp.asarray(maximums).reshape(n_params,1))
