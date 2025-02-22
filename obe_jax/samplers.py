import numpy as np
from jax import random, jit
import jax.numpy as jnp
from functools import partial

@partial(jit, static_argnames=['n'])
def sample_inds(key, particles, weights, n=1):
    """Provides an integer index that produces random samples from a particle distribution.
    
        Particles are selected randomly with probabilities given by 
        ``weights``.

    Parameters
    ----------
    key : jax.random.PRNGKey
        The pseudo-random number generator key used to seed all 
        jax random functions.
    particles : Array
        The set of particles initializes to initialize the distribution.
        Has size ``n_dims``x``n_particles``. 
    weights : Array
        The set of weights for each particle. Has size (``n_particles``,). 
    n : int, optional
        The number of indices to return, by default 1

    Returns
    -------
    Vector, Int
        Returns the indices of the particle to be sampled.
    """    
    num_particles = particles.shape[1]
    I = random.choice(key,num_particles,shape=(n,),p=weights)
    return I


@partial(jit, static_argnames=['n'])
def sample_particles(key, particles, weights, n=1):
    """Provides an new array of particles representing samples
     from the particle distribution.
    
    Particles are selected randomly with probabilities given by 
    ``weights``.

    Parameters
    ----------
    key : jax.random.PRNGKey
        The pseudo-random number generator key used to seed all 
        jax random functions.
    particles : Array
        The set of particles initializes to initialize the distribution.
        Has size ``n_dims``x``n_particles``. 
    weights : Array
        The set of weights for each particle. Has size (``n_particles``,). 
    n : int, optional
        The number of sampled particles to return, by default 1

    Returns
    -------
    Array
        An array of particles. 
    """    
    I = sample_inds(key, particles, weights, n=n)
    return particles[:,I]

@partial(jit, static_argnames=['a','scale'])
def Liu_West_resampler(key, particles, weights, a=0.98, scale=True):
    """Provides an new array of particles that have been
    resampled according to the Liu-West algorithm.

    Parameters
    ----------
    key : jax.random.PRNGKey
        The pseudo-random number generator key used to seed all 
        jax random functions.
    particles : Array
        The set of particles initializes to initialize the distribution.
        Has size ``n_dims``x``n_particles``. 
    weights : Array
        The set of weights for each particle. Has size (``n_particles``,). 
    a : Float, optional
        Determines the spread of the newly sampled particles about the previous 
        particle locations. 
    scale : bool, optional
        Determines whether or not the newly sampled distribution is 
        contracted around the mean of the distribution to adjust for increased
        variance during the resampling. 

    Returns
    -------
    Array
        An array of particles. 
    """    
    ndim, num_particles = particles.shape
    origin = jnp.zeros(ndim)
    # coords is n_dims x n_particles
    key1, key2 = random.split(key)
    coords = sample_particles(key1, particles, weights, n=num_particles).T
    scaled_mean = jnp.average(particles, axis=1, weights=weights)* (1 - a)
    # newcovar is a small version of covar that determines the size of
    # the nudge.
    newcovar = (1-a**2)*jnp.cov(particles, aweights=weights, ddof=0)
    # multivariate normal returns n_particles x n_dims array. ".T"
    # transposes to match coords shape.
    nudged = coords + random.multivariate_normal(key2, origin, newcovar,
                                                       shape=(num_particles,))
    
    if scale:
            nudged = nudged * a
            nudged = nudged + scaled_mean

    return nudged.T
