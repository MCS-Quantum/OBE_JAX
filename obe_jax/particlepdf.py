import jax.numpy as jnp
from jax import random, jit
from obe_jax.samplers import sample, Liu_West_resampler


class ParticlePDF:
    """A probability distribution function.

    A probability distribution :math:`P(\\theta_0, \\theta_1, \\ldots,
    \\theta_{n\_dims})` over parameter variables :math:`\\theta_i` is
    represented by a large-ish number of samples from the distribution,
    each with a weight value.  The distribution can be visualized as a cloud
    of particles in parameter space, with each particle corresponding to a
    weighted random draw from the distribution.  The methods implemented
    here largely follow the algorithms published in Christopher E Granade et
    al. 2012 *New J. Phys.* **14** 103013.

    Warnings:

        The number of samples (i.e. particles) required for good performance
        will depend on the application.  Too many samples will slow down the
        calculations, but too few samples can produce incorrect results.
        With too few samples, the probability distribution can become overly
        narrow, and it may not include the "true" parameter values. See the
        ``resample()`` method documentation for details.

    Arguments:
        key (:obj:`jax.PRNGKey`):
            The pseudo-random number generator key used to seed all 
            random functions'

        particles (:obj:`2D array-like`):
            The Bayesian *prior*, which initializes the :obj:`ParticlePDF`
            distribution. Each of ``n_dims`` sub-arrays contains
            ``n_particles`` values of a single parameter, so that the *j*\
            _th elements of the sub-arrays determine the coordinates of a
            point in parameter space. Users are encouraged to experiment with
            different ``n_particles`` sizes to assure consistent results.
            
        weights (:obj:`2D array-like`):
            The Bayesian *prior*, which initializes the :obj:`ParticlePDF`
            distribution. Each of ``n_dims`` sub-arrays contains
            ``n_particles`` values of a single parameter, so that the *j*\
            _th elements of the sub-arrays determine the coordinates of a
            point in parameter space. Users are encouraged to experiment with
            different ``n_particles`` sizes to assure consistent results.

    Keyword Args:

        TBD

    **Attributes:**
    """

    def __init__(self, key, particles, weights, resampler=Liu_West_resampler,
                 tuning_parameters = {'resample_threshold':0.5,'auto_resample':True},
                 resampling_parameters = {'a':0.98, 'scale':True}, 
                 just_resampled=False, **kwargs):
        
        # The jax.random.PRNGkey() for random number sampling
        self.key = key
        
        #: ``n_dims x n_particles ndarray`` of ``float64``: Together with
        #: ``weights``,#: these ``n_particles`` points represent
        #: the parameter probability distribution. Initialized by the
        #: ``prior`` argument.
        self.particles = jnp.asarray(particles).copy()
        
        #: ndarray of ``float64``: Array of probability weights
        #: corresponding to the particles.
        self.weights = jnp.asarray(weights).copy()
        
        #: ``int``: the number of parameter samples representing the
        #: probability distribution. Determined from the trailing dimension
        #: of ``prior``.
        self.n_particles = self.particles.shape[1 ]

        #: ``int``: The number of parameters, i.e. the dimensionality of
        #: parameter space. Determined from the leading dimension of ``prior``.
        self.n_dims = self.particles.shape[0]
        
        self.resampler=resampler

        #: dict: A package of parameters affecting the resampling algorithm
        #:
        #:     - ``'resample_threshold'`` (:obj:`float`):  Initially,
        #:       the value of the ``resample_threshold`` keyword argument.
        #:       Default ``0.5``.
        #:
        #:     - ``'auto_resample'`` (:obj:`bool`): Initially, the value of the
        #:       ``auto_resample`` keyword argument. Default ``True``.
        self.tuning_parameters = tuning_parameters
        self.resampling_parameters = resampling_parameters

        #: ``bool``: A flag set by the ``resample_test()`` function. ``True`` if
        #: the last ``bayesian_update()`` resulted in resampling,
        #: else ``False``.
        self.just_resampled = just_resampled

    @jit
    def mean(self):
        """Calculates the mean of the probability distribution.

        The weighted mean of the parameter distribution. See also
        :obj:`std()` and :obj:`covariance()`.

        Returns:
            Size ``n_dims`` array.
        """
        return jnp.average(self.particles, axis=1,
                          weights=self.weights)
    @jit
    def covariance(self):
        """Calculates the covariance of the probability distribution.

        Returns:
            The covariance of the parameter distribution as an
            ``n_dims`` X ``n_dims`` array. See also :obj:`mean()` and
            :obj:`std()`.
        """
        n_dims = self.particles.shape[0]
        raw_covariance = jnp.cov(self.particles, aweights=self.weights)
        if n_dims == 1:
            return raw_covariance.reshape((1, 1))
        else:
            return raw_covariance
        
#     def std(self):
#         """Calculates the standard deviation of the distribution.

#         Calculates the square root of the diagonal elements of the
#         covariance matrix.  See also :obj:`covariance()` and :obj:`mean()`.

#         Returns:
#             The standard deviation as an n_dims array.
#         """
#         n_dims = self.particles.shape[0]
#         var = jnp.zeros(n_dims)
#         for i, p in enumerate(self.particles):
#             mean = np.dot(p, self.weights)
#             msq = np.dot(p*p, self.weights)
#             var[i] = msq - mean ** 2
#         return np.sqrt(var)


    @jit
    def update_weights(self, likelihood):
        """Performs a Bayesian update on the probability distribution.

        Multiplies ``weights`` by the ``likelihood`` and
        renormalizes the probability
        distribution.  After the update, the distribution is tested for
        resampling depending on
        ``self.tuning_parameters['auto_resample']``.

        Args:
            likelihood: (:obj:`ndarray`):  An ``n_samples`` sized array
                describing the Bayesian likelihood of a measurement result
                calculated for each parameter combination.
         """
        weights = (likelihood*self.weights)
        return weights/jnp.sum(weights)
        

    def bayesian_update(self, likelihood):
        """Performs a Bayesian update on the probability distribution.

        Multiplies ``weights`` by the ``likelihood`` and
        renormalizes the probability
        distribution.  After the update, the distribution is tested for
        resampling depending on
        ``self.tuning_parameters['auto_resample']``.

        Args:
            likelihood: (:obj:`ndarray`):  An ``n_samples`` sized array
                describing the Bayesian likelihood of a measurement result
                calculated for each parameter combination.
         """
        self.weights = self.update_weights(likelihood)
        
        if self.tuning_parameters['auto_resample']:
            self.resample_test()
    
    @jit
    def n_eff(self):
        wsquared = jnp.square(self.weights)
        return 1 / jnp.sum(wsquared)
    
    def resample_test(self):
        """Tests the distribution and performs a resampling if required.

        If the effective number of particles falls below
        ``self.tuning_parameters['resample_threshold'] * n_particles``,
        performs a resampling.  Sets the ``just_resampled`` flag.
        """
        threshold = self.tuning_parameters['resample_threshold']
        n_eff = self.n_eff()
        if n_eff / self.n_particles < threshold:
            key, subkey = random.split(self.key)
            self.particles, self.weights = self.resample(subkey)
            self.key = key
            self.just_resampled = True
        else:
            self.just_resampled = False
        
    @jit
    def resample(self,key):
        """Performs a resampling of the distribution as specified by 
        and self.resampler and self.resampler_params.

        Resampling refreshes the random draws that represent the probability
        distribution.  As Bayesian updates are made, the weights of
        low-probability particles can become very small.  These particles
        consume memory and computation time, and they contribute little to
        values that are determined from the distribution.  Resampling
        abandons some low-probability particles while allowing
        high-probability particles to multiply in higher-probability regions.

        *Sample impoverishment* can occur if there are too few particles. In
        this phenomenon, a resampling step fails to sample particles from an
        important, but low-probability region, effectively removing that
        region from future consideration. The symptoms of this ``sample
        impoverishment`` phenomenon include:

            - Inconsistent results from repeated runs.  Standard deviations
              from individual final distributions will be too small to
              explain the spread of individual mean values.

            - Sudden changes in the standard deviations or other measures of
              the distribution on resampling. The resampling is not
              *supposed* to change the distribution, just refresh its
              representation.
        """

        # Call the resampler function to get a new set of particles
        new_particles = self.resampler(key, self.particles, self.weights, **self.resampling_parameters)
        # Re-fill the current particle weights with 1/n_particles
        new_weights = jnp.full(self.n_particles, 1/self.n_particles)
        return new_particles, new_weights

    def randdraw(self, n_draws=1):
        """Provides random parameter draws from the distribution

        Particles are selected randomly with probabilities given by
        ``self.weights``.

        Args:
            n_draws (:obj:`int`): the number of draws requested.  Default
              ``1``.

        Returns:
            An ``n_dims`` x ``N_DRAWS`` :obj:`ndarray` of parameter draws.
        """
        key, subkey = random.split(self.key)
        self.key = key
        return sample(subkey, self.particles, self.weights, n=n_draws)
    
    
    def _tree_flatten(self):
        children = (self.key, self.particles, self.weights)  # arrays / dynamic values
        aux_data = {'n_particles':self.n_particles, 
                    'resampler':self.resampler,
                    'n_dims':self.n_dims, 
                    'tuning_parameters': self.tuning_parameters,
                    'resampling_parameters':self.resampling_parameters,
                    'just_resampled':self.just_resampled}  # static values
        return (children, aux_data)
    
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children,**aux_data)

# end ParticlePDF definition


from jax import tree_util
tree_util.register_pytree_node(ParticlePDF,
                               ParticlePDF._tree_flatten,
                               ParticlePDF._tree_unflatten)