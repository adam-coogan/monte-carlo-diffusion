from dataclasses import dataclass, field
from typing import Callable, Tuple, Union

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.random import PRNGKey, PRNGKeyArray, split
from jaxtyping import Array  # type: ignore


@dataclass
class UnadjustedLangevin:
    """
    MCD and AIS for the time-inhomogeneous unadjusted Langevin algorithm.
    """

    pi_0: dist.Distribution
    """Prior distribution for :math:`x` at time :math:`t=0`.
    """
    get_log_gamma: Callable[[Array], Array]
    """Gets the target log probability (potentially unnormalized).
    """
    get_score_pi: Callable[[Array, Array], Array]
    """Gets :math:`\nabla \log \pi(t, x)`.
    """
    n_timesteps: int
    T: float = 1.0
    """Duration of diffusion process.
    """
    delta: Array = field(init=False)
    """Timestep size.
    """
    shape: Tuple[int, ...] = field(init=False)
    """Data shape.
    """

    def __post_init__(self):
        self.delta = jnp.array(self.T / self.n_timesteps)
        self.shape = self.pi_0.sample(PRNGKey(0)).shape

    def get_F_k(self, t: Array, x_km1: Array) -> dist.Distribution:
        """
        Gets the forward Langevin kernel.
        """
        score_pi = self.get_score_pi(t, x_km1)
        mean = x_km1 + self.delta * score_pi
        stdev = jnp.sqrt(2 * self.delta)
        return dist.Normal(mean, stdev).to_event(x_km1.ndim)

    # def _train_diffuse_iter(self, key, k, x_km1, model):
    def _train_iter(
        self,
        k: float,
        val: Tuple[PRNGKeyArray, Array, Array],
        model: Callable[[Array, Array], Array],
    ) -> Tuple[PRNGKeyArray, Array, Array]:
        """
        Runs a single step of the training diffusion to get the contribution to
        the loss going from step k-1 to k.
        """
        key, x_km1, loss = val
        key, subkey = split(key)
        t = k * self.delta

        # Sample x
        F_k = self.get_F_k(t, x_km1)
        x_k = F_k.sample(subkey)

        # Update loss
        s = model(t, x_k)
        score_F_k = jax.grad(lambda x: F_k.log_prob(x).sum())(x_k)
        loss += self.delta * ((s - score_F_k) ** 2).sum()

        return key, x_k, loss

    def get_loss(
        self, model: Callable[[Array, Array], Array], key: PRNGKeyArray
    ) -> Array:
        """
        Computes the loss for a single sample from the diffusion process.
        """
        key_0, key_traj = split(key)
        x_0 = self.pi_0.sample(key_0)
        init_val = (key_traj, x_0, 0.0)
        body_fn = lambda k, val: self._train_iter(k, val, model)
        _, _, loss = jax.lax.fori_loop(1, self.n_timesteps + 1, body_fn, init_val)
        return loss

    def _sample_iter(
        self,
        k: Array,
        key: PRNGKeyArray,
        x_km1: Array,
        log_w: Array,
        get_B_km1: Callable[[Array, Array], dist.Distribution],
    ) -> Tuple[PRNGKeyArray, Array, Array]:
        """
        Runs a single iteration of sampling.
        """
        key, subkey = split(key)
        t = k * self.delta

        # Sample x
        F_k = self.get_F_k(t, x_km1)
        x_k = F_k.sample(subkey)

        # Compute weight
        B_km1 = get_B_km1(t, x_k)
        log_w += B_km1.log_prob(x_km1).sum() - F_k.log_prob(x_k).sum()

        return key, x_k, log_w

    def get_B_km1_ais(self, t: Array, x_k: Array) -> dist.Distribution:
        """
        Gets the AIS backward proposal.
        """
        return self.get_F_k(t, x_k)

    def get_B_km1_mcd(
        self, t: Array, x_k: Array, model: Callable[[Array, Array], Array]
    ) -> dist.Distribution:
        """
        Gets the MCD backward kernel.
        """
        score_pi = self.get_score_pi(t, x_k)
        s = model(t, x_k)
        mean = x_k - self.delta * score_pi + 2 * self.delta * s
        stdev = jnp.sqrt(2 * self.delta)
        return dist.Normal(mean, stdev).to_event(x_k.ndim)

    def _get_sample_helper(
        self, key: PRNGKeyArray, get_B_km1: Callable[[Array, Array], dist.Distribution]
    ) -> Tuple[Array, Array]:
        """
        Helper to generate a sample and its weight with an arbitrary backwards kernel.
        """
        key_0, key_traj = split(key)
        x_0 = self.pi_0.sample(key_0)
        init_val = (key_traj, x_0, -self.pi_0.log_prob(x_0))

        def body_fn(k: Array, val: Tuple[Array, Array, Array]):
            return self._sample_iter(k, *val, get_B_km1)

        _, x_K, log_w = jax.lax.fori_loop(1, self.n_timesteps + 1, body_fn, init_val)
        log_w += self.get_log_gamma(x_K)
        return x_K, log_w

    def get_sample_ais(self, key: PRNGKeyArray) -> Tuple[Array, Array]:
        """
        Generates a sample and its log importance weight using AIS.
        """
        return self._get_sample_helper(key, self.get_B_km1_ais)

    def get_sample_mcd(
        self, model: Callable[[Array, Array], Array], key: PRNGKeyArray
    ) -> Tuple[Array, Array]:
        """
        Generates a sample and its log importance weight using MCD.
        """
        get_B_km1 = lambda t, x_k: self.get_B_km1_mcd(t, x_k, model)
        return self._get_sample_helper(key, get_B_km1)

    def _get_trajectory_helper(
        self, key: PRNGKeyArray, get_B_km1: Callable[[Array, Array], dist.Distribution]
    ) -> Tuple[Array, Array]:
        """
        Helper to generate a trajectory and its weight with an arbitrary backwards kernel.
        """
        key_0, key_traj = split(key)
        x_0 = self.pi_0.sample(key_0)
        log_w_0 = -self.pi_0.log_prob(x_0)
        init = (key_traj, x_0, log_w_0)
        ks = jnp.arange(1, self.n_timesteps, 1)

        def f(
            carry: Tuple[Array, Array, Array], k: Array
        ) -> Tuple[Tuple[Array, Array, Array], Array]:
            carry = self._sample_iter(k, *carry, get_B_km1)
            x_k = carry[1]
            return carry, x_k

        (_, _, log_w), xs = jax.lax.scan(f, init, ks)
        log_w += self.get_log_gamma(xs[-1])
        return xs, log_w

    def get_trajectory_ais(self, key: PRNGKeyArray) -> Tuple[Array, Array]:
        """
        Generates a trajectory and its log importance weight with AIS.
        """
        return self._get_trajectory_helper(key, self.get_B_km1_ais)

    def get_trajectory_mcd(
        self, model: Callable[[Array, Array], Array], key: PRNGKeyArray
    ) -> Tuple[Array, Array]:
        """
        Generates a trajectory and its log importance weight with MCD.
        """
        get_B_km1 = lambda t, x_k: self.get_B_km1_mcd(t, x_k, model)
        return self._get_trajectory_helper(key, get_B_km1)
