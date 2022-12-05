from dataclasses import dataclass, field
from typing import Callable, Tuple

from icecream import ic
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.random import PRNGKey, split
from jaxtyping import Array  # type: ignore


def fori_loop(lower, upper, body_fun, init_val):
    val = init_val
    for i in range(lower, upper):
        val = body_fun(i, val)
    return val


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
    delta: float = field(init=False)
    """Timestep size.
    """
    shape: Tuple[int, ...] = field(init=False)
    """Data shape.
    """

    def __post_init__(self):
        self.delta = self.T / self.n_timesteps
        self.shape = self.pi_0.sample(PRNGKey(0)).shape

    def get_F_k(self, t, x_km1):
        score_pi = self.get_score_pi(t, x_km1)
        mean = x_km1 + self.delta * score_pi
        stdev = jnp.sqrt(2 * self.delta)
        return dist.Normal(mean, stdev).to_event(x_km1.ndim)

    # def _train_diffuse_iter(self, key, k, x_km1, model):
    def _train_iter(self, k, val, model: Callable[[Array, Array], Array]):
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

    def get_loss(self, model: Callable[[Array, Array], Array], key):
        """
        Computes the loss for a single sample from the diffusion process.
        """
        key_0, key_traj = split(key)
        x_0 = self.pi_0.sample(key_0)
        init_val = (key_traj, x_0, 0.0)
        body_fn = lambda k, val: self._train_iter(k, val, model)
        _, _, loss = jax.lax.fori_loop(1, self.n_timesteps, body_fn, init_val)
        # _, _, loss = fori_loop(1, self.n_timesteps, body_fn, init_val)
        return loss

    def _sample_iter(self, k, key, x_km1, log_w, get_B_km1):
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

    def get_B_km1_ais(self, t, x_k):
        return self.get_F_k(t, x_k)

    def get_B_km1_mcd(self, t, x_k, model: Callable[[Array, Array], Array]):
        score_pi = self.get_score_pi(t, x_k)
        s = model(t, x_k)
        mean = x_k - self.delta * score_pi + 2 * self.delta * s
        stdev = jnp.sqrt(2 * self.delta)
        return dist.Normal(mean, stdev).to_event(x_k.ndim)

    def _get_sample_helper(
        self, key, get_B_km1: Callable[[Array, Array], dist.Distribution]
    ):
        """
        Helper to generate a sample with an arbitrary backwards kernel.
        """
        key_0, key_traj = split(key)
        x_0 = self.pi_0.sample(key_0)
        init_val = (key_traj, x_0, -self.pi_0.log_prob(x_0))

        def body_fn(k: int, val: Tuple[Array, Array, Array]):
            return self._sample_iter(k, *val, get_B_km1)

        _, x_K, log_w = jax.lax.fori_loop(1, self.n_timesteps, body_fn, init_val)
        log_w += self.get_log_gamma(x_K)
        return x_K, log_w

    def get_sample_ais(self, key):
        """
        Generates a sample and its log importance weight with AIS.
        """
        return self._get_sample_helper(key, self.get_B_km1_ais)

    def get_sample_mcd(self, model, key):
        """
        Generates a sample and its log importance weight with MCD.
        """
        get_B_km1 = lambda t, x_k: self.get_B_km1_mcd(t, x_k, model)
        return self._get_sample_helper(key, get_B_km1)

    def _get_trajectory_helper(self, key, get_B_km1) -> Tuple[Array, Array]:
        key_0, key_traj = split(key)
        x_0 = self.pi_0.sample(key_0)
        log_w_0 = -self.pi_0.log_prob(x_0)
        init = (key_traj, x_0, log_w_0)
        ks = jnp.arange(1, self.n_timesteps, 1)

        def f(carry: Tuple[Array, Array, Array], k: Array):
            carry = self._sample_iter(k, *carry, get_B_km1)
            x_k = carry[1]
            return carry, x_k

        (_, _, log_w), xs = jax.lax.scan(f, init, ks)
        log_w += self.get_log_gamma(xs[-1])
        return xs, log_w

    def get_trajectory_ais(self, key) -> Tuple[Array, Array]:
        """
        Generates a trajectory and its log importance weight with AIS.
        """
        return self._get_trajectory_helper(key, self.get_B_km1_ais)

    def get_trajectory_mcd(self, model, key) -> Tuple[Array, Array]:
        """
        Generates a trajectory and its log importance weight with MCD.
        """
        get_B_km1 = lambda t, x_k: self.get_B_km1_mcd(t, x_k, model)
        return self._get_trajectory_helper(key, get_B_km1)
