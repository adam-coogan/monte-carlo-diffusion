from typing import Callable, List

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.random import PRNGKeyArray, split
from jaxtyping import Array  # type: ignore


class MCDNetResBlock(eqx.Module):
    """
    Residual MLP block from `Score-Based Diffusion meets Annealed Importance Sampling <https://arxiv.org/abs/2208.07698>`_.
    """

    d_t: int = eqx.static_field()
    d_h: int = eqx.static_field()
    norm: eqx.nn.LayerNorm
    t_emb_proj: eqx.nn.Linear
    h_proj: eqx.nn.Linear
    act: Callable[[Array], Array]
    deproj: eqx.nn.Linear

    def __init__(self, d_h, d_t, act=jax.nn.swish, *, key):
        super().__init__()
        self.d_t = d_t
        self.d_h = d_h
        self.act = act

        keys = split(key, 3)
        self.norm = eqx.nn.LayerNorm(d_h)
        self.t_emb_proj = eqx.nn.Linear(d_t, 2 * d_h, key=keys[1])
        self.h_proj = eqx.nn.Linear(d_h, 2 * d_h, key=keys[0])
        self.deproj = eqx.nn.Linear(2 * d_h, d_h, key=keys[2])

    def __call__(self, t_emb, h):
        h = self.norm(h)
        h = self.act(h)
        h = self.h_proj(h)
        h += self.t_emb_proj(t_emb)
        h = self.act(h)
        return self.deproj(h)


class MCDNet(eqx.Module):
    """
    Residual MLP network similar to the one used in the experiments in `Score-Based Diffusion meets Annealed Importance Sampling <https://arxiv.org/abs/2208.07698>`_.
    Parametrized as :math:`c_{\\theta}(t, x) + (1 + s_{\\theta}(t, x)) \\nabla \\log \\gamma_t(x)`.

    """

    x_dim: int = eqx.static_field()
    get_score_gamma_t: Callable[[Array, Array], Array]
    d_t: int = eqx.static_field()
    d_h: int = eqx.static_field()
    depth: int = eqx.static_field()
    act: Callable[[Array], Array]
    # Layers
    t_emb: eqx.nn.Linear
    x_emb: eqx.nn.Linear
    res_layers: List[MCDNetResBlock]
    final_layer: eqx.nn.Linear
    const_scale: Array
    """constant multiplying the :math:`c_{\\theta}(t, x)` network, initialized to zero.
    """
    score_scale: Array
    """constant multiplying the :math:`s_{\\theta}(t, x)` network, initialized to zero.
    """

    def __init__(
        self,
        x_dim: int,
        get_score_gamma_t: Callable[[Array, Array], Array],
        d_t: int = 4,
        d_h: int = 64,
        depth: int = 3,
        act: Callable[[Array], Array] = jax.nn.swish,
        *,
        key: PRNGKeyArray
    ):
        """
        Initializes the network with the output set to :math:`\\gamma_t(x)`.

        Args:
            x_dim: dimensionality of `x`.
            get_score_gamma_t: :math:`\\gamma_t(x)`, which is used to initialize
                the network's output to the standard AIS backward kernel.
            d_t: dimensionality of `t` embedding.
            d_h: dimensionalixy of `x` embedding.
            depth: number of residual MLP blocks.
            act: activation function.
            key: PRNG key used to initialize layers.

        """
        super().__init__()
        self.x_dim = x_dim
        self.get_score_gamma_t = get_score_gamma_t
        self.d_t = d_t
        self.d_h = d_h
        self.depth = depth
        self.act = act

        # Embedding layers
        key_t, key_x, key_final, *keys_res = split(key, 3 + depth)
        self.t_emb = eqx.nn.Linear(1, d_t, key=key_t)
        self.x_emb = eqx.nn.Linear(x_dim, d_h, key=key_x)
        self.res_layers = [MCDNetResBlock(d_h, d_t, act, key=k) for k in keys_res]
        # Final layer
        self.final_layer = eqx.nn.Linear(d_h, 2 * x_dim, key=key_final)
        # Scaling for score term
        self.const_scale = jnp.array(0.0)
        self.score_scale = jnp.array(0.0)

    def __call__(self, t: Array, x: Array) -> Array:
        t_emb = self.t_emb(t[None])
        h = self.x_emb(x)
        for res_layer in self.res_layers:
            h = res_layer(t_emb, h)
        h = self.act(h)
        h = self.final_layer(h)
        # Scale terms by constants
        const = self.const_scale * h[: self.x_dim]
        score_factor = 1 + self.score_scale * h[: self.x_dim]
        # Combine with score
        score_t = self.get_score_gamma_t(t, x)
        return const + score_factor * score_t
