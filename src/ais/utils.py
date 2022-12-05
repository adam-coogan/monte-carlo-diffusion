from typing import Callable, Tuple

import jax
from jaxtyping import Array  # type: ignore


def batch_mul(a, b):
    """
    Notes:
        Taken from Yang Song's `score_sde <https://github.com/yang-song/score_sde>`_
        repo.
    """
    return jax.vmap(lambda a, b: a * b)(a, b)


# TODO: typing
def get_value_and_div(eps: Array, fn: Callable, x, *args, **kwargs) -> Tuple[Array, Array]:
    """
    Constructs divergence function using the Hutchinson-Skilling trace estimator.

    Args:
        eps: random vector for estimator with the same shape as `x`. Must be sampled
            from a distribution with zero mean and covariance equal to the identity.
            `vmap` over this argument to reduce the estimator's error.
        fn: function taking an argument `x` and possibly other args/kwargs which
            returns a value with the same shape as `x`.
        x: position at which to evaluate function and its divergence.
        args: arguments for `fn`.
        kwargs: keyword arguments for `fn`.

    Returns:
        Value of the function evaluated at `x` and an estimate of the divergence
        :math:`\nabla_{\mathbf{x}} \mathbf{f}(\mathbf{x})`.
    """
    value, get_vjp = jax.vjp(lambda x: fn(x, *args, **kwargs), x)
    return value, (get_vjp(eps)[0] * eps).sum()
