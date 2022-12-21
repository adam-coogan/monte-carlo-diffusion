# mcd

[![docs](https://readthedocs.org/projects/monte-carlo-diffusion/badge/?version=latest)](http://monte-carlo-diffusion.readthedocs.io/?badge=latest)

`mcd` implements the Monte Carlo diffusion algorithm from [Score-Based Diffusion meets Annealed Importance Sampling](https://arxiv.org/abs/2208.07698) in `jax`, using [`diffrax`](https://github.com/patrick-kidger/diffrax)
to handle differential equations and [`equinox`](https://github.com/patrick-kidger/equinox/)
for building neural networks. It also implements Neal's original version of [annealed importance sampling](https://arxiv.org/abs/physics/9803008).

Check out the [docs](https://monte-carlo-diffusion.readthedocs.io/en/latest/) badge
above for more details, or try out the scripts applying the method to some low-dimensional
problems (runnable on a laptop).

## Installation

```
git clone git@github.com:adam-coogan/monte-carlo-diffusion.git
cd monte-carlo-diffusion
pip install .
```
