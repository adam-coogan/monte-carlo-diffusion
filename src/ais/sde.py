from abc import ABC, abstractmethod
from typing import Tuple

import numpyro.distributions as dist


class SDE(ABC):
    def __init__(self, T):
        super().__init__()
        self.T = T

    @abstractmethod
    def get_drift(self, t, x):
        ...

    @abstractmethod
    def get_diffusion(self, t, x):
        ...

    @abstractmethod
    def get_prior_T(self, shape: Tuple[int, ...] = ()) -> dist.Distribution:
        ...
