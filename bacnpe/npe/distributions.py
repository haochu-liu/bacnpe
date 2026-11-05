"""Probability distributions used to build NPE priors."""
from __future__ import annotations

import torch
from torch.distributions import Distribution, constraints


class DiscreteUniform(Distribution):
    """
    Uniform distribution on the integers ``low, low + 1, ..., high``.

    Used as the prior on the number of sites ``L``. Samples are returned as
    float tensors (integer-valued) so they can be combined with continuous
    priors in :class:`sbi.utils.MultipleIndependent`.

    ``log_prob`` returns ``-log(high - low + 1)`` for every value in
    ``[low, high]``, including non-integers. NPE treats ``L`` as continuous,
    so its posterior samples are generally not integers, and they must still
    count as inside the prior's support.

    Parameters
    ----------
    low, high : float or torch.Tensor
        Integer-valued bounds, both included (``low <= high``). Tensors give
        a batch of independent distributions.
    validate_args : bool, optional
        Passed to :class:`torch.distributions.Distribution`.
    """

    arg_constraints = {"low": constraints.real, "high": constraints.real}

    def __init__(self, low, high, validate_args=None):
        self.low = torch.as_tensor(low, dtype=torch.float32)
        self.high = torch.as_tensor(high, dtype=torch.float32)
        self.low, self.high = torch.broadcast_tensors(self.low, self.high)
        if torch.any(self.low != torch.round(self.low)) or torch.any(self.high != torch.round(self.high)):
            raise ValueError("`low` and `high` must be integer-valued.")
        if torch.any(self.low > self.high):
            raise ValueError("`low` must be <= `high`.")
        self._num_categories = (self.high - self.low) + 1
        super().__init__(batch_shape=self.low.size(), validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        u = torch.rand(shape, device=self.low.device)
        samples = torch.floor(self.low + u * self._num_categories)
        # Guard against float rounding pushing a sample to high + 1
        return torch.minimum(samples, self.high.expand(shape))

    def log_prob(self, value):
        value = torch.as_tensor(value, dtype=self.low.dtype, device=self.low.device)
        log_p = -torch.log(self._num_categories).expand_as(value)
        in_bounds = (value >= self.low) & (value <= self.high)
        return torch.where(in_bounds, log_p, torch.full_like(log_p, float("-inf")))

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return constraints.interval(self.low, self.high)

    @property
    def mean(self):
        return (self.high + self.low) / 2.0

    @property
    def variance(self):
        n = self._num_categories
        return (n ** 2 - 1) / 12.0

    @property
    def stddev(self):
        return torch.sqrt(self.variance)
