"""Uncertainty heads used by the CSD-HNR experiments."""

import math

import torch
import torch.nn as nn


class ScalarTotalVarianceHead(nn.Module):
    """Predict one bounded total variance and broadcast it over dimensions.

    CSD only observes ``sum(exp(log_variance))``.  Predicting a full diagonal
    vector is therefore unidentifiable: many very different vectors produce
    exactly the same CSD score.  This head predicts the identifiable scalar
    directly, then returns a diagonal isotropic log-variance for compatibility
    with the existing PCME++ criterion and evaluation code.
    """

    def __init__(
            self, input_dim, embed_dim, total_var_min=0.01,
            total_var_max=0.30, total_var_init=0.10):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.total_var_min = float(total_var_min)
        self.total_var_max = float(total_var_max)
        self.total_var_init = float(total_var_init)
        if not 0 < self.total_var_min < self.total_var_max:
            raise ValueError(
                'Expected 0 < total_var_min < total_var_max, got '
                f'{self.total_var_min=} and {self.total_var_max=}')
        if not self.total_var_min < self.total_var_init < self.total_var_max:
            raise ValueError(
                'total_var_init must lie strictly inside the configured '
                'variance interval')

        self.proj = nn.Linear(int(input_dim), 1)
        nn.init.zeros_(self.proj.weight)
        init_ratio = (
            (self.total_var_init - self.total_var_min)
            / (self.total_var_max - self.total_var_min)
        )
        init_logit = math.log(init_ratio / (1.0 - init_ratio))
        nn.init.constant_(self.proj.bias, init_logit)

    def forward(self, features):
        total_variance = self.total_var_min + (
            self.total_var_max - self.total_var_min
        ) * torch.sigmoid(self.proj(features))
        per_dim_variance = total_variance / self.embed_dim
        log_variance = per_dim_variance.clamp_min(
            torch.finfo(features.dtype).tiny).log()
        return log_variance.expand(-1, self.embed_dim)
