import numpy as np
from abc import ABC, abstractmethod
import torch
import math
import random
import logging


def create_time_sampler(name, **kwargs):
    """
    Create a ScheduleSampler from a library of pre-defined samplers.

    :param name: the name of the sampler.
    """
    logging.info(f"Using noise proposal distribution: {name}")
    if name == "uniform":
        return UniformSampler(**kwargs)
    elif name == "unique":
        return UniqueSampler(**kwargs)
    elif name == "lognormal" or name == "lognormal_continuous":
        return LogNormalSampler(**kwargs)
    elif name == "lognormal-ect":
        return LogNormalSamplerECT(**kwargs)
    elif name == "unique_lognormal":
        return UniqueLogNormalSampler(**kwargs)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniqueSampler(ScheduleSampler):
    # NOTE: This sampler won't sample indices==len(p)-2. 2024.04.30 Jiachen
    def __init__(self, **kwargs):
        self._weights = None

    def weights(self):
        return self._weights

    def update_weights(self, num_scales):
        self._weights = np.ones([num_scales-1])

    def sample(self, batch_size, device, num_scales=None, **kwargs):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        self.update_weights(num_scales)
        w = self.weights()
        # print(w)
        p = w / np.sum(w)

        # NOTE: it won't sample indices==len(p)-2
        indices_np = np.random.choice(len(p), size=(1,), p=p)
        indices = torch.from_numpy(indices_np).repeat(batch_size).long().to(device)

        return indices


class LogNormalSampler:
    def __init__(self, p_mean=-1.2, p_std=2.0, even=False, **kwargs):
        self.p_mean = p_mean
        self.p_std = p_std
        logging.info(f"Params of noise proposal distribution: ({p_mean}, {p_std})")

        # self.even = even
        # if self.even:
        #     self.inv_cdf = lambda x: norm.ppf(x, loc=p_mean, scale=p_std)
        #     self.rank, self.size = dist.get_rank(), dist.get_world_size()

    def sample(self, bs, device, **kwargs):
        # if self.even:
        #     # buckets = [1/G]
        #     start_i, end_i = self.rank * bs, (self.rank + 1) * bs
        #     global_batch_size = self.size * bs
        #     locs = (torch.arange(start_i, end_i) + torch.rand(bs)) / global_batch_size
        #     log_sigmas = torch.tensor(self.inv_cdf(locs), dtype=torch.float32, device=device)
        # else:
        log_sigmas = self.p_mean + self.p_std * torch.randn(bs, device=device)
        sigmas = torch.exp(log_sigmas)
        return sigmas


class LogNormalSamplerECT:
    def __init__(self, p_mean=-1.2, p_std=2.0,
        q = 4,
        k = 8,
        b = 1,
        d = (50000, 50000),
        delta = 0,
        p = 0.0,
    ):
        self.p_mean = p_mean
        self.p_std = p_std
        self.sigma_min = 0.002
        self.q, self.k, self.b, self.d = q, k, b, d

        self.delta = delta
        self.p = p # probability of setting r to sigma_min, which equals to the diffusion model training

        logging.info(f"Params of noise proposal distribution: ({p_mean}, {p_std}), q: {q}, k: {k}, b: {b}, d: {d}, delta: {delta}, p: {p}")


    def t_to_r_sigmoid(self, step, t):
        # Method from ECT
        adj = 1 + self.k * torch.sigmoid(-self.b * t)

        if not isinstance(self.d, int):
            _step = (step - self.d[0])
            if _step >= 0:
                stage = (step - (self.d[0] - self.d[1]))//self.d[1]+ self.delta
            else:
                stage = self.delta
        else:
            stage = step//self.d + self.delta

        decay = 1 / self.q ** (stage)
        print("ect sampling stage:", stage)
        ratio = 1 - decay * adj
        r = t * ratio
        return torch.clamp(r, min=self.sigma_min)

    def sample(self, bs, device, step=None, **kwargs):

        log_sigmas = self.p_mean + self.p_std * torch.randn(bs, device=device)
        t = torch.exp(log_sigmas)
        t = torch.clamp(t, min=self.sigma_min)
        r = self.t_to_r_sigmoid(step, t)

        # if self.p > 0 and random.random() <= self.p:
        #     dm_r = torch.full((bs, ), self.sigma_min, device=device)            

        return t, r


class LogNormalSamplerMeanFlow:
    def __init__(self, p_mean=-1.2, p_std=2.0,
        p = 0.75 # probability of setting r=t
        # From MeanFlow r,t sampler Version 1: ensure t >= r.
    ):
        self.p_mean = p_mean
        self.p_std = p_std
        self.p = p

    def logit_normal_timestep_sample(self, bs, device):
        log_sigmas = self.p_mean + self.p_std * torch.randn(bs, device=device)
        t = torch.exp(log_sigmas)
        t = torch.clamp(t, min=self.sigma_min)
        return t

    def sample(self, bs, device, step=None, **kwargs):

        t = self.logit_normal_timestep_sample(bs, device)
        r = self.logit_normal_timestep_sample(bs, device)
        mask = torch.rand(bs, device=device) < self.p
        r = torch.where(mask, t, r)
        r = torch.minimum(t, r)
        return t, r