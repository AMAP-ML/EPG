import math
import torch
import torch.nn.functional as F
from rcm.resample import create_time_sampler

def append_dims(x, target_dims):
    if not isinstance(x, torch.Tensor):
        return x

    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def x0_to_d(x, sigma, denoised):
    """Converts a x0-prediction denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)

def epsilon_to_d(x, sigma, denoised):
    """Converts a epsilon-prediction denoiser output to a Karras ODE derivative."""
    return denoised


class RepresentationKarrasDenoiser:
    def __init__(
        self,
        # shared parameters
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        rescale_t="cm", # rescale time condition or not

        num_timesteps = 20,
        device = "cuda",

        prediction_target = "edm",
        time_sample_schedule = "unique",
        sample_param = {},
        tmin_tau=0.1,
        tau_schedule="interpolated",  # [interpolated, constant]

        pseudo_huber_c = 0.08,
        **kwargs,

    ):

        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.rho = rho
        self.rescale_t = rescale_t

        self.time_sample_schedule = time_sample_schedule
        self.time_sampler = create_time_sampler(time_sample_schedule, **sample_param)
        self.num_timesteps = num_timesteps
        self.device = device
        self.prediction_target = prediction_target

        self.pseudo_huber_c = pseudo_huber_c
        self.tau_schedule = tau_schedule
        self.tmin_tau = tmin_tau

    def to(self, device):
        self.device = device

    def set_scale(self, scale):
        self.num_timesteps = scale
        self.sigmas = torch.tensor([(self.sigma_max ** (1 / self.rho) + idx / (self.num_timesteps - 1) * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
            )**self.rho for idx in range(0, self.num_timesteps)]).to(self.device)

    def sample_time(self, batch_size, step=None):
        return self.time_sampler.sample(batch_size, self.device, step=step, num_scales=self.num_timesteps)

    def get_scalings_for_boundary_condition(self, sigma, h=224):
        _sigma_min = self.sigma_min * (h/64)
        c_skip = self.sigma_data**2 / (
            (sigma - _sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - _sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def denoise(self, model, x_t, sigmas, get_rep_fn, **model_kwargs):
        c_skip, c_out, c_in = self.get_scalings_for_boundary_condition(sigmas, h=x_t.shape[-2])
        rescaled_t = 1000 * 0.25 * torch.log(sigmas + 1e-44)
        ret = model(append_dims(c_in, x_t.ndim) * x_t, rescaled_t, **model_kwargs)
        return get_rep_fn(ret)

    def get_tau(self, sigmas, tmax_tau):
        # used when computing representation consistency loss
        if self.tau_schedule == "constant":
            return self.tmin_tau
        elif self.tau_schedule == "interpolated":
            # smaller tau at smaller t
            return (sigmas-self.sigma_min)/(self.sigma_max - self.sigma_min)*tmax_tau + (self.sigma_max-sigmas)/(self.sigma_max - self.sigma_min)*self.tmin_tau
        else:
            raise NotImplementedError(f"unknown tau schedule: {self.tau_schedule}")

    def contrastive_loss(self, f, f_target, accelerator, sigmas=None, tau=None):

        if accelerator.num_processes > 1:
            rank = accelerator.process_index
            world_size = accelerator.num_processes
            B, C = f_target.shape
            global_ftarget = accelerator.gather(f_target).reshape(world_size, B, C)
            # rearrange batch for correct positive similarity
            all_f_target = [global_ftarget[rank]]
            for i in range(world_size):
                if i != rank:
                    all_f_target.append(global_ftarget[i])
            f_target = torch.cat(all_f_target, dim=0) # cat along batch dimension

        f = F.normalize(f.flatten(1), dim=1)
        f_target = F.normalize(f_target.flatten(1), dim=1)

        tau = append_dims(self.get_tau(sigmas, tmax_tau=tau), f.ndim) if sigmas is not None else 0.2

        logits = torch.einsum("nc,ck->nk", [f, f_target.T])
        l_pos = logits.diagonal(0)
        logsumexp_pos = l_pos / tau
        logsumexp_all = torch.logsumexp(logits.float() / tau, dim=1).to(f.dtype)
        nce = logsumexp_all  - logsumexp_pos

        return nce

    def PreTrainStep(
        self,
        model,
        x_start,
        indices=None,
        model_kwargs=None,
        target_model=None,
        noise=None,
        x_aug = None,
        x_aug2 = None,
        accelerator = None,
        tau=None,
    ):
        if model_kwargs is None:
            model_kwargs = {}

        if noise is None:
            noise = torch.randn_like(x_start)

        assert target_model is not None, "Must have a target model"
        dims = x_start.ndim

        def denoise_fn(x, t, get_rep_fn):
            return self.denoise(model, x, t, get_rep_fn, **model_kwargs)

        @torch.no_grad()
        def target_denoise_fn(x, t, get_rep_fn):
            return self.denoise(target_model, x, t, get_rep_fn, **model_kwargs)

        assert self.time_sample_schedule == "unique", "Only unique time sampling is supported for pretraining"

        t = self.sigma_max ** (1 / self.rho) + indices / (self.num_timesteps - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (self.num_timesteps - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        indices0 = torch.full((x_start.shape[0],), self.num_timesteps-1, device=x_start.device)
        t0 = self.sigma_max ** (1 / self.rho) + indices0 / (self.num_timesteps - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t0 = t0**self.rho

        *_, h, _ = x_start.shape
        # shift sigma w.r.t image size 64
        # Reference: Simple diffusion, Link: https://arxiv.org/abs/2301.11093
        t, t2, t0 = t*(h/64), t2*(h/64), t0*(h/64)

        # compute symmetric contrastive loss
        x_t0 = x_aug + noise*append_dims(t0, dims)
        x_t0_target = x_aug2 + noise*append_dims(t0, dims)

        h = denoise_fn(x_t0, t0, lambda x: x[0])
        h_target = target_denoise_fn(x_t0_target, t0, lambda x: x[0].detach())
        contrastive_loss = self.contrastive_loss(h, h_target, accelerator=accelerator)

        h = denoise_fn(x_t0_target, t0, lambda x: x[0])
        h_target = target_denoise_fn(x_t0, t0, lambda x: x[0].detach())
        contrastive_loss += self.contrastive_loss(h, h_target, accelerator=accelerator)

        # compute representation consistency loss
        x_t = x_start + noise*append_dims(t, dims)
        x_t2 = x_start + noise*append_dims(t2, dims)

        h = denoise_fn(x_t, t, lambda x: x[1])
        h_target = denoise_fn(x_t2, t2, lambda x: x[1].detach())
        xt_consistency = self.contrastive_loss(
            h, h_target, 
            accelerator=accelerator, 
            sigmas=t*64/x_start.shape[-2], # shift sigma w.r.t image size 64
            tau=tau
        )

        terms = {}
        terms["loss"] = contrastive_loss + xt_consistency
        terms["contrastive_loss"] = contrastive_loss
        terms["consistency"] = xt_consistency

        return terms

    """
         Denoising/Consistency Fine-tuning 
    """
    def gen_denoise(self, model, x_t, sigmas, **model_kwargs):
        c_skip, c_out, c_in =  self.get_scalings_for_boundary_condition(sigmas, h=x_t.shape[-2])
        rescaled_t = 1000 * 0.25 * torch.log(sigmas + 1e-44)
        model_output = model(append_dims(c_in, x_t.ndim) * x_t, rescaled_t, **model_kwargs)

        if self.prediction_target == "edm":
            # edm parameterization used in consistency fine-tuning
            c_skip, c_out = map(lambda x: append_dims(x, x_t.ndim), [c_skip, c_out])
            denoised_output = c_skip*x_t + c_out*model_output
        else: 
            # x0 prediction used in denoising fine-tuning
            denoised_output = model_output

        return model_output, denoised_output

    def sample(self, model, bs, shape, steps, sampler="euler", 
                return_sample_traj=False, model_kwargs={}, **kwargs):

        def get_sigmas_karras(n):
            ramp = torch.linspace(0, 1, n)
            min_inv_rho = self.sigma_min ** (1 / self.rho)
            max_inv_rho = self.sigma_max ** (1 / self.rho)
            sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
            sigmas = sigmas*(shape[1]/64) # shift sigma w.r.t image size 64

            return sigmas

        def denoiser(x_t, sigma):
            _, denoised = self.gen_denoise(model, x_t, sigma, **model_kwargs)
            denoised = denoised.clamp(-1, 1)
            return denoised

        x_T = torch.randn((bs, *shape), device=self.device) * self.sigma_max
        x_T = x_T*(shape[1]/64) # shift w.r.t image size 64

        sigmas = get_sigmas_karras(steps).to(self.device)
        to_d = x0_to_d if self.prediction_target == "x0" or self.prediction_target == "edm" else epsilon_to_d

        return SAMPLE_FUNCTIONS[sampler](denoiser, x_T, sigmas, return_sample_traj=return_sample_traj, to_d=to_d, **kwargs)

    def get_weightings(self, sigmas, r=None, index=None):
        
        sigmas1, sigmas2 = (sigmas, r) if self.time_sample_schedule == "lognormal-ect" else (self.get_sigmas(index), self.get_sigmas(index+1))

        weightings =  1/(sigmas1-sigmas2)
        weightings[sigmas1 == sigmas2] = 0.0

        return weightings

    def find_nearest(self, t):
        idx = ( t[:, None].expand(-1, self.num_timesteps) - self.sigmas[None, :].expand(t.shape[0], -1) ).abs().argmin(dim=1)
        return idx

    def get_sigmas(self, index):
        sigma = self.sigma_max ** (1 / self.rho) + index / (self.num_timesteps - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        sigma = sigma**self.rho

        return sigma

    def DenoiseTuneStep(
        self,
        model,
        x_start,
        sigmas=None,
        model_kwargs = {}, # {class label}
    ):

        dims = x_start.ndim
        B, C, H, W = x_start.shape

        def denoise_fn(x, t):
            return self.gen_denoise(model, x, t, **model_kwargs)

        idx = self.find_nearest(sigmas)
        idx = torch.clamp(idx, max=self.num_timesteps-1)
        t = self.sigma_max ** (1 / self.rho) + idx / (self.num_timesteps - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t = t*(H/64)
        noise = torch.randn_like(x_start)
        x_t = x_start + noise*append_dims(t, dims)
        _, denoised_output = denoise_fn(x_t, t)

        loss = F.mse_loss(denoised_output, x_start, reduction="none")
        weights = self.get_weightings(append_dims(t, loss.ndim), index=append_dims(idx, loss.ndim))
        loss *= weights

        return dict(loss=loss)

    def ConsisTuneStep(
        self,
        model,
        x_start,
        sigmas=None,
        model_kwargs={},
        target_model=None,
        perceptual_model = None, # perceptual model for computing contrastive loss
        accelerator=None,
        ect_sample_stage=0,

    ):

        noise = torch.randn_like(x_start)

        assert target_model is not None, "Must have a target model"
        dims = x_start.ndim

        def denoise_fn(x, t):
            return self.gen_denoise(model, x, t, **model_kwargs)
        @torch.no_grad()
        def target_denoise_fn(x, t):
            return self.gen_denoise(target_model, x, t, **model_kwargs)

        assert self.time_sample_schedule == "lognormal-ect"

        t, t2 = sigmas
        idx = torch.clamp(
            self.find_nearest(t), 
            max=self.num_timesteps-2,
        ) if ect_sample_stage == 0 else -1

        *_, H, W = x_start.shape
        t = t*(H/64) # shift sigma w.r.t image size 64
        t2 = t2*(H/64) # shift sigma w.r.t image size 64
        noise = torch.randn_like(x_start)

        x_t = x_start + noise*append_dims(t, dims)
        x_t2 = x_start + noise*append_dims(t2, dims)

        rng_state = torch.cuda.get_rng_state()
        _, denoised_output = denoise_fn(x_t, t)

        if ect_sample_stage != 0:
            torch.cuda.set_rng_state(rng_state)
            _, denoised_output_target = target_denoise_fn(x_t2, t2)
        else:
            # denoising train, as warmup for consistency tuning
            denoised_output_target = x_start

        loss_dict = dict()

        # psudo-huber loss
        loss = (denoised_output - denoised_output_target)**2
        loss = loss.reshape(loss.shape[0], -1).sum(dim=-1) 
        loss /= ((loss.detach() + self.pseudo_huber_c ** 2).sqrt())

        weights = self.get_weightings(append_dims(t, loss.ndim), r=append_dims(t2, loss.ndim), index=append_dims(idx, loss.ndim))
        loss *= weights

        if perceptual_model is not None and ect_sample_stage != 0:
            bs = x_start.shape[0]
            t0 = torch.full((bs,), self.sigma_min).to(x_start.device)
            t0 *= (H/64) # shift sigma w.r.t image size 64

            f1 = self.denoise(perceptual_model, denoised_output, t0, lambda x: x[0])
            f2 = self.denoise(perceptual_model, x_start, t0, lambda x: x[0]).detach()
            perceptual_loss = self.contrastive_loss(f1, f2, accelerator=accelerator, sigmas=t)

            loss += perceptual_loss
            loss_dict["perceptual_loss"] = perceptual_loss.detach()

        loss_dict["loss"] = loss

        return loss_dict


def sample_edm_sde(
    denoiser,
    x, 
    sigmas,
    sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    to_d = None,
    **kwargs,
):

    s_in = x.new_ones([x.shape[0]])
    # Main sampling loop.
    # x_next = latents.to(torch.float64) * t_steps[0]
    indices = range(len(sigmas) - 1)
    num_steps = len(sigmas)

    for idx in indices: # 0, ..., N-1
        t_cur = sigmas[idx]
        t_next = sigmas[idx + 1]

        noise = torch.randn_like(x)
        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, math.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = t_cur + gamma * t_cur
        x_hat = x + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * noise
        # Euler step.
        denoised = denoiser(x_hat, s_in*t_hat)
        d_cur = to_d(x_hat, t_hat, denoised)
        x = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if idx < len(indices) - 1:
            denoised = denoiser(x, s_in*t_next)
            d_prime = to_d(x, t_next, denoised)
            x = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x


def get_ancestral_step(sigma_from, sigma_to):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    sigma_up = (
        sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2
    ) ** 0.5
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


def sample_euler_ancestral(
    denoiser,
    x,
    sigmas,
    progress=False,
    callback=None,
    return_sample_traj=False,
    model_kwargs = None,
    to_d = None,
    **kwargs,
):
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    all_x = []
    x_T = x

    for i in indices:
        sigma = sigmas[i]
        denoised = denoiser(x, sigma * s_in)
        d = to_d(x, sigma, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "denoised": denoised,
                }
            )
        sigma_down, sigma_up = get_ancestral_step(sigma, sigmas[i + 1])
        d = to_d(x, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        x = x + torch.randn_like(x) * sigma_up

    if return_sample_traj:
        all_x = torch.cat(all_x, dim=0)
        return all_x
    else:
        return x


def sample_euler_sde(
    denoiser,
    x,
    sigmas,
    progress=False,
    callback=None,
    return_sample_traj=False,
    model_kwargs = None,
    to_d = None,
    **kwargs,
):
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    all_x = []
    x_T = x
    # sigmas = torch.linspace(sigmas[0], sigmas[-1], len(sigmas), device=x.device)
    # print(sigmas)
    for i in indices:
        sigma = sigmas[i]
        denoised = denoiser(x, sigma * s_in)
        d = to_d(x, sigma, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "denoised": denoised,
                }
            )
        dt = sigmas[i + 1] - sigma
        noise = torch.randn_like(x)
        x = x + 2*d * dt + (2*sigma*(sigma-sigmas[i+1])).sqrt()*noise
        if return_sample_traj:
            all_x.append(x)

    if return_sample_traj:
        all_x = torch.cat(all_x, dim=0)
        return all_x
    else:
        return x

# @torch.no_grad()
def sample_euler(
    denoiser,
    x,
    sigmas,
    progress=False,
    callback=None,
    return_sample_traj=False,
    model_kwargs = None,
    to_d = None,
    **kwargs,
):
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    all_x = []
    x_T = x
    for i in indices:
        sigma = sigmas[i]
        denoised = denoiser(x, sigma * s_in)
        d = to_d(x, sigma, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "denoised": denoised,
                }
            )
        dt = sigmas[i + 1] - sigma
        x = x + d * dt
        if return_sample_traj:
            all_x.append(x)

    if return_sample_traj:
        all_x = torch.cat(all_x, dim=0)
        return all_x
    else:
        return x


# @torch.no_grad()
def sample_heun(
    denoiser,
    x,
    sigmas,
    progress=False,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    to_d=None,
    return_sample_traj=False,
    **kwargs,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    all_x = []
    for i in indices:
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = denoiser(x, sigma_hat * s_in)
        d = to_d(x, sigma_hat, denoised)

        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            # Euler method
            x = x + d * dt
        else:
            # Heun's method
            x_2 = x + d * dt
            denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt

        if return_sample_traj:
            all_x.append(x)

    if return_sample_traj:
        all_x = torch.cat(all_x, dim=0)
        return all_x
    else:
        return x


# @torch.no_grad()
def sample_onestep(
    denoiser,
    x,
    sigmas,
    **kwargs,
):

    s_in = x.new_ones([x.shape[0]])
    sigma = sigmas[0]
    x = denoiser(x, sigma * s_in)
    return x


SAMPLE_FUNCTIONS = {
    "euler": sample_euler,
    "heun": sample_heun,
    "onestep": sample_onestep,
    "euler_sde": sample_euler_sde,
    "euler_ancestral": sample_euler_ancestral,
    "edm_sde": sample_edm_sde,
}