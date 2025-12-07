import math
import torch


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
        shift_sigma=False,

        prediction_target = "edm",
        **kwargs,

    ):
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.rho = rho
        self.rescale_t = rescale_t
        self.shift_sigma = shift_sigma

        self.num_timesteps = num_timesteps
        self.device = device
        self.prediction_target = prediction_target

    def to(self, device):
        self.device = device

    def get_scalings_for_boundary_condition(self, sigma, h=224):
        _sigma_min = self.sigma_min * (h/64 ) if self.shift_sigma else self.sigma_min
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

    def gen_denoise(self, model, x_t, sigmas, **model_kwargs):
        c_skip, c_out, c_in =  self.get_scalings_for_boundary_condition(sigmas, h=x_t.shape[-2])
        rescaled_t = 1000 * 0.25 * torch.log(sigmas + 1e-44) if self.rescale_t == "cm" else 0.25 * torch.log(sigmas + 1e-44) # 1000*1/4*ln(\sigma)
        model_output = model(append_dims(c_in, x_t.ndim) * x_t, rescaled_t, **model_kwargs)

        if self.prediction_target == "edm":
            c_skip, c_out = map(lambda x: append_dims(x, x_t.ndim), [c_skip, c_out])
            denoised_output = c_skip*x_t + c_out*model_output
        else:
            denoised_output = model_output

        return model_output, denoised_output

    def sample(self, model, bs, shape, steps, sampler="euler", 
                return_sample_traj=False, print_loss=False, model_kwargs={}, **kwargs):

        def get_sigmas_karras(n):
            ramp = torch.linspace(0, 1, n)
            min_inv_rho = self.sigma_min ** (1 / self.rho)
            max_inv_rho = self.sigma_max ** (1 / self.rho)
            sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
            if self.shift_sigma:
                sigmas = sigmas*(shape[1]/64)

            return sigmas

        # print(model_kwargs)
        def denoiser(x_t, sigma):
            _, denoised = self.gen_denoise(model, x_t, sigma, **model_kwargs)
            denoised = denoised.clamp(-1, 1)
            return denoised

        x_T = torch.randn((bs, *shape), device=self.device) * self.sigma_max
        if self.shift_sigma:
            x_T = x_T*(shape[1]/64)
        sigmas = get_sigmas_karras(steps).to(self.device)
        # print(type(sigmas), sigmas)

        to_d = x0_to_d if self.prediction_target == "x0" or self.prediction_target == "edm" else epsilon_to_d

        return SAMPLE_FUNCTIONS[sampler](denoiser, x_T, sigmas, return_sample_traj=return_sample_traj, to_d=to_d, **kwargs)


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
    print_loss = False,
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
    print_loss = False,
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
    print_loss = False,
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