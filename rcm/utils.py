import torch
import torch.nn as nn
import numpy as np
import os
import collections
import collections.abc
from absl import logging

import rcm.vit as vit_models
from rcm.rcm_denoiser import RepresentationKarrasDenoiser

import abc
import einops
from itertools import repeat


def set_logger(log_level='info', fname=None):
    import logging as _logging
    handler = logging.get_absl_handler()
    formatter = _logging.Formatter('%(asctime)s - %(filename)s - %(message)s')
    handler.setFormatter(formatter)
    logging.set_verbosity(log_level)
    if fname is not None:
        handler = _logging.FileHandler(fname)
        handler.setFormatter(formatter)
        logging.get_absl_logger().addHandler(handler)


def dct2str(dct):
    return str({k: f'{v:.6g}' for k, v in dct.items()})


class MetricLogger(object):

    def __init__(self, **metrics):

        self.metrics = collections.OrderedDict(metrics)

    def add(self, metric_of_the_step:dict):
        for k, v in metric_of_the_step.items():
            if k not in self.metrics:
                self.metrics[k] = {
                    "value": 0
                }
            self.metrics[k]["value"] = v

    def update(self, metric_of_the_step: dict):
        for k, v in metric_of_the_step.items():
            if k not in self.metrics:
                self.metrics[k] = {
                    "value": 0,
                    "cnt": 0,
                }
            oldval, cnt = self.metrics[k]["value"], self.metrics[k]["cnt"] 
            self.metrics[k]["value"] = oldval*cnt/(cnt+1) + v/(cnt+1)
            self.metrics[k]["cnt"] += 1

    def get(self, key=None):
        if key is None:
            return {k:self.metrics[k]["value"] for k in dict(sorted(self.metrics.items())).keys()}
        else:
            return self.metrics[key]["value"]

    def _mean(self, metric_list:list):
        return sum(metric_list)/len(metric_list)

    def clean(self):
        self.metrics = collections.OrderedDict({})

    def __repr__(self) -> str:
        return ",".join(["{}:{:03f}".format(k, self._mean(v)) for k,v in self.metrics.items()])


def create_ema_and_scales_fn(
    start_scales,
    end_scales,
    total_steps,

    tau_schedule,
    start_tau,
    end_tau,
    **kwargs,
):
    def ema_and_scales_fn(step):

        def icm_scale(step):
            K = total_steps
            k = step
            K_prime = np.ceil(
                K / np.log2(np.ceil(end_scales/start_scales) + 1)
            )
            scales = min(start_scales*(2**np.ceil(k/K_prime)), end_scales) + 1
            return scales

        def cosine_tau(step):
            if 1.5*(step/total_steps)+1 <= 2:
                return end_tau*np.cos(np.pi/2*(1.5*step/total_steps+1)) + start_tau
            else:
                return end_tau

        scales = icm_scale(step)

        if tau_schedule == "constant":
            tau = start_tau
        elif tau_schedule == "cosine":
            tau = cosine_tau(step)
        else:
            raise NotImplementedError(f"unknown tau schedule:{tau_schedule}")

        return int(scales), float(tau)

    return ema_and_scales_fn


def create_diffusion(**kwargs):
    return RepresentationKarrasDenoiser(**kwargs)


def create_model(model_type="vit", **kwargs):
    _model_class = vit_models.__dict__
    if model_type in _model_class:
        return _model_class[model_type](**kwargs)
    else:
        raise ValueError(f"Unknown model type:{model_type}")


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def ema(model_dest: nn.Module, model_src: nn.Module, rate):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.detach().mul_(rate).add_(p_src, alpha=1 - rate)


class TrainState(object):
    def __init__(self, step, optimizer, lr_scheduler, nnet=None, nnet_ema=None, target_model=None):
        
        self.is_warmup = True
        self.step = step

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.nnet_ema = nnet_ema
        self.nnet = nnet
        self.target_model = target_model

    def target_update(self, rate=0.99):
        # used in pre-training and consistency fintuning
        if self.target_model is not None:
            ema(self.target_model, self.nnet, rate)

    def ema_update(self, rate=0.9999):
        # used in denoising and consistency fine-tuning training
        if self.nnet_ema is not None:
            ema(self.nnet_ema, self.nnet, rate)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.step, os.path.join(path, 'step.pth'))
        for key, val in self.__dict__.items():
            if key in ['optimizer', 'lr_scheduler', 'nnet', 'target_model', "nnet_ema"] and val is not None:
                torch.save(val.state_dict(), os.path.join(path, f'{key}.pth'))

    def load(self, path, step=-1):
        logging.info(f'load from {path}')
        try:
            self.step = torch.load(os.path.join(path, 'step.pth'))
        except:
            self.step = 0

        for key, val in self.__dict__.items():
            if key in ["step", "is_warmup"] or val is None: continue

            try:
                if key in ["nnet", "target_model"]:
                    state_dict = torch.load(os.path.join(path, f'{key}.pth'), map_location='cpu')
                    missing, unexpected = val.load_state_dict(state_dict, strict=False)

                    if len(missing) != 0:
                        logging.info(f"Missing keys:{missing} when loading ckpt of {key}")
                    elif len(unexpected) != 0:
                        logging.info(f"Unexpected keys:{missing} when loading ckpt of {key}")

            except Exception as ex:
                logging.info(f'error when loading ckpt {key}: {ex}, automatically skipping...')

    def resume(self, ckpt_root, step=None):
        if not os.path.exists(ckpt_root):
            logging.info("training from scratch")
            return
        if step is None:
            ckpts = list(filter(lambda x: '.ckpt' in x, os.listdir(ckpt_root)))
            if len(ckpts) == 0:
                return
            else: # resume from the latest step
                steps = list(
                    map(
                        lambda x: int(x.split(".")[0]), 
                        [c for c in ckpts if "latest" not in c]
                    )
                )
                max_step = max(steps) if len(steps) != 0 else -1
                latest = torch.load(os.path.join(ckpt_root, 'latest.ckpt', "step.pth"))
                max_step = torch.load(os.path.join(ckpt_root, f'{max_step}.ckpt', "step.pth"))
                if latest > max_step:
                    load_step = "latest"
                else:
                    load_step = str(max_step)

        ckpt_path = os.path.join(ckpt_root, f'{load_step}.ckpt')
        logging.info(f'resume from {ckpt_path}')
        self.load(ckpt_path)

    def to(self, device):
        for key, val in self.__dict__.items():
            if isinstance(val, nn.Module):
                val.to(device)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with torch.no_grad():
            torch.distributed.broadcast(p, 0)


def customized_lr_scheduler(optimizer, min_scale=-1, name="warmup-cosine", warmup_steps=-1, total_training_steps=100000):
    from torch.optim.lr_scheduler import LambdaLR
    import math

    if name == "warmup-cosine": # linear warmup then cosine decay
        def fn(step):
            if warmup_steps > 0:
                if step <= warmup_steps:
                    return min(step / warmup_steps, 1)
                elif step <= total_training_steps:
                    lr_scale = 0.5*(1+math.cos((step-warmup_steps)*math.pi/(total_training_steps-warmup_steps)))
                    if min_scale != -1:
                        lr_scale = max(lr_scale, min_scale)
                    return lr_scale
                else:
                    return min_scale if min_scale != -1 else 0
            else:
                return 1
    elif name == "warmup":  # linear warmup
        def fn(step):
            if warmup_steps > 0 and step <= warmup_steps:
                return min(step / warmup_steps, 1)
            else:
                return 1

    return LambdaLR(optimizer, fn)


def param_groups_wd(model, lr=1e-4, weight_decay=0.05):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    ignore = model.no_weight_decay()

    for n, p in model.named_parameters():
        # print(n)
        if not p.requires_grad:
            continue

        this_decay = 0 if n in ignore else weight_decay
        group_name = "no_decay" if n in ignore else "decay"

        if group_name not in param_groups:

            param_group_names[group_name] = {
                "lr": lr,
                "weight_decay": this_decay,
                "params": [],
            }

            param_groups[group_name] = {
                "lr": lr,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    logging.info(param_group_names)
    return list(param_groups.values())


def initialize_train_state(args, accelerator):

    logging.info("creating model and diffusion...")
    device = accelerator.device

    model = create_model(**args.nnet).train() 
    logging.info(f"Total number of parameters: {model.get_num_param()}")
    target_model = create_model(**args.nnet).eval()

    for param in target_model.parameters():
        param.requires_grad_(False) # freeze target model parameters

    if accelerator.num_processes > 1:
        logging.info("Distributed training, use synchronized batch norm")
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        target_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(target_model)

    optimizer = torch.optim.AdamW(
        param_groups_wd(model, lr=args.optimizer.lr, weight_decay=args.optimizer.weight_decay) if args.train.group_wd else [p for p in model.parameters() if p.requires_grad],
        **args.optimizer
    )

    train_state = TrainState(
        step=0, # by default, start step is 0
        optimizer=optimizer,
        lr_scheduler=customized_lr_scheduler(optimizer, **args.lr_scheduler),
        nnet=model, target_model=target_model,
    )

    train_state.to(device)
    accelerator.wait_for_everyone()

    logging.info("synchronizing online model parameters...")
    if accelerator.num_processes > 1:
        sync_params(model.parameters())
        sync_params(model.buffers())

    # sync parameters with online model
    logging.info("synchronizing target model parameters...")
    train_state.target_update(0)

    return train_state


def log_loss_dict(diffusion, ts, losses):
    metrics = {}
    for key, values in losses.items():
        if values.ndim == 0:
            metrics[key] = values
        else:
            metrics[key] = values.mean() # log overall loss
            if diffusion.time_sample_schedule == "lognormal" or diffusion.time_sample_schedule == "lognormal_continuous": # ts might be indices or sigmas
                t = diffusion.find_nearest(ts)
            else:
                t = ts
            for sub_t, sub_loss in zip(t.cpu().numpy(), values.detach().cpu().numpy()):
                quartile = int(4 * sub_t / diffusion.num_timesteps)
                metrics[f"{key}_q{quartile}"] =  sub_loss.mean()

    return metrics


def _compute_norms(state_dict, grad_scale=1.0):
    grad_norm = 0.0
    param_norm = 0.0
    for k, p in state_dict:
        if not p.requires_grad: continue
        with torch.no_grad():
            param_norm += torch.norm(p, p=2, dtype=torch.float32).item() ** 2
            if p.grad is not None:
                grad_norm += torch.norm(p.grad, p=2, dtype=torch.float32).item() ** 2
    return np.sqrt(grad_norm) / grad_scale, np.sqrt(param_norm)
