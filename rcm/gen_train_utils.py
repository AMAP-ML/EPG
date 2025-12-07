import os
from absl import logging

import torch
import torch.nn as nn

import rcm.vit as vit_models
from rcm.rcm_denoiser import RepresentationKarrasDenoiser


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
    def __init__(self, step, optimizer=None, lr_scheduler=None, nnet=None, nnet_ema=None, target_model=None, nnet_poor=None):
        
        self.is_warmup = True
        self.step = step

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.nnet = nnet
        self.nnet_ema = nnet_ema
        self.target_model = target_model
        self.nnet_poor = nnet_poor

    def target_update(self, rate=0.99):
        if self.target_model is not None:
            ema(self.target_model, self.nnet, rate)

    def ema_update(self, rate=0.9999):
        if self.nnet_ema is not None:
            ema(self.nnet_ema, self.nnet, rate)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.step, os.path.join(path, 'step.pth'))
        for key, val in self.__dict__.items():
            if key in ['optimizer', 'lr_scheduler', 'nnet', 'target_model', "nnet_ema"] and val is not None:
                torch.save(val.state_dict(), os.path.join(path, f'{key}.pth'))

    def load(self, path):
        logging.info(f'load from {path}')

        for key, val in self.__dict__.items():
            if key in ["nnet", "nnet_ema"] and val is not None:
                if not os.path.exists(os.path.join(path,  f'{key}.pth')):
                    logging.info(f"file not found, skip loading {key}")
                    continue

                state_dict = torch.load(os.path.join(path, f'{key}.pth'), map_location='cpu')
                missing, unexpected = val.load_state_dict(state_dict, strict=False)

                if len(missing) != 0:
                    logging.info(f"Missing keys:{missing} when loading ckpt of {key}")
                if len(unexpected) != 0:
                    # this is expected when loading imagenet for training on cifar10
                    logging.info(f"Unexpected keys:{unexpected} when loading ckpt of {key}")

    def to(self, device):
        for key, val in self.__dict__.items():
            if isinstance(val, nn.Module):
                val.to(device)


def initialize_eval_state(args, accelerator):

    logging.info("creating model and diffusion...")
    device = accelerator.device

    model = create_model(**args.nnet).train() 
    ema_model= create_model(**args.nnet).eval()
    logging.info(f"num parameters: {model.get_num_param()}")
    logging.info(f"Patch embedding requires_grad: {model.patch_embed.proj.weight.requires_grad}")

    nnet_poor = None
    if args.sample.poor_path:
        nnet_poor = create_model(**args.nnet).eval()
        nnet_poor.load_state_dict(torch.load(args.sample.poor_path, map_location="cpu"))

    eval_state = TrainState(step=0, nnet=model, nnet_ema=ema_model, nnet_poor=nnet_poor)
    eval_state.to(device)
    accelerator.wait_for_everyone()

    return eval_state