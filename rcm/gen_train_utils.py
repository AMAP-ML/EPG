import os
import abc
import einops
import numpy as np
import collections
from absl import logging

import torch
import torch.nn as nn

import rcm.vit as vit_models
from rcm.rcm_denoiser import RepresentationKarrasDenoiser
from rcm.utils import create_model, create_diffusion

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

    def interpolate_pos_embed(self, model, ckpt):
        pos_embed_checkpoint = ckpt['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            # print(num_patches, num_extra_tokens, orig_size, new_size)
            logging.info("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            ckpt['pos_embed'] = new_pos_embed

        return ckpt

    def load(self, path, load_step=False):
        logging.info(f'load from {path}')
        step_ckpt_path = os.path.join(path, f'step.pth')

        if load_step and os.path.exists(step_ckpt_path):
            self.step = torch.load(step_ckpt_path, map_location='cpu')
            self.is_warmup = False
        else:
            logging.info("Step starts from 0")
            self.step = 0

        for key, val in self.__dict__.items():
            if key in ["nnet", "nnet_ema", "target_model"] and val is not None:
                if not os.path.exists(os.path.join(path,  f'{key}.pth')):
                    logging.info(f"file not found, skip loading {key}")
                    continue

                state_dict = torch.load(os.path.join(path, f'{key}.pth'), map_location='cpu')
                try:
                    missing, unexpected = val.load_state_dict(state_dict, strict=False)
                except Exception as e:
                    logging.info(f"[WARNING]: Fail to load checkpoint for {key}, skip... Exception: {e}")

                if len(missing) != 0:
                    logging.info(f"Missing keys:{missing} when loading ckpt of {key}")
                if len(unexpected) != 0:
                    # this is expected when loading imagenet for training on cifar10
                    logging.info(f"Unexpected keys:{unexpected} when loading ckpt of {key}")

            if key in ["optimizer", "lr_scheduler"] and val is not None and load_step:
                if not os.path.exists(os.path.join(path,  f'{key}.pth')):
                    print(f"file not found, skip loading {key}")
                    continue
                try:
                    state_dict = torch.load(os.path.join(path, f'{key}.pth'), map_location='cpu')
                    val.load_state_dict(state_dict)
                except Exception as ex:
                    print(f"fail to load ckpt for {key}, skip. Error: {ex}")

    def resume(self, ckpt_root, step=None):
        if not os.path.exists(ckpt_root):
            logging.info("training from scratch")
            return 0

        if step is None:
            ckpts = list(filter(lambda x: '.ckpt' in x, os.listdir(ckpt_root)))
            if len(ckpts) == 0:
                logging.info("training from scratch")
                return 0
            elif len(ckpts) == 1:
                load_step = ckpts[0].split(".")[0]
            else:
                steps = map(lambda x: int(x.split(".")[0]), [c for c in ckpts if "latest" not in c])
                max_step = max(steps)
                if os.path.exists(os.path.join(ckpt_root, 'latest.ckpt')):
                    latest = torch.load(os.path.join(ckpt_root, 'latest.ckpt', "step.pth"))
                    max_step = torch.load(os.path.join(ckpt_root, f'{max_step}.ckpt', "step.pth"))
                    if latest > max_step:
                        load_step = "latest"
                    else:
                        load_step = str(max_step)
                else:
                    load_step = str(max_step)
        else:
            load_step = step

        ckpt_path = os.path.join(ckpt_root, f'{load_step}.ckpt')
        logging.info(f'resume from {ckpt_path}')
        self.load(ckpt_path, load_step=True)
        return 1 # successfully resume from weights

    def to(self, device):
        for key, val in self.__dict__.items():
            if isinstance(val, nn.Module):
                val.to(device)


def cnt_params(model):
    return sum(param.numel() for param in model.parameters())


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

    if name == "warmup-cosine":
        def fn(step):
            if warmup_steps > 0:
                if step <= warmup_steps:
                    return min(step / warmup_steps, 1)
                elif step <= total_training_steps:
                    # return lr_min/lr_base + 0.5*(1-lr_min/lr_base)*(1+math.cos(step*math.pi/total_steps))
                    lr_scale = 0.5*(1+math.cos((step-warmup_steps)*math.pi/(total_training_steps-warmup_steps)))
                    if min_scale != -1:
                        lr_scale = max(lr_scale, min_scale)
                    return lr_scale
                else:
                    return min_scale if min_scale != -1 else 0
            else:
                return 1
    elif name == "warmup-step":
        def fn(step):
            if warmup_steps > 0 and step <= warmup_steps:
                return min(step / warmup_steps, 1)
            else:
                if step <= 400000:
                    return 1.0 # 1e-4
                elif step > 400000 and step <= 500000:
                    return 0.3 # 3e-5
                else:
                    return 0.08 # 8e-6

    elif name == "warmup":
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


def param_groups_lrd(model, lr=1e-4, weight_decay=0.05, layer_decay=.75, lrd_mode="encoder"):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    if getattr(model, "name", None) == "rin":
        # only implement lrd on encoder when using rin-like architecture
        num_layers = [len(model.blocks), 0, 0]
        assert lrd_mode == "encoder", f"lrd mode:{lrd_mode} is not supported when using a RIN-like architecture"
    else:
        # lrd for vit-like architecture.
        # middle_blocks may or may not exist.
        num_layers = [len(model.blocks), len(getattr(model, "middle_blocks", [])), len(model.decoder_blocks)]

    # print(num_layers)
    if lrd_mode == "encoder":
        total_layers = num_layers[0] + 1
        layer_scales = list(layer_decay ** (total_layers - i) for i in range(total_layers + 1))
    elif lrd_mode == "all":
        total_layers = sum(num_layers) + 1
        layer_scales = list(layer_decay ** (total_layers - i) for i in range(total_layers + 1))
    else:
        raise ValueError(f"not supported lrd mode:{lrd_mode}")

    ignore = model.no_weight_decay()

    for n, p in model.named_parameters():
        # print(n)
        if not p.requires_grad:
            continue

        this_decay = 0 if n in ignore else weight_decay
        g_decay = "no_decay" if n in ignore else "decay"
        layer_id = get_layer_id_for_vit(n, num_layers, mode=lrd_mode) # lr only decay on encoder
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_groups:
            this_scale = layer_scales[layer_id] if layer_id != -1 else 1

            param_group_names[group_name] = {
                "lr": lr*this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

            param_groups[group_name] = {
                "lr": lr*this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    logging.info(param_group_names)

    return list(param_groups.values())

def all_lrd(name, num_layers, name2id):
    """
        only apply lrd to all layers
    """
    total_layers = sum(num_layers) + 1

    encoder_layers = num_layers[0]
    middle_block_layers = num_layers[1] if len(num_layers) > 2 else 0
    decoder_layers = num_layers[1] if len(num_layers) == 2 else num_layers[2]

    if name in name2id:
        return name2id[name]
    elif name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    elif name.startswith('norm'):
        return encoder_layers # same lr as the last encoder block
    elif name.startswith('mlp'):
        return encoder_layers + 1 # same lr as the first from-scratch block
    elif name.startswith("middle_blocks"):
        return int(name.split('.')[1]) + 1 + encoder_layers
    elif name.startswith("decoder_blocks"):
        return int(name.split('.')[1]) + 1 + encoder_layers + middle_block_layers
    else:
        # out layer
        return total_layers

def encoder_lrd(name, num_layers, name2id):
    """
        only apply lrd to encoder layers
    """
    total_layers = num_layers[0] + 1 

    if name in name2id:
        return name2id[name]
    elif name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed') or name.startswith('time_embed'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    else:
        # norm layer and decoder layers
        return total_layers


def get_layer_id_for_vit(name, num_layers, name2id={}, mode="legacy"):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """

    if mode == "encoder":
        return encoder_lrd(name, num_layers, name2id)
    elif mode == "all":
        return all_lrd(name, num_layers, name2id)


def reload_forward(self):

    def forward(x, timesteps):
        x = self.patch_embed(x)
        B, L, D = x.shape

        time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim))
        time_token = time_token.unsqueeze(dim=1)
        x = torch.cat((time_token, x), dim=1)

        cls_tokens = self.cls_token.expand(B, -1, -1).to(self.dtype)

        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = x.to(self.dtype)

        for blk in self.blocks:
            x = blk(x)

        if self.output_format == "clstoken":
            x = self.norm(x[:, 0])
        elif self.output_format == "mean":
            # use pooled feature as output
            # this might benefit image denoising tuning
            x = self.norm(x[:, self.extras:]).mean(dim=1)
        
        return x

    return forward


def initialize_train_state(args, accelerator):

    logging.info("creating model and diffusion...")
    device = accelerator.device

    model = create_model(**args.nnet).train() 
    ema_model= create_model(**args.nnet).eval()
    logging.info(f"num parameters: {model.get_num_param()}")

    target_model = None
    if args.train.mode == "consis":
        logging.info("Consistency training: creating target model")
        target_model = create_model(**args.nnet).train()
        for param in target_model.parameters():
            param.requires_grad_(False) # freeze the parameters of target model
 
    # if args.train.freeze_encoder:
    #     for p in model.parameters():
    #         p.requires_grad_(False)
    #     logging.info("freeze all encoder parameters")
    #     logging.info(f"trainable parameters: {[n for n,p in model.named_parameters() if p.requires_grad]}")
    #     optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], **args.optimizer)
    # el
    if args.train.lrd > 0:
        optimizer = torch.optim.AdamW(param_groups_lrd(model, args.optimizer.lr, args.optimizer.weight_decay, layer_decay=args.train.lrd, lrd_mode=args.train.lrd_mode), **args.optimizer)
        logging.info("use smaller lr for RCM encoder parameters")
    else:
        optimizer = torch.optim.AdamW(
            # [p for p in model.parameters() if p.requires_grad],
            param_groups_wd(model, lr=args.optimizer.lr, weight_decay=args.optimizer.weight_decay), # set weight decay
            **args.optimizer)

    train_state = TrainState(step=0, optimizer=optimizer, lr_scheduler=customized_lr_scheduler(optimizer, **args.lr_scheduler),
                            nnet=model, target_model=target_model, nnet_ema=ema_model)

    train_state.to(device)
    accelerator.wait_for_everyone()

    logging.info("synchronizing model parameters...")
    if accelerator.num_processes > 1:
        sync_params(model.parameters())
        sync_params(model.buffers())

    train_state.ema_update(0)
    if args.train.mode == "consis":
        train_state.target_update(0)

    return train_state


def initialize_eval_state(args, accelerator):

    logging.info("creating model and diffusion...")
    device = accelerator.device

    model = create_model(**args.nnet).train() 
    ema_model= create_model(**args.nnet).eval()
    logging.info(f"num parameters: {model.get_num_param()}")

    nnet_poor = None
    if args.sample.poor_path: # path to model of poorer generation performance (e.g., due to less training steps), used in auto-guidance
        nnet_poor = create_model(**args.nnet).eval()
        nnet_poor.load_state_dict(torch.load(args.sample.poor_path, map_location="cpu"))

    eval_state = TrainState(step=0, nnet=model, nnet_ema=ema_model, nnet_poor=nnet_poor)

    eval_state.to(device)
    accelerator.wait_for_everyone()

    return eval_state


def log_loss_dict(diffusion, ts, losses):
    metrics = {}
    for key, values in losses.items():
        if values.ndim == 0:
            metrics[key] = values
        else: # training loss
            metrics[key] = values.mean() # log overall loss
            if "lognormal" in diffusion.time_sample_schedule: # ts might be indices or sigmas
                new_ts = ts[0] if isinstance(ts, tuple) else ts
                new_ts = diffusion.find_nearest(new_ts)
            else:
                new_ts = ts.clone()

            for i in range(4):
                quartile = (4 * new_ts / diffusion.num_timesteps).int()
                if (quartile == i).any():
                    metrics[f"{key}_q{i}"] =  values[quartile == i].mean().detach().cpu()

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