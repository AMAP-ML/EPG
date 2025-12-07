import os
import math
import functools
import tempfile
from tqdm import tqdm
from ml_collections import config_flags

from absl import app
from absl import flags
from absl import logging

import torch
from torchvision.utils import save_image
import accelerate
import multiprocessing as mp

import rcm.gen_train_utils as gen_train_utils
from rcm.gen_dataset import load_data
from evaluations.fid_score import calculate_fid_given_paths


def eval(args):
    """ 
        step 1: prepare model, data loader, diffusion, and initializing accelerator
    """
    mp.set_start_method('spawn')
    accelerator = accelerate.Accelerator(split_batches=False, mixed_precision="no")
    device = accelerator.device

    if accelerator.num_processes > 1:
        torch.cuda.set_device(device)
    
    logging.info(f"Sampling in: {accelerator.mixed_precision} mode")

    accelerate.utils.set_seed(args.seed, device_specific=True)
    gen_train_utils.setup_for_distributed(accelerator.is_main_process)
    if not accelerator.is_main_process:
        gen_train_utils.set_logger(log_level='error', fname=None)

    args.dataset.batch_size = args.sample.mini_batch_size
    dataset, _ = load_data(**args.dataset, rank=accelerator.process_index, world_size=accelerator.num_processes)

    eval_state = gen_train_utils.initialize_eval_state(args, accelerator) # initialize models, optimizer, and lr scheduler
    eval_state.load(args.path)
    diffusion = gen_train_utils.create_diffusion(**args.diffusion, num_timesteps=1280, device=device)

    """ 
        step2: prepare sampling function
    """
    ema_model = eval_state.nnet_ema
    poor_ema_model = eval_state.nnet_poor

    @torch.no_grad()
    def simple_forward(x, t, **model_kwargs):
        return ema_model(x, t, **model_kwargs)

    uncond_token =  dataset.get_uncond_token(bs=args.sample.mini_batch_size, device=device)
    @torch.no_grad()
    def cfg_forward(x, t, **model_kwargs):
        # print(model_kwargs)
        cond = ema_model(x, t, **model_kwargs)
        if args.diffusion.shift_sigma:
            sigma_low = args.sample.cfg_sigma_low*x.shape[-2]/64
            sigma_high = args.sample.cfg_sigma_high*x.shape[-2]/64
        else:
            sigma_low = args.sample.cfg_sigma_low
            sigma_high = args.sample.cfg_sigma_high

        t_value = t[0].cpu().item()
        unscaled_t = math.exp(t_value/250) - 1e-44 if args.diffusion.rescale_t == "cm" else math.exp(t_value*4) - 1e-44
        if unscaled_t > sigma_low and unscaled_t <= sigma_high:
            print(sigma_low, sigma_high, unscaled_t)
            uncond = ema_model(x, t, y=uncond_token) # unconditional token
            return uncond + args.sample.cfg_scale*(cond-uncond)
        else:
            return cond

    @torch.no_grad()
    def autog_forward(x, t, **model_kwargs):
        cond = ema_model(x, t, **model_kwargs)
        uncond = poor_ema_model(x, t, **model_kwargs)
        return cond + args.sample.autog_scale*(cond-uncond)

    if args.sample.cfg_scale > 0:
        forward_fn = cfg_forward
    elif args.sample.autog_scale > 0:
        forward_fn = autog_forward
    else:
        forward_fn = simple_forward

    """ 
        step3: start sampling. The samples will be saved sequentially after all sampling ends.
    """
    def sample(temp_path):
        path = args.sample.path or temp_path # path to save generated images

        steps = args.sample.sampling_step
        batch_size = args.sample.mini_batch_size
        sampler = args.sample.sampler

        global_batch_size = batch_size * accelerator.num_processes
        batch_size_lst = [batch_size for i in range(args.sample.num_samples // global_batch_size)]
        if args.sample.num_samples % global_batch_size !=0:
            batch_size_lst += [batch_size]

        # samples = []
        sample_fn = functools.partial(
            diffusion.sample,
            model=forward_fn,
            shape=dataset.data_shape(),
            steps=steps,
            sampler=sampler,
            return_sample_traj=args.sample.save_sample_traj,
            print_loss = args.sample.print_loss,
        )

        idx = 0 # used to specify image name when saving images sequentially
        for bs in tqdm(batch_size_lst, disable=(not accelerator.is_main_process)):
            if bs == 0: continue
            if "imagenet" in args.dataset.name and args.nnet.num_classes > 0:
                if args.sample.mode == "uncond":
                    model_kwargs = dict(y=uncond_token)
                elif args.sample.mode == "cond":
                    model_kwargs = dict(y=dataset.sample_label(bs, device=accelerator.device))
            else:
                model_kwargs = dict()

            mini_sample = sample_fn(bs=bs, model_kwargs=model_kwargs, **args.sample.get("edm_sde_param", {}))
            all_sample = accelerator.gather(mini_sample) # shape: (bs*num_processes, c, h, w)
            all_sample = dataset.unpreprocess(all_sample) if dataset else (all_sample + 1 ) / 2
            all_sample = all_sample.cpu()

            if args.sample.save_sample_traj:
                # For debug only: save the whole sampling trajectory, which contains `args.sample.batch_size * args.sample.sampling_step` data points
                assert os.path.exists(args.sample.path), "path to save sampling trajectories must be specified" 
                # ensure the path to save sampling trajectories is given instead of a temporal directory
                os.makedirs(path, exist_ok=True)
                torch.save(all_sample.cpu(), os.path.join(path, f"traj_{idx}.pt"))
                idx += 1
            else:
                all_sample = all_sample[:min(args.sample.num_samples - idx, bs*accelerator.num_processes)]
                if accelerator.is_main_process:
                    os.makedirs(path, exist_ok=True)
                    for sample in all_sample:
                        save_image(sample, os.path.join(path, f"{idx}.png"))
                        idx += 1

        if accelerator.is_main_process:
            print("Computing fid score...")
            fid = calculate_fid_given_paths((args.dataset.fid_stat_path, path)) # by default, the bs is 50
            print(f"fid score:{fid}")

    with tempfile.TemporaryDirectory() as temp_path:
        sample(temp_path)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.DEFINE_string("workdir", None, "Work unit directory.")
flags.mark_flags_as_required(["config"])


def main(argv):
    config = FLAGS.config
    config.workdir = FLAGS.workdir
    eval(config)


if __name__ == "__main__":
    app.run(main)