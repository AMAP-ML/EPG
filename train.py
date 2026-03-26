import os
import shutil
import functools
from absl import flags
from absl import app
from pathlib import Path
from tqdm import tqdm
from absl import logging
import multiprocessing as mp
from ml_collections import config_flags

import torch
import accelerate
from accelerate import DistributedDataParallelKwargs, GradScalerKwargs

import rcm.utils as utils
from rcm.dataset import load_data


def train(args):

    mp.set_start_method('spawn')
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    amp_kwargs = GradScalerKwargs(init_scale=2**14, growth_factor=1.0006933874625807, growth_interval=1, backoff_factor=0.5)
    accelerator = accelerate.Accelerator(split_batches=True, mixed_precision="fp16", kwargs_handlers=[ddp_kwargs, amp_kwargs])
    device = accelerator.device if accelerator.num_processes > 1 else "cuda:0"
    torch.cuda.set_device(device)
    logging.info(f"Training in: {accelerator.mixed_precision} mode")

    accelerate.utils.set_seed(args.seed, device_specific=True)
    if accelerator.is_main_process:
        logging.info(f"logging to {args.workdir}")
        os.makedirs(args.workdir, exist_ok=True)
        exp_name = args.workdir.split("/")[-1]
        utils.set_logger(log_level='info', fname=os.path.join(args.workdir, 'output.log'))
        logging.info(args)
    else:
        os.makedirs(args.workdir, exist_ok=True)
        utils.set_logger(log_level='error', fname=os.path.join(args.workdir, 'error.log'))

    logging.info(f'Process {accelerator.process_index} using device: {device} world size: {accelerator.num_processes}')
    utils.setup_for_distributed(accelerator.is_main_process)

    data_loader = load_data(**args.dataset)
    train_state = utils.initialize_train_state(args, accelerator)
    # automatically resume from existing checkpoints in current workdir
    train_state.resume(args.workdir)

    lr_scheduler = train_state.lr_scheduler
    model, optimizer, target_model, data_loader = accelerator.prepare(train_state.nnet, train_state.optimizer, train_state.target_model, data_loader)

    def get_data_generator():
        while True:
            for data in tqdm(data_loader, disable=not accelerator.is_main_process, desc='epoch'):
                    yield data
    data_generator = get_data_generator()

    ema_scale_fn = utils.create_ema_and_scales_fn(**args.ema_scale)
    _, num_scales, *_ = ema_scale_fn(train_state.step)
    diffusion = utils.create_diffusion(
        **args.diffusion, 
        num_timesteps=num_scales, device=device
    )

    def train_step(batch):

        num_scales, tau = ema_scale_fn(train_state.step)
        diffusion.set_scale(num_scales)
        optimizer.zero_grad()

        x_start, x_aug, x_aug2 = batch
        bs = x_start.shape[0]
        indices = diffusion.sample_time(bs)

        if accelerator.num_processes > 1:
            torch.distributed.broadcast(indices, 0) # use exactly the same t across all processes

        compute_losses = functools.partial(
            diffusion.PreTrainStep,
            model,
            x_start,
            indices=indices,
            target_model=target_model,
            x_aug = x_aug,
            x_aug2 = x_aug2,
            accelerator=accelerator,
            tau = tau,
        )

        with accelerator.autocast():
            losses = compute_losses()

        loss = (losses["loss"]).mean()
        accelerator.backward(loss)

        optimizer.step()
        if accelerator.optimizer_step_was_skipped:
            # inf encoutered in mixed_precision training, skip the update of all train states
            logging.info("Found Inf grad, skip this iteration..")
            return None

        lr_scheduler.step()

        train_state.target_update(args.train.target_ema)
        train_state.step += 1

        grad_norm, param_norm = utils._compute_norms(accelerator.unwrap_model(model).named_parameters())
        metrics = utils.log_loss_dict(
            diffusion, indices, {k: v.clone().detach() for k, v in losses.items()}
        )
        metrics.update({
            "grad_norm": grad_norm,
            "param_norm": param_norm,
            "tau": tau,
        })
        return metrics

    metric_logger = utils.MetricLogger()
    logging.info("training...") 

    while (
        train_state.step < args.train.total_training_steps
    ): 

        batch = next(data_generator)
        metrics = train_step(batch)

        if train_state.step == args.lr_scheduler.warmup_steps:
            # warmup end
            logging.info("Warmup ended")
            train_state.is_warmup = False

        if metrics is not None:
            metric_logger.update(metrics)
            metric_logger.add({"num_scales": diffusion.num_timesteps, "bs": args.dataset.batch_size})

        if (
           train_state.step % args.train.save_interval == 0 and accelerator.is_main_process
        ): 
            save_dir = os.path.join(args.workdir, str(train_state.step)+".ckpt")
            os.makedirs(save_dir, exist_ok=True)
            train_state.save(save_dir)
            torch.cuda.empty_cache()

        elif (
            train_state.step != 0 and train_state.step % 10000 == 0 and accelerator.is_main_process
        ):
            save_dir = os.path.join(args.workdir, "latest.ckpt")
            os.makedirs(save_dir, exist_ok=True)
            train_state.save(save_dir)
            torch.cuda.empty_cache()

        if train_state.step % args.train.log_interval == 0 and accelerator.is_main_process:
            logging.info(utils.dct2str(dict(step=train_state.step,lr=train_state.optimizer.param_groups[0]['lr'], **metric_logger.get())))
            metric_logger.clean()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        try:
            latest_ckpt_dir = os.path.join(args.workdir, "latest.ckpt")
            shutil.rmtree(latest_ckpt_dir)
        except:
            pass


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.DEFINE_string("workdir", None, "Work unit directory.")
flags.mark_flags_as_required(["config"])


def main(argv):
    config = FLAGS.config
    config.workdir = FLAGS.workdir
    train(config)

if __name__ == "__main__":
    app.run(main)
