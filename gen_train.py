import os
import time
import shutil
import functools
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

from absl import flags
from absl import app
from absl import logging
from pathlib import Path
from ml_collections import config_flags

import torch
import torch.distributed
from torchvision.utils import save_image

import accelerate
from accelerate import DistributedDataParallelKwargs, GradScalerKwargs

import rcm.utils as utils
import rcm.gen_train_utils as gen_train_utils
from rcm.gen_dataset import load_data


def train(args):

    mp.set_start_method('spawn')
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    amp_kwargs = GradScalerKwargs(init_scale=2**14, growth_factor=1.0006933874625807, growth_interval=1, backoff_factor=0.5)
    accelerator = accelerate.Accelerator(split_batches=True, kwargs_handlers=[ddp_kwargs, amp_kwargs])
    device = accelerator.device if accelerator.num_processes > 1 else "cuda:0"

    torch.cuda.set_device(device)
    logging.info(f"Training in: {accelerator.mixed_precision} mode")

    accelerate.utils.set_seed(args.seed, device_specific=True)
    if accelerator.is_main_process:
        os.makedirs(args.workdir, exist_ok=True)
        os.makedirs(args.sample_dir, exist_ok=True)
        utils.set_logger(log_level='info', fname=os.path.join(args.workdir, 'output.log'))
        logging.info(args)
    else:
        os.makedirs(args.workdir, exist_ok=True)
        utils.set_logger(log_level='error', fname=os.path.join(args.workdir, 'error.log'))

    logging.info(f'Process {accelerator.process_index} using device: {device} world size: {accelerator.num_processes}')
    utils.setup_for_distributed(accelerator.is_main_process)

    dataset, data_loader = load_data(**args.dataset)
    train_state = gen_train_utils.initialize_train_state(args, accelerator)

    if os.path.exists(args.path):
        train_state.load(args.path) # load RCM pre-trained encoder
        # synchronize param
        logging.info("synchronize ema model parameters with those of online model")
        train_state.ema_update(0.0) # 0.9999
        if args.train.mode == "consis":
            logging.info("synchronize target model parameters with those of online model")
            train_state.target_update(0.0)

    else:
        ret = train_state.resume(args.workdir)# resume from given path or current working directory 
        if ret == 1: 
            # when successfully resume from weights
            # overwrite optimizer & lr scheduler's state dict with config settings

            sd = train_state.optimizer.state_dict()
            no_decay_params = train_state.nnet.no_weight_decay()
            logging.info(f"detected no weight decay params:{no_decay_params}")
            for k,v in args.optimizer.items():
                for i in range(len(sd["param_groups"])):
                    if k == "weight_decay" and len(sd["param_groups"][i]["params"]) == len(no_decay_params):
                        # this is the param group that has no weight decay, set wd to 0 then skip
                        sd["param_groups"][i][k] = 0.0
                        continue
                    sd["param_groups"][i][k] = v

            num_param_groups = len(sd["param_groups"])
            train_state.optimizer.load_state_dict(sd)
            sd = train_state.lr_scheduler.state_dict()
            new_lr = args.optimizer.lr

            sd["base_lrs"] = [new_lr]*num_param_groups
            sd['_last_lr'] = [new_lr]*num_param_groups
            train_state.lr_scheduler.load_state_dict(sd)
            logging.info("updated optimizer and lr scheduler settings: ")
            logging.info(train_state.optimizer.state_dict()["param_groups"])
            logging.info(train_state.lr_scheduler.state_dict())

    lr_scheduler = train_state.lr_scheduler
    model, optimizer, data_loader = accelerator.prepare(train_state.nnet, train_state.optimizer, data_loader)

    def get_data_generator():
        inner_epoch = 0
        while True:
            for data in tqdm(data_loader, disable=not accelerator.is_main_process, desc='epoch'):
                yield data
            inner_epoch += 1

    data_generator = get_data_generator()

    perceptual_model = None # used in consistency fine-tuning
    if getattr(args.train, "perceptual_model_path", None):
        _ckpt_path = os.path.join(args.train.perceptual_model_path, "nnet.pth")
        assert os.path.exists(_ckpt_path), f"perceptual model path does not exist: {args.train.perceptual_model_path}"
        arch_dict = args.nnet
        arch_dict["model_type"] = "RCMViT" # use pre-training model architecture
        _model = gen_train_utils.create_model(**arch_dict).to(device)
        # args.path to the directory containing the pre-trained nnet.pth
        missing, unexpected = _model.load_state_dict(torch.load(_ckpt_path, map_location="cpu"), strict=False)
        logging.info(f"Loaded perceptual model, missing:{missing}, unexpected:{unexpected}")
        _model.eval()

        def fn(self):
            def inference(x, t, **kwargs):
                return self.forward_features(x, t)
            return inference

        perceptual_model = fn(_model)

    diffusion = gen_train_utils.create_diffusion(**args.diffusion, num_timesteps=args.ema_scale.start_scales, device=device)
    diffusion.set_scale(args.ema_scale.start_scales)

    def train_step(batch):

        x_start, model_kwargs = batch
        bs = x_start.shape[0]
        indices_or_sigmas = diffusion.sample_time(bs, step=train_state.step)

        if args.train.mode == "consis":
            _stage = 0
            if train_state.step > args.diffusion.sample_param.d[0]:
                _stage = (train_state.step - (args.diffusion.sample_param.d[0] - args.diffusion.sample_param.d[1])) // args.diffusion.sample_param.d[1] + args.diffusion.sample_param.delta
            else:
                _stage =  args.diffusion.sample_param.delta

            with accelerator.autocast():
                ret = diffusion.ConsisTuneStep(
                    model,
                    x_start,
                    sigmas=indices_or_sigmas,
                    model_kwargs=model_kwargs,
                    target_model=train_state.target_model,
                    perceptual_model=perceptual_model,
                    accelerator=accelerator,
                    ect_sample_stage=_stage,
                )
                loss = ret["loss"].mean()
        elif args.train.mode == "denoise":
            with accelerator.autocast():
                ret = diffusion.DenoiseTuneStep(
                    model,
                    x_start,
                    sigmas=indices_or_sigmas,
                    model_kwargs=model_kwargs,
                )
                loss = ret["loss"].mean()
        else:
            raise NotImplementedError(f"Unknown training mode: {args.train.mode}")

        optimizer.zero_grad()
        accelerator.backward(loss)
        if not train_state.is_warmup and args.train.gradclip > 0:
            accelerator.clip_grad_norm_(model.parameters(), args.train.gradclip)
        optimizer.step()

        if accelerator.optimizer_step_was_skipped:
            logging.info("Found Inf grad, skip this iteration..")
            return None

        lr_scheduler.step()
        train_state.ema_update(args.train.ema_rate) # 0.9999
        if args.train.mode == "consis":
            train_state.target_update(0.0)
        train_state.step += 1

        # print("before logging")
        grad_norm, param_norm = gen_train_utils._compute_norms(accelerator.unwrap_model(model).named_parameters())
        metrics = gen_train_utils.log_loss_dict(
            diffusion, indices_or_sigmas, {k: v.clone().detach() for k, v in ret.items() if isinstance(v, torch.Tensor)},
        )
        metrics.update({
            "grad_norm": grad_norm,
            "param_norm": param_norm,
        })

        return metrics

    @torch.no_grad()
    def ema_forward(x, t, **model_kwargs):
        return train_state.nnet_ema(x, t, **model_kwargs)

    @torch.no_grad()
    def sample_step(step):
        steps = args.sample.sampling_step
        batch_size = args.sample.mini_batch_size
        sampler = args.sample.sampler

        batch_size_lst = [batch_size for i in range(args.sample.num_samples // batch_size)]
        batch_size_lst += [] if args.sample.num_samples % batch_size == 0 else [args.sample.num_samples % batch_size]
        samples = []
        for bs in batch_size_lst:
            model_kwargs = dict(y=dataset.sample_label(bs, device=accelerator.device))
            with accelerator.autocast():
                mini_sample = diffusion.sample(
                    ema_forward,
                    bs,
                    dataset.data_shape(),
                    steps,
                    sampler,
                    model_kwargs=model_kwargs,
                    return_sample_traj=args.sample.save_sample_traj,
                )
            samples.append(mini_sample)

        samples = torch.cat(samples, dim=0)
        samples = dataset.unpreprocess(samples)
        try:
            save_image(samples, os.path.join(args.sample_dir, f"{step}.jpg"), nrow=10)
        except PermissionError as e:
            return

    metric_logger = utils.MetricLogger()
    logging.info("training...") 

    while (
        train_state.step < args.train.total_training_steps
    ):  # keep training until interrupted.

        batch = next(data_generator)
        metrics = train_step(batch)

        if train_state.step == args.lr_scheduler.warmup_steps:
            # warmup end
            logging.info("Warmup ended")
            train_state.is_warmup = False

        if metrics is not None:
            metric_logger.update(metrics)
            metric_logger.add({"num_scales": diffusion.num_timesteps, "bs": args.dataset.batch_size})

        # save ckpt periodically
        if train_state.step != 0:
            if (
            train_state.step % args.train.save_interval == 0 and accelerator.is_main_process
            ): 
                save_dir = os.path.join(args.workdir, str(train_state.step)+".ckpt")
                os.makedirs(save_dir, exist_ok=True)
                train_state.save(save_dir)
                torch.cuda.empty_cache()

            elif (
                train_state.step % 10000 == 0 and accelerator.is_main_process
            ):
                save_dir = os.path.join(args.workdir, "latest.ckpt")
                os.makedirs(save_dir, exist_ok=True)
                train_state.save(save_dir)
                torch.cuda.empty_cache()

        # logging
        if train_state.step % args.train.log_interval == 0 and accelerator.is_main_process:
            logging.info(dict(
                step=train_state.step,
                lr=lr_scheduler.get_last_lr(),
                **{
                    k: f"{v:.6g}" for k,v in metric_logger.get().items()
                }
            ))
            metric_logger.clean()

        # sample periodically
        if train_state.step!=0 and train_state.step % args.train.sample_interval == 0 and accelerator.is_main_process:
            samples = sample_step(train_state.step)

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
    config.sample_dir = os.path.join(config.workdir, f"samples")
    train(config)

if __name__ == "__main__":
    app.run(main)
