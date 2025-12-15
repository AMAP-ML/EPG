run_args="--main_process_ip localhost \
        --main_process_port 12345 \
        --num_machines 1 \
        --machine_rank 0 \
        --mixed_precision no \
        --num_processes 8 \
"

accelerate launch $run_args sample.py --config=configs/imagenet256_denoise.py \
                --config.sample.num_samples=50000 \
                --config.nnet.model_type="DMViT" \
                --config.nnet.depth=12 \
                --config.nnet.num_heads=12 \
                --config.nnet.embed_dim=768 \
                --config.nnet.decoder_depth=12 \
                --config.nnet.decoder_num_heads=22 \
                --config.nnet.decoder_embed_dim=1584 \
                --config.sample.sampling_step=32 \
                --config.sample.sampler="heun" \
                --config.sample.cfg_scale=2.5 \
                --config.sample.cfg_sigma_low=0.19 \
                --config.sample.cfg_sigma_high=1.61 \
                --config.path="[Path to the checkpoint directory that contains nnet.pth and nnet_ema.pth]"  \
                --config.dataset.data_dir="[Path to the ImageNet256 train subset]" \
                --config.dataset.fid_stat_path="[Path to the reference statistics npz file for FID evaluation, e.g.  VIRTUAL_imagenet256_labeled.npz from ADM]" \
                --config.sample.path="[Path to save the generated images, if empty then the images will be saved to a random temporary path]" 
