run_args="--main_process_ip localhost \
        --main_process_port 12345 \
        --num_machines 1 \
        --machine_rank 0 \
        --mixed_precision no \
        --num_processes 8 \
"
path="[Path to the checkpoint directory that contains nnet.pth and nnet_ema.pth]"
data_dir="[Path to the ImageNet512 train subset]"
fid_stat_path="[Path to the reference statistics npz file for FID evaluation, e.g.  VIRTUAL_imagenet256_labeled.npz from ADM]"
sample_output_path="[Path to save the generated images, if empty then the images will be saved to a random temporary path]"

accelerate launch $run_args sample.py --config=configs/imagenet512_denoise.py \
                --config.sample.num_samples=50000 \
                --config.nnet.model_type="EPGViT" \
                --config.nnet.depth=16 \
                --config.nnet.num_heads=16 \
                --config.nnet.embed_dim=1024 \
                --config.nnet.decoder_depth=16 \
                --config.nnet.decoder_num_heads=16 \
                --config.nnet.decoder_embed_dim=1024 \
                --config.sample.sampling_step=32 \
                --config.sample.sampler="heun" \
                --config.sample.cfg_scale=3.2 \
                --config.sample.cfg_sigma_low=0.19 \
                --config.sample.cfg_sigma_high=1.61 \
                --config.path=$path  \
                --config.dataset.data_dir=$data_dir \
                --config.dataset.fid_stat_path=$fid_stat_path \
                --config.sample.path=$sample_output_path \
