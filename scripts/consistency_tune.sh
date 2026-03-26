run_args=" --main_process_ip=127.0.0.1
            --main_process_port=12345 \
            --machine_rank=0 --num_processes=8 --num_machines=1
            --mixed_precision fp16
            "

export TORCH_DISTRIBUTED_DEBUG=OFF
export NCCL_DEBUG=OFF
export OMP_NUM_THREADS=32 
path="Path to the pre-trained checkpoint directory that contains nnet.pth"
data_dir="Path to ImageNet256/train"
workdir="Path to save ckpts and logs"
accelerate launch $run_args gen_train.py --config=configs/imagenet256_consistency.py \
                                            --workdir=$workdir \
                                            --config.dataset.batch_size=256 \
                                            --config.dataset.num_workers=12 \
                                            --config.nnet.model_type="EPGViT" \
                                            --config.nnet.depth=16 \
                                            --config.nnet.embed_dim=1024 \
                                            --config.nnet.num_heads=16 \
                                            --config.nnet.decoder_depth=16 \
                                            --config.nnet.decoder_embed_dim=1024 \
                                            --config.nnet.decoder_num_heads=16 \
                                            --config.nnet.drop=0.5 \
                                            --config.dataset.data_dir=$data_dir \
                                            --config.train.perceptual_model_path=$path \
                                            --config.path=$path