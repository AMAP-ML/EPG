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
accelerate launch $run_args gen_train.py --config=configs/imagenet256_denoise.py \
                                            --workdir=$workdir \
                                            --config.nnet.model_type="EPGViT" \
                                            --config.nnet.decoder_depth=12 \
                                            --config.nnet.decoder_embed_dim=1584 \
                                            --config.nnet.decoder_num_heads=22 \
                                            --config.nnet.depth=12 \
                                            --config.nnet.embed_dim=768 \
                                            --config.nnet.num_heads=12 \
                                            --config.dataset.data_dir=$data_dir \
                                            --config.path=$path \
                                            --config.dataset.batch_size=1024