run_args=" --main_process_ip=127.0.0.1
            --main_process_port=12345 \
            --machine_rank=0 --num_processes=8 --num_machines=1
            --mixed_precision fp16
            "

data_dir="Path to ImageNet256/train"
workdir="Path to save ckpts and logs"
accelerate launch $run_args train.py --config=configs/imagenet256_pretrain.py \
            --workdir=$workdir \
            --config.dataset.batch_size=8 \
            --config.dataset.num_workers=12 \
            --config.nnet.depth=12 \
            --config.nnet.embed_dim=768 \
            --config.nnet.num_heads=12 \
            --config.nnet.patch_size=16 \
            --config.nnet.output_dim=768 \
            --config.nnet.hidden_dim=4096 \
            --config.optimizer.lr=6.0e-4 \
            --config.optimizer.betas=\(0.9,0.95\) \
            --config.optimizer.weight_decay=0.03 \
            --config.dataset.data_dir=$data_dir