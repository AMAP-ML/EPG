import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()
    config.seed = 1234
    config.path = "" # path to pre-trained weights: **/nnet.pth
    _train_steps = 1000000

    config.train = d(
        total_training_steps=_train_steps,
        ema_rate = 0.9999, # ema rate for EMA model
        log_interval=10,
        save_interval=50000,

        gradclip = 0.0,
        lrd=0.0, # per-layer learning rate decay coefficient
        mode="consis",
        sample_interval=5000,
        perceptual_model_path="path to perceptual model",
    )

    config.lr_scheduler = d(
        name="warmup-step", # ["wamrup-step", "warmup-cosine"]
        warmup_steps = 10000, # lr scheduler
        total_training_steps=_train_steps,
        min_scale=-1.,
    )

    config.optimizer = d(
        lr = 1e-4,
        betas=(0.9, 0.99),
        weight_decay = 0.03,
    )

    config.ema_scale = d(
        # total time discretization steps
        start_scales=2560,
        end_scales=2560,
        total_steps=_train_steps,
    )

    config.diffusion = d(
        sigma_data = .5,
        sigma_max = 80.0,
        sigma_min = 0.002,
        prediction_target = "edm",

        # used when computing perceptual loss
        tmin_tau=0.1, # tau value for consistency training at sigma_min (0.002 by default)
        tau_schedule="constant",  # [interpolated, constant]

        time_sample_schedule = "lognormal-ect",
        sample_param = d(
            p_mean = -0.4,
            p_std = 1.6,
            d = (100000,100000),
            q = 2,
            delta = 1,
            k = 8,
            b = 1,
        ),
        pseudo_huber_c=0.06,
    )

    config.nnet = d(
        model_type="EPGViT",
        in_channels=3,
        mlp_ratio=4,
        qkv_bias=False,
        qk_scale=None,

        image_size=256,
        patch_size=16,
        depth=16,
        embed_dim=1024,
        num_heads=16,
        decoder_depth=16,
        decoder_embed_dim=1024,
        decoder_num_heads=16,

        drop=0., attn_drop=0., drop_path=0.,
        qk_norm=False,
        tokens = 1,

        num_classes = 1000,
        skip_post_norm = True,
    )

    config.dataset = d(
        image_size=256,
        data_dir='PATH TO ImageNet256/train',
        batch_size = 1024,
        num_workers = 12,
        num_classes = 1000,
        cfg_drop_rate = 0.0,
        fid_stat_path = "path to fid stats", # enabled during sampling, after which the FID will be computed automatically.
        std="imagenet", # normalize image data with statistics from [imagenet, simple], where `simple` means normalize via x*2-1
        horizontal_flip=True,
    )

    config.sample = d(
        sampler = "onestep", # euler/heun
        sampling_step = 1,
        num_samples = 100,
        mini_batch_size = 100,

        save_sample_traj = False,
        print_loss = False,

        path = "", # path to save sampled images, if empty, then temporary path will be used
    )

    return config