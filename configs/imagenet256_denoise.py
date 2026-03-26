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
        group_wd = False, # if True, then weight decay of cls_token and pos_embed will be set to 0
        gradclip = 0.5,
        mode="denoise",
        lrd=0.0,
        sample_interval=5000,
    )

    config.lr_scheduler = d(
        name="warmup-cosine", # ["wamrup-step", "warmup-cosine"]
        warmup_steps = 10000, # lr scheduler
        total_training_steps=_train_steps,
        min_scale=-1.,
    )

    config.optimizer = d(
        lr = 1e-4,
        betas=(0.9, 0.999),
        weight_decay = 0.01,
    )

    config.ema_scale = d(
        # total time discretization steps
        start_scales=1280,
        end_scales=1280,
        total_steps=_train_steps,
    )

    config.diffusion = d(
        sigma_data = .5,
        sigma_max = 80.0,
        sigma_min = 0.002,
        prediction_target = "x0",
        time_sample_schedule = "lognormal",
        sample_param = d(
            p_mean = -1.2,
            p_std = 1.6,
        )
    )

    config.nnet = d(
        model_type="EPGViT",
        in_channels=3,
        mlp_ratio=4,
        qkv_bias=False,
        qk_scale=None,

        image_size=256,
        patch_size=16,
        depth=12,
        embed_dim=768,
        num_heads=12,
        decoder_depth=12,
        decoder_embed_dim=1584,
        decoder_num_heads=22,

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
        cfg_drop_rate = 0.1,
        fid_stat_path = "path to fid stats", # enabled during sampling, after which the FID will be computed automatically.
        std="imagenet", # normalize image data with statistics from [imagenet, simple], where `simple` means normalize via x*2-1
        horizontal_flip=True,
    )

    config.sample = d(
        sampler = "euler", # euler/heun
        sampling_step = 50,
        num_samples = 10000,
        mini_batch_size = 100,

        save_sample_traj = False,
        print_loss = False,
        balanced_sampling=False,

        path = "", # path to save sampled images, if empty, then temporary path will be used

        cfg_scale = -1., # cfg sampling enabled when cfg_scale>0
        cfg_sigma_high=1.61, # inclusive upper bound of interval cfg
        cfg_sigma_low=0.19, # exclusive lower bound of interval cfg

        autog_scale = -1., 
        poor_path = "", # enabled when autog_scale > 0 path to ckpt of poorer performance

        edm_sde_param = d(
            S_churn=40,
            S_min=0.05,
            S_max=50,
            S_noise=1.003,
        )
    )

    return config