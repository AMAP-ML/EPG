import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    # global variables
    _image_size = 256
    _train_steps = 600000
    config.seed = 1234

    config.train = d(
        total_training_steps=_train_steps,
        ema_rate = 0.9999, # ema rate for EMA model, not used in pre-training
        log_interval=10,
        save_interval=50000,
        group_wd = False, # if True, then weight decay of cls_token and pos_embed will be set to 0
        target_ema=0.99, # ema for target model
    )

    config.lr_scheduler = d(
        name="warmup-cosine",
        warmup_steps = 10000, # lr scheduler
        total_training_steps=_train_steps,
        min_scale=-1.,
    )

    config.optimizer = d(
        lr = 2e-4,
        betas=(0.9, 0.95),
        weight_decay = 0.03,
    )

    config.ema_scale = d(
        # total time discretization steps
        start_scales=20,
        end_scales=1280,
        total_steps=_train_steps,
        # tau schedule
        tau_schedule="constant", # [cosine, constant]
        start_tau = 0.1,
        end_tau = 0.1,
    )

    config.diffusion = d(
        time_sample_schedule = "unique",
        sigma_data = .5,
        sigma_max = 80.0,
        sigma_min = 0.002,
        rho=7.0,
        rescale_t="cm",

        tmin_tau=0.1, # tau value for consistency training at sigma_min (0.002 by default)
        tau_schedule="constant",  # [interpolated, constant]
    )

    config.nnet = d(
        model_type="RCMViT",
        in_channels=3,
        mlp_ratio=4,
        qkv_bias=False,
        qk_scale=None,

        image_size=_image_size,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,

        hidden_dim=4096,
        output_dim=256,

        drop=0., attn_drop=0., drop_path=0.,
        tokens = 1, # extra learnable tokens
        proj_layers=3, # projection layers
    )

    config.dataset = d(
        image_size=_image_size,
        data_dir='[PATH to ImageNet256/train]',
        batch_size = 1024,
        num_workers = 12,

        load_from_memory = False, # if store image data on RAM, useful when training on cloud server
        std="imagenet", # normalize image data with statistics from [imagenet, simple], where `simple` means normalize via x*2-1
    )

    return config