import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()
    config.seed = 1234
    config.path = "" # path to pre-trained ckpt

    config.diffusion = d(
        sigma_data = .5,
        sigma_max = 80.0,
        sigma_min = 0.002,
        rescale_t="cm", # [cm, edm]
        shift_sigma=True, # noise shift by simple diffusion: https://arxiv.org/pdf/2301.11093
        prediction_target = "x0",
    )

    config.nnet = d(
        model_type="DMViT",
        in_channels=3,
        mlp_ratio=4,
        qkv_bias=False,
        qk_scale=None,
        mlp_time_embed=False,

        image_size=256,
        patch_size=16,
        depth=12,
        embed_dim=768,
        num_heads=12,
        decoder_depth=12,
        decoder_embed_dim=1584,
        decoder_num_heads=22,

        use_checkpoint= False,
        drop=0., attn_drop=0., drop_path=0.,
        qk_norm=False,
        tokens = 1,

        num_classes = 1000,
        skip_post_norm = True,
    )

    config.dataset = d(
        image_size=256,
        data_dir='path to Imagenet train folder, e.g., /data/ImageNet256/train, we recommend preprocess (e.g., center crop) the data following ADM first',
        batch_size = 1024,
        num_workers = 12,
        num_classes = 1000,
        cfg_drop_rate = 0.0,
        fid_stat_path = "path to reference statistics .npz, e.g., VIRTUAL_imagenet256_labeled.npz from ADM",
        std="imagenet", # different normalization strategy, [imagenet, simple]
        horizontal_flip=True,
    )

    config.sample = d(
        mode = "cond",
        sampler = "euler", # euler/heun
        sampling_step = 50,
        num_samples = 100,
        mini_batch_size = 100,
        img_shape = [3, 256, 256],
        save_sample_traj = False,
        print_loss = False,

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