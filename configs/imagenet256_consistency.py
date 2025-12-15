import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()
    config.seed = 1234
    config.path = ""

    config.diffusion = d(
        sigma_data = .5,
        sigma_max = 80.0,
        sigma_min = 0.002,
        rescale_t="cm", # [cm, edm]
        shift_sigma=True, # noise shift by simple diffusion: https://arxiv.org/pdf/2301.11093
        prediction_target = "edm",
    )

    config.nnet = d(
        model_type="DMViT",

        in_channels=3,
        mlp_ratio=4,
        qkv_bias=False,
        qk_scale=None,
        mlp_time_embed=False,
        drop=0., attn_drop=0., drop_path=0.,
        qk_norm=False,
        tokens = 1,
        num_classes = -1,

        image_size=256,
        patch_size=16,
        depth=16,
        embed_dim=1024,
        num_heads=16,
        decoder_depth=16,
        decoder_embed_dim=1024,
        decoder_num_heads=16,

        use_checkpoint= False,
        skip_post_norm = True,
    )

    config.dataset = d(
        image_size=256,
        data_dir='path to Imagenet train folder, e.g., /data/ImageNet256/train, we recommend preprocess (e.g., center crop) the data following ADM first',
        batch_size = 1024,
        num_workers = 12,
        num_classes = 1000,
        cfg_drop_rate = 0.0,
        fid_stat_path = "path to reference statistics .npz, for example VIRTUAL_imagenet256_labeled.npz from ADM",
    )

    config.sample = d(
        mode = "cond", # [uncond, cond], sampling mode
        sampler = "onestep", # huen/euler 
        sampling_step = 1,
        num_samples = 10000,
        mini_batch_size = 100,
        img_shape = [3, 256, 256],
        save_sample_traj = False,
        print_loss = False,

        path = "", # save image to the path
        cfg_scale = -1., # cfg guidance scale
        autog_scale = -1., # auto guidance
        poor_path = "", # enabled when autog_scale > 0, path to ckpt of poorer performance
    )

    return config