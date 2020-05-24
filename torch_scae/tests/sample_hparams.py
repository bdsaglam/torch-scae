model_params = dict(
    image_shape=(1, 28, 28),
    n_classes=10,
    n_part_caps=40,
    n_obj_caps=32,
    pcae_cnn_encoder_params=dict(
        out_channels=[128] * 4,
        kernel_sizes=[3, 3, 3, 3],
        strides=[2, 2, 1, 1],
        activate_final=True
    ),
    pcae_encoder_params=dict(
        n_poses=6,
        n_special_features=16,
        similarity_transform=False,
    ),
    pcae_template_generator_params=dict(
        template_size=(11, 11),
        template_nonlin='sigmoid',
        colorize_templates=True,
        color_nonlin='sigmoid',
    ),
    pcae_decoder_params=dict(
        learn_output_scale=False,
        use_alpha_channel=True,
        background_value=True,
    ),
    ocae_encoder_set_transformer_params=dict(
        n_layers=3,
        n_heads=1,
        dim_hidden=16,
        dim_out=256,
        layer_norm=True,
    ),
    ocae_decoder_capsule_params=dict(
        dim_caps=32,
        hidden_sizes=(128,),
        caps_dropout_rate=0.0,
        learn_vote_scale=True,
        allow_deformations=True,
        noise_type='uniform',
        noise_scale=4.,
        similarity_transform=False,
    ),
    scae_params=dict(
        vote_type='enc',
        presence_type='enc',
        stop_grad_caps_input=True,
        stop_grad_caps_target=True,
        caps_ll_weight=1.,
        cpr_dynamic_reg_weight=10,
        prior_sparsity_loss_type='l2',
        prior_within_example_sparsity_weight=2.0,
        prior_between_example_sparsity_weight=0.35,
        posterior_sparsity_loss_type='entropy',
        posterior_within_example_sparsity_weight=0.7,
        posterior_between_example_sparsity_weight=0.2,
    )
)
