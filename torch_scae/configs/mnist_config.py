import torch
from monty.collections import AttrDict

image_shape = (1, 28, 28)
n_classes = 10
n_obj_caps = 32

pcae_cnn_encoder = AttrDict(
    input_shape=image_shape,
    out_channels=[128] * 4,
    kernel_sizes=[3, 3, 3, 3],
    strides=[2, 2, 1, 1],
    activate_final=True
)

pcae_primary_capsule = AttrDict(
    input_shape=image_shape,
    n_caps=40,
    n_poses=6,
    n_special_features=16,
    similarity_transform=False,
)

pcae_template_decoder = AttrDict(
    n_templates=pcae_primary_capsule.n_caps,
    template_size=(11, 11),
    output_size=image_shape[1:],
    n_channels=1,
    n_template_features=pcae_primary_capsule.n_special_features,
    learn_output_scale=False,
    colorize_templates=True,
    use_alpha_channel=True,
    template_nonlin=torch.sigmoid,
    color_nonlin=torch.sigmoid,
)

ocae_encoder_set_transformer = AttrDict(
    n_layers=3,
    n_heads=1,
    dim_in=pcae_template_decoder.n_template_features,
    dim_hidden=16,
    dim_out=256,
    n_outputs=n_obj_caps,
    layer_norm=True,
)

ocae_decoder_capsule = AttrDict(
    n_caps=ocae_encoder_set_transformer.n_outputs,
    dim_feature=ocae_encoder_set_transformer.dim_out,
    n_votes=pcae_template_decoder.n_templates,
    n_caps_params=32,
    hidden_sizes=(128,),
    caps_dropout_rate=0.0,
    learn_vote_scale=True,
    deformations=True,
    noise_type='uniform',
    noise_scale=4.,
    similarity_transform=False,
)

scae = AttrDict(
    n_classes=n_classes,
    dynamic_l2_weight=10,
    caps_ll_weight=1.,
    vote_type='enc',
    presence_type='enc',
    stop_grad_caps_input=True,
    stop_grad_caps_target=True,
    prior_sparsity_loss_type='l2',
    prior_within_example_sparsity_weight=2.0,
    prior_between_example_sparsity_weight=0.35,
    posterior_sparsity_loss_type='entropy',
    posterior_within_example_sparsity_weight=0.7,
    posterior_between_example_sparsity_weight=0.2,
)
