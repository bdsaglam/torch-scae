import torch
from monty.collections import AttrDict

from torch_scae.general_utils import prod

__all__ = [
    'image_shape',
    'n_classes',
    'n_part_caps',
    'n_obj_caps',
    'pcae_cnn_encoder',
    'pcae_encoder',
    'pcae_template_generator',
    'pcae_decoder',
    'ocae_encoder_set_transformer',
    'ocae_decoder_capsule',
    'scae',
]

image_shape = (1, 28, 28)
n_classes = 10
n_part_caps = 40
n_obj_caps = 32

pcae_cnn_encoder = AttrDict(
    input_shape=image_shape,
    out_channels=[128] * 4,
    kernel_sizes=[3, 3, 3, 3],
    strides=[2, 2, 1, 1],
    activate_final=True
)

pcae_encoder = AttrDict(
    input_shape=image_shape,
    n_caps=n_part_caps,
    n_poses=6,
    n_special_features=16,
    similarity_transform=False,
)

pcae_template_generator = AttrDict(
    n_templates=pcae_encoder.n_caps,
    n_channels=image_shape[0],
    template_size=(11, 11),
    template_nonlin=torch.sigmoid,
    dim_feature=pcae_encoder.n_special_features,
    colorize_templates=True,
    color_nonlin=torch.sigmoid,
)

pcae_decoder = AttrDict(
    n_templates=pcae_template_generator.n_templates,
    template_size=pcae_template_generator.template_size,
    output_size=image_shape[1:],
    learn_output_scale=False,
    use_alpha_channel=True,
    background_value=True,
)

_ocae_st_dim_in = (
        pcae_encoder.n_poses + pcae_template_generator.dim_feature + 1 \
        + pcae_template_generator.n_channels * prod(pcae_template_generator.template_size)
)
ocae_encoder_set_transformer = AttrDict(
    n_layers=3,
    n_heads=1,
    dim_in=_ocae_st_dim_in,
    dim_hidden=16,
    dim_out=256,
    n_outputs=n_obj_caps,
    layer_norm=True,
)

ocae_decoder_capsule = AttrDict(
    n_caps=ocae_encoder_set_transformer.n_outputs,
    dim_feature=ocae_encoder_set_transformer.dim_out,
    n_votes=pcae_decoder.n_templates,
    dim_caps=32,
    hidden_sizes=(128,),
    caps_dropout_rate=0.0,
    learn_vote_scale=True,
    allow_deformations=True,
    noise_type='uniform',
    noise_scale=4.,
    similarity_transform=False,
)

scae = AttrDict(
    n_classes=n_classes,
    cpr_dynamic_reg_weight=10,
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
