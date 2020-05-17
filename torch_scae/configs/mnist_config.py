import torch
from monty.collections import AttrDict

image_shape = (1, 28, 28)

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
