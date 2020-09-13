from argparse import Namespace

from torch_scae.object_decoder import CapsuleLayer, CapsuleObjectDecoder
from torch_scae.part_decoder import TemplateGenerator, TemplateBasedImageDecoder
from torch_scae.part_encoder import CNNEncoder, CapsuleImageEncoder
from torch_scae.set_transformer import SetTransformer
from torch_scae.stacked_capsule_auto_encoder import SCAE


def prepare_model_params(
        image_shape,
        n_classes,
        n_part_caps,
        n_obj_caps,
        pcae_cnn_encoder_params=None,
        pcae_encoder_params=None,
        pcae_template_generator_params=None,
        pcae_decoder_params=None,
        ocae_encoder_set_transformer_params=None,
        ocae_decoder_capsule_params=None,
        scae_params=None,

):
    pcae_cnn_encoder_params = pcae_cnn_encoder_params or dict()
    pcae_encoder_params = pcae_encoder_params or dict()
    pcae_template_generator_params = pcae_template_generator_params or dict()
    pcae_decoder_params = pcae_decoder_params or dict()
    ocae_encoder_set_transformer_params = ocae_encoder_set_transformer_params or dict()
    ocae_decoder_capsule_params = ocae_decoder_capsule_params or dict()
    scae_params = scae_params or dict()

    assert 'input_shape' not in pcae_cnn_encoder_params
    pcae_cnn_encoder = dict(
        input_shape=image_shape,
        out_channels=[128] * 4,
        kernel_sizes=[3, 3, 3, 3],
        strides=[2, 2, 1, 1],
        activate_final=True
    )
    pcae_cnn_encoder.update(pcae_cnn_encoder_params)

    assert 'input_shape' not in pcae_encoder_params
    pcae_encoder = dict(
        input_shape=image_shape,
        n_caps=n_part_caps,
        n_poses=6,
        n_special_features=16,
        similarity_transform=False,
    )
    pcae_encoder.update(pcae_encoder_params)

    assert 'n_templates' not in pcae_template_generator_params
    assert 'n_channels' not in pcae_template_generator_params
    assert 'dim_feature' not in pcae_template_generator_params
    pcae_template_generator = dict(
        n_templates=pcae_encoder['n_caps'],
        n_channels=image_shape[0],
        template_size=(11, 11),
        template_nonlin='sigmoid',
        dim_feature=pcae_encoder['n_special_features'],
        colorize_templates=True,
        color_nonlin='sigmoid',
    )
    pcae_template_generator.update(pcae_template_generator_params)

    assert 'n_templates' not in pcae_decoder_params
    assert 'template_size' not in pcae_decoder_params
    assert 'output_size' not in pcae_decoder_params
    pcae_decoder = dict(
        n_templates=pcae_template_generator['n_templates'],
        template_size=pcae_template_generator['template_size'],
        output_size=image_shape[1:],
        learn_output_scale=False,
        use_alpha_channel=True,
        background_value=True,
    )
    pcae_decoder.update(pcae_decoder_params)

    _ocae_st_dim_in = (
            pcae_encoder['n_poses']
            + pcae_template_generator['dim_feature']
            + 1
            + (pcae_template_generator['n_channels']
               * pcae_template_generator['template_size'][0]
               * pcae_template_generator['template_size'][0])
    )

    assert '_ocae_st_dim_in' not in ocae_encoder_set_transformer_params
    assert 'n_obj_caps' not in ocae_encoder_set_transformer_params
    ocae_encoder_set_transformer = dict(
        n_layers=3,
        n_heads=1,
        dim_in=_ocae_st_dim_in,
        dim_hidden=16,
        dim_out=256,
        n_outputs=n_obj_caps,
        layer_norm=True,
    )
    ocae_encoder_set_transformer.update(ocae_encoder_set_transformer_params)

    assert 'n_caps' not in ocae_decoder_capsule_params
    assert 'dim_feature' not in ocae_decoder_capsule_params
    assert 'n_votes' not in ocae_decoder_capsule_params
    ocae_decoder_capsule = dict(
        n_caps=ocae_encoder_set_transformer['n_outputs'],
        dim_feature=ocae_encoder_set_transformer['dim_out'],
        n_votes=pcae_decoder['n_templates'],
        dim_caps=32,
        hidden_sizes=(128,),
        caps_dropout_rate=0.0,
        learn_vote_scale=True,
        allow_deformations=True,
        noise_type='uniform',
        noise_scale=4.,
        similarity_transform=False,
    )
    ocae_decoder_capsule.update(ocae_decoder_capsule_params)

    assert 'n_classes' not in scae_params
    scae = dict(
        n_classes=n_classes,
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
    scae.update(scae_params)

    return dict(
        image_shape=image_shape,
        n_classes=n_classes,
        n_part_caps=n_part_caps,
        n_obj_caps=n_obj_caps,
        pcae_cnn_encoder=pcae_cnn_encoder,
        pcae_encoder=pcae_encoder,
        pcae_template_generator=pcae_template_generator,
        pcae_decoder=pcae_decoder,
        ocae_encoder_set_transformer=ocae_encoder_set_transformer,
        ocae_decoder_capsule=ocae_decoder_capsule,
        scae=scae,
    )


def make_scae(model_params: dict):
    config = Namespace(**prepare_model_params(**model_params))

    cnn_encoder = CNNEncoder(**config.pcae_cnn_encoder)
    part_encoder = CapsuleImageEncoder(
        encoder=cnn_encoder,
        **config.pcae_encoder
    )

    template_generator = TemplateGenerator(**config.pcae_template_generator)
    part_decoder = TemplateBasedImageDecoder(**config.pcae_decoder)

    obj_encoder = SetTransformer(**config.ocae_encoder_set_transformer)

    obj_decoder_capsule = CapsuleLayer(**config.ocae_decoder_capsule)
    obj_decoder = CapsuleObjectDecoder(obj_decoder_capsule)

    scae = SCAE(
        part_encoder=part_encoder,
        template_generator=template_generator,
        part_decoder=part_decoder,
        obj_encoder=obj_encoder,
        obj_decoder=obj_decoder,
        **config.scae
    )

    return scae
