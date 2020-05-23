from argparse import Namespace

from torch_scae.object_decoder import CapsuleLayer, CapsuleObjectDecoder
from torch_scae.part_decoder import TemplateGenerator, TemplateBasedImageDecoder
from torch_scae.part_encoder import CNNEncoder, CapsuleImageEncoder
from torch_scae.set_transformer import SetTransformer
from torch_scae.stacked_capsule_auto_encoder import SCAE


def make_scae(config):
    if isinstance(config, dict):
        config = Namespace(**config)

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
