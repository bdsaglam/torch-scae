import unittest

import torch

from torch_scae.configs import mnist_config
from torch_scae.part_decoder import TemplateBasedImageDecoder
from torch_scae.part_encoder import CNNEncoder, CapsuleImageEncoder


class TemplateBasedImageDecoderTestCase(unittest.TestCase):
    def test_init(self):
        config = mnist_config
        cnn_encoder = CNNEncoder(
            **config.pcae_cnn_encoder
        )
        capsule_image_encoder = CapsuleImageEncoder(
            encoder=cnn_encoder,
            **config.pcae_primary_capsule
        )

        template_decoder = TemplateBasedImageDecoder(
            **config.pcae_template_decoder)

        n_templates = config.pcae_template_decoder.n_templates
        template_size = config.pcae_template_decoder.template_size
        n_channels = config.pcae_template_decoder.n_channels
        input_shape = config.pcae_primary_capsule.input_shape
        output_size = config.pcae_template_decoder.output_size
        self.assertTrue(input_shape[1:] == output_size)

        with torch.no_grad():
            batch_size = 4
            image = torch.rand(batch_size, *config.image_shape)
            encoding_result = capsule_image_encoder(image)
            decoding_result = template_decoder(
                pose=encoding_result.pose,
                presence=encoding_result.presence,
                template_feature=encoding_result.feature,
            )
            raw_templates = decoding_result.raw_templates
            transformed_templates = decoding_result.transformed_templates
            mixing_logits = decoding_result.mixing_logits
            self.assertTrue(
                raw_templates.shape == (1, n_templates, n_channels, *template_size)
            )
            self.assertTrue(
                transformed_templates.shape == (
                    batch_size, n_templates, n_channels, *output_size)
            )
            self.assertTrue(
                mixing_logits.shape == (batch_size, n_templates, n_channels, *output_size)
            )


if __name__ == '__main__':
    unittest.main()
