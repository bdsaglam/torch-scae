import unittest

import torch

from torch_scae.part_decoder import TemplateGenerator, TemplateBasedImageDecoder


class TemplateGeneratorTestCase(unittest.TestCase):
    def helper(self,
               image_shape=(1, 28, 28),
               n_templates=40,
               template_size=(11, 11),
               colorize_templates=True):

        if colorize_templates:
            dim_feature = 16
        else:
            dim_feature = None

        n_channels = image_shape[0]
        template_nonlin = 'sigmoid'
        color_nonlin = 'sigmoid'

        template_generator = TemplateGenerator(
            n_templates=n_templates,
            n_channels=n_channels,
            template_size=template_size,
            template_nonlin=template_nonlin,
            dim_feature=dim_feature,
            colorize_templates=colorize_templates,
            color_nonlin=color_nonlin,
        )

        batch_size = 4
        with torch.no_grad():
            if colorize_templates:
                feature = torch.rand(batch_size, n_templates, dim_feature)
                res = template_generator(feature)
            else:
                res = template_generator(batch_size=batch_size)

        self.assertTrue(
            res.raw_templates.shape == (
                1, n_templates, n_channels, *template_size)
        )
        self.assertTrue(
            res.templates.shape == (
                batch_size, n_templates, n_channels, *template_size)
        )

    def test_shape_single_channel_without_color(self):
        self.helper(image_shape=(1, 28, 28),
                    colorize_templates=False)

    def test_shape_single_channel_with_color(self):
        self.helper(image_shape=(1, 28, 28),
                    colorize_templates=True)

    def test_shape_multi_channel_without_color(self):
        self.helper(image_shape=(3, 28, 28),
                    colorize_templates=False)

    def test_shape_multi_channel_with_color(self):
        self.helper(image_shape=(3, 28, 28),
                    colorize_templates=True)

    def test_shape_with_different_template_size(self):
        image_shape = (1, 28, 28)
        self.helper(image_shape=image_shape, template_size=(11, 11))
        self.helper(image_shape=image_shape, template_size=(22, 22))
        self.helper(image_shape=image_shape, template_size=(32, 32))


class TemplateBasedImageDecoderTestCase(unittest.TestCase):
    def helper(self,
               image_shape=(1, 28, 28),
               n_templates=40,
               template_size=(11, 11),
               learn_output_scale=False,
               use_alpha_channel=True,
               background_value=True,
               presence=True,
               background_image=True,
               ):
        n_channels = image_shape[0]
        output_size = image_shape[1:]

        template_decoder = TemplateBasedImageDecoder(
            n_templates=n_templates,
            template_size=template_size,
            output_size=output_size,
            learn_output_scale=learn_output_scale,
            use_alpha_channel=use_alpha_channel,
            background_value=background_value,
        )

        batch_size = 4

        templates = torch.rand(
            batch_size, n_templates, n_channels, *template_size)

        pose = torch.rand(batch_size, n_templates, 6)

        if presence:
            presence = torch.rand(batch_size, n_templates)
        else:
            presence = None

        if background_image:
            bg_image = torch.rand(batch_size, n_channels, *output_size)
        else:
            bg_image = None

        with torch.no_grad():
            decoding_result = template_decoder(templates=templates,
                                               pose=pose,
                                               presence=presence,
                                               bg_image=bg_image)

        transformed_templates = decoding_result.transformed_templates
        mixing_logits = decoding_result.mixing_logits

        self.assertTrue(
            transformed_templates.shape == (
                batch_size, n_templates, n_channels, *output_size)
        )
        self.assertTrue(
            mixing_logits.shape == (
                batch_size, n_templates, 1, *output_size)
        )

    def test_shape_with_color(self):
        self.helper(image_shape=(3, 28, 28))

    def test_shape_without_color(self):
        self.helper(image_shape=(1, 28, 28))

    def test_shape_with_scale(self):
        self.helper(learn_output_scale=True)

    def test_shape_without_scale(self):
        self.helper(learn_output_scale=False)

    def test_shape_with_alpha(self):
        self.helper(use_alpha_channel=True)

    def test_shape_without_alpha(self):
        self.helper(use_alpha_channel=False)

    def test_shape_with_presence(self):
        self.helper(presence=True)

    def test_shape_without_presence(self):
        self.helper(presence=False)

    def test_shape_with_bg_value(self):
        self.helper(background_value=True)

    def test_shape_without_bg_value(self):
        self.helper(background_value=False)

    def test_shape_with_bg_image(self):
        self.helper(background_image=True)

    def test_shape_without_bg_image(self):
        self.helper(background_image=False)


if __name__ == '__main__':
    unittest.main()
