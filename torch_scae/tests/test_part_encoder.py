import unittest

import torch

from torch_scae.part_encoder import CNNEncoder, CapsuleImageEncoder


class CapsuleImageEncoderTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.image_shape = (3, 32, 32)

    def make_cnn_encoder(self):
        return CNNEncoder(
            input_shape=self.image_shape,
            out_channels=[128] * 4,
            kernel_sizes=[3, 3, 3, 3],
            strides=[2, 2, 1, 1],
            activate_final=True
        )

    def test_cnn_encoder(self):
        cnn_encoder = self.make_cnn_encoder()

        batch_size = 4
        image = torch.rand(batch_size, *self.image_shape)

        with torch.no_grad():
            out = cnn_encoder(image)

        self.assertTrue(
            list(out.shape) == [batch_size] + list(cnn_encoder.output_shape)
        )

    def test_pcae_primary_capsule(self):
        cnn_encoder = self.make_cnn_encoder()

        n_caps = 40
        n_poses = 6
        n_special_features = 16
        capsule_image_encoder = CapsuleImageEncoder(
            encoder=cnn_encoder,
            input_shape=self.image_shape,
            n_caps=n_caps,
            n_poses=n_poses,
            n_special_features=n_special_features,
            similarity_transform=False,
        )

        batch_size = 4
        image = torch.rand(batch_size, *self.image_shape)

        with torch.no_grad():
            result = capsule_image_encoder(image)

        self.assertTrue(
            result.pose.shape == (batch_size, n_caps, n_poses)
        )
        self.assertTrue(
            result.feature.shape == (batch_size, n_caps, n_special_features)
        )
        self.assertTrue(
            result.presence.shape == (batch_size, n_caps)
        )


if __name__ == '__main__':
    unittest.main()
