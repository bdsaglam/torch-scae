import unittest

import torch
from monty.collections import AttrDict

from torch_scae.object_decoder import CapsuleLayer, CapsuleLikelihood, CapsuleObjectDecoder


class CapsuleLayerTestCase(unittest.TestCase):
    def test_capsule_layer(self):
        capsule_layer_config = AttrDict(
            n_caps=32,
            dim_feature=256,
            n_votes=40,
            n_caps_params=32,
            hidden_sizes=(128,),
            learn_vote_scale=True,
            deformations=True,
            noise_type='uniform',
            noise_scale=4.,
            similarity_transform=False,
            caps_dropout_rate=0.0
        )

        B = 24
        O = capsule_layer_config.n_caps
        F = capsule_layer_config.dim_feature
        V = capsule_layer_config.n_votes
        H = capsule_layer_config.n_caps_params

        feature = torch.rand(B, O, F)
        parent_transform = None
        parent_presence = None

        capsule_layer = CapsuleLayer(**capsule_layer_config)

        with torch.no_grad():
            result = capsule_layer(feature,
                                   parent_presence=parent_presence,
                                   parent_transform=parent_transform)

        self.assertTrue(
            result.vote.shape == (B, O, V, 3, 3)
        )
        self.assertTrue(
            result.scale.shape == (B, O, V)
        )
        self.assertTrue(
            result.vote_presence.shape == (B, O, V)
        )
        self.assertTrue(
            result.pres_logit_per_caps.shape == (B, O, 1)
        )
        self.assertTrue(
            result.pres_logit_per_vote.shape == (B, O, V)
        )
        self.assertTrue(
            result.dynamic_weights_l2.shape == tuple()
        )
        self.assertTrue(
            result.raw_caps_param.shape == (B, O, H)
        )
        self.assertTrue(
            result.raw_caps_feature.shape == feature.shape
        )


class CapsuleLikelihoodTestCase(unittest.TestCase):
    def test_capsule_likelihood(self):
        B = 24
        O = 32
        V = 40
        P = 6

        votes = torch.rand(B, O, V, P)
        scales = torch.rand(B, O, V)
        vote_presence_probs = torch.rand(B, O, V)

        with torch.no_grad():
            capsule_likelihood = CapsuleLikelihood(votes,
                                                   scales,
                                                   vote_presence_probs)
        part_pose = torch.rand(B, V, P)
        presence = torch.rand(B, V)

        result = capsule_likelihood(part_pose, presence)

        self.assertTrue(
            result.log_prob.shape == tuple()
        )
        self.assertTrue(
            result.vote_presence.shape == (B, O, V)
        )
        self.assertTrue(
            result.winner.shape == (B, V, P)
        )
        self.assertTrue(
            result.winner_presence.shape == (B, V)
        )
        self.assertTrue(
            result.soft_winner.shape == (B, V, P)
        )
        self.assertTrue(
            result.soft_winner_presence.shape == (B, V)
        )
        self.assertTrue(
            result.posterior_mixing_prob.shape == (B, V, O)
        )
        self.assertTrue(
            result.mixing_logit.shape == (B, O + 1, V)
        )
        self.assertTrue(
            result.mixing_log_prob.shape == (B, O + 1, V)
        )
        self.assertTrue(
            result.is_from_capsule.shape == (B, V)
        )


class CapsuleObjectDecoderTestCase(unittest.TestCase):
    def test_capsule_likelihood(self):
        capsule_layer_config = AttrDict(
            n_caps=32,
            dim_feature=256,
            n_votes=40,
            n_caps_params=32,
            hidden_sizes=(128,),
            learn_vote_scale=True,
            deformations=True,
            noise_type='uniform',
            noise_scale=4.,
            similarity_transform=False,
            caps_dropout_rate=0.0
        )
        capsule_layer = CapsuleLayer(**capsule_layer_config)
        capsule_obj_decoder = CapsuleObjectDecoder(capsule_layer)

        B = 24
        O = capsule_layer_config.n_caps
        D = capsule_layer_config.dim_feature
        V = capsule_layer_config.n_votes
        H = capsule_layer_config.n_caps_params
        P = 6

        h = torch.rand(B, O, D)
        x = torch.rand(B, V, P)
        presence = torch.rand(B, V)

        with torch.no_grad():
            result = capsule_obj_decoder(h, x, presence)

        self.assertTrue(
            result.vote.shape == (B, O, V, P)
        )
        self.assertTrue(
            result.scale.shape == (B, O, V)
        )
        self.assertTrue(
            result.vote_presence.shape == (B, O, V)
        )
        self.assertTrue(
            result.pres_logit_per_caps.shape == (B, O, 1)
        )
        self.assertTrue(
            result.pres_logit_per_vote.shape == (B, O, V)
        )
        self.assertTrue(
            result.dynamic_weights_l2.shape == tuple()
        )
        self.assertTrue(
            result.raw_caps_param.shape == (B, O, H)
        )
        self.assertTrue(
            result.raw_caps_feature.shape == h.shape
        )
        self.assertTrue(
            result.log_prob.shape == tuple()
        )
        self.assertTrue(
            result.vote_presence.shape == (B, O, V)
        )
        self.assertTrue(
            result.winner.shape == (B, V, P)
        )
        self.assertTrue(
            result.winner_presence.shape == (B, V)
        )
        self.assertTrue(
            result.soft_winner.shape == (B, V, P)
        )
        self.assertTrue(
            result.soft_winner_presence.shape == (B, V)
        )
        self.assertTrue(
            result.posterior_mixing_prob.shape == (B, V, O)
        )
        self.assertTrue(
            result.mixing_logit.shape == (B, O + 1, V)
        )
        self.assertTrue(
            result.mixing_log_prob.shape == (B, O + 1, V)
        )
        self.assertTrue(
            result.is_from_capsule.shape == (B, V)
        )
        self.assertTrue(
            result.caps_presence_prob.shape == (B, O)
        )


if __name__ == '__main__':
    unittest.main()
