import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scae.object_decoder import sparsity_loss


class SCAE(nn.Module):
    """Stacked Capsule Auto-Encoder"""

    def __init__(
            self,
            part_encoder,
            template_generator,
            part_decoder,
            obj_encoder,
            obj_decoder,
            n_classes=None,
            vote_type='soft',
            presence_type='enc',
            stop_grad_caps_input=True,
            stop_grad_caps_target=True,
            dynamic_l2_weight=0.,
            caps_ll_weight=0.,
            prior_sparsity_loss_type='l2',
            prior_within_example_sparsity_weight=0.,
            prior_between_example_sparsity_weight=0.,
            prior_within_example_constant=0.,
            posterior_sparsity_loss_type='entropy',
            posterior_within_example_sparsity_weight=0.,
            posterior_between_example_sparsity_weight=0.,
            part_caps_sparsity_weight=0.,
    ):

        super().__init__()
        self._part_encoder = part_encoder
        self._template_generator = template_generator
        self._part_decoder = part_decoder
        self._obj_encoder = obj_encoder
        self._obj_decoder = obj_decoder

        self._n_classes = n_classes

        self._vote_type = vote_type
        self._presence_type = presence_type

        self._stop_grad_caps_input = stop_grad_caps_input
        self._stop_grad_caps_target = stop_grad_caps_target

        if n_classes:
            self._classifier = nn.Sequential(
                nn.Linear(obj_decoder.n_obj_capsules, n_classes),
                nn.Softmax(-1),
            )
        else:
            self._classifier = None

        self._dynamic_l2_weight = dynamic_l2_weight
        self._caps_ll_weight = caps_ll_weight
        self._prior_sparsity_loss_type = prior_sparsity_loss_type
        self._prior_within_example_sparsity_weight = prior_within_example_sparsity_weight
        self._prior_between_example_sparsity_weight = prior_between_example_sparsity_weight
        self._prior_within_example_constant = prior_within_example_constant
        self._posterior_sparsity_loss_type = posterior_sparsity_loss_type
        self._posterior_within_example_sparsity_weight = posterior_within_example_sparsity_weight
        self._posterior_between_example_sparsity_weight = posterior_between_example_sparsity_weight
        self._part_caps_sparsity_weight = part_caps_sparsity_weight

    def forward(self, image, label=None, reconstruction_target=None):
        device = next(iter(self.parameters())).device

        if reconstruction_target is None:
            reconstruction_target = image

        batch_size = image.shape[0]
        part_encoding = self._part_encoder(image)

        input_pose = torch.cat(
            [part_encoding.pose, 1. - part_encoding.presence.unsqueeze(-1)],
            -1
        )
        input_presence = part_encoding.presence

        if self._stop_grad_caps_input:
            input_pose = input_pose.detach()
            input_presence = input_presence.detach()

        target_pose, target_presence = part_encoding.pose, part_encoding.presence
        if self._stop_grad_caps_target:
            target_pose = target_pose.detach()
            target_presence = target_presence.detach()

        # skip connection from the img to the higher level capsule
        if part_encoding.feature is not None:
            input_pose = torch.cat([input_pose, part_encoding.feature], -1)

        template_res = self._template_generator(feature=part_encoding.feature)
        templates = template_res.templates

        input_templates = templates
        if self._stop_grad_caps_input:
            input_templates = templates.detach()

        if input_templates.shape[0] == 1:
            input_templates = input_templates.repeat(batch_size, 1, 1, 1, 1)

        input_templates = input_templates.view(*input_templates.shape[:2], -1)
        pose_with_templates = torch.cat([input_pose, input_templates], -1)

        h = self._obj_encoder(pose_with_templates, input_presence)
        del input_pose
        del input_presence
        del input_templates
        del pose_with_templates

        res = self._obj_decoder(h, target_pose, target_presence)
        del h
        del target_pose
        del target_presence

        res.part_presence = part_encoding.presence

        if self._vote_type == 'enc':
            part_dec_vote = part_encoding.pose
        elif self._vote_type == 'soft':
            part_dec_vote = res.soft_winner
        elif self._vote_type == 'hard':
            part_dec_vote = res.winner
        else:
            raise ValueError('Invalid vote_type="{}"".'.format(self._vote_type))

        if self._presence_type == 'enc':
            part_dec_presence = part_encoding.presence
        elif self._presence_type == 'soft':
            part_dec_presence = res.soft_winner_presence
        elif self._presence_type == 'hard':
            part_dec_presence = res.winner_presence
        else:
            raise ValueError(f'Invalid pres_type: {self._presence_type}')

        res.bottom_up_decoding = self._part_decoder(
            templates=templates,
            pose=part_encoding.pose,
            presence=part_encoding.presence)

        res.top_down_decoding = self._part_decoder(
            templates=templates,
            pose=res.winner,
            presence=part_encoding.presence)

        part_decoding = self._part_decoder(
            templates=templates,
            pose=part_dec_vote,
            presence=part_dec_presence)

        #
        n_obj_caps = res.vote.shape[1]
        tiled_presence = part_encoding.presence.repeat(n_obj_caps, 1)

        tiled_feature = part_encoding.feature
        if tiled_feature is not None:
            tiled_feature = tiled_feature.repeat(n_obj_caps, 1, 1)

        tiled_templates = self._template_generator(feature=tiled_feature).templates
        res.top_down_per_caps_rec = self._part_decoder(
            templates=tiled_templates,
            pose=res.vote.view(-1, *res.vote.shape[2:]),
            presence=res.vote_presence.view(
                -1, *res.vote_presence.shape[2:]) * tiled_presence)
        del tiled_presence
        del tiled_feature
        del tiled_templates

        #
        res.templates = templates
        res.template_presence = part_encoding.presence
        res.used_templates = part_decoding.transformed_templates

        res.rec_mode = part_decoding.pdf.mode()
        res.rec_mean = part_decoding.pdf.mean

        res.mse_per_pixel = (reconstruction_target - res.rec_mode) ** 2
        res.mse = res.mse_per_pixel.view(
            res.mse_per_pixel.shape[0], -1).sum(-1).mean()

        res.rec_ll_per_pixel = part_decoding.pdf.log_prob(reconstruction_target)
        res.rec_ll = res.rec_ll_per_pixel.view(
            res.rec_ll_per_pixel.shape[0], -1).sum(-1).mean()

        if label is not None:
            assert self._classifier is not None

            mass_explained_by_capsule = res.posterior_mixing_prob.sum(1)
            res.posterior_cls_xe, res.posterior_cls_acc = self._classify(
                mass_explained_by_capsule.detach(), label)
            del mass_explained_by_capsule

            res.prior_cls_xe, res.prior_cls_acc = self._classify(
                res.caps_presence_prob.detach(), label)

            res.best_cls_acc = torch.max(res.prior_cls_acc,
                                         res.posterior_cls_acc)

        res.part_caps_l1 = res.part_presence.sum(-1).mean()

        return res

    def _classify(self, x, label):  # (B, O), (B, )
        predicted_prob = self._classifier(x)
        xe_loss = F.cross_entropy(input=predicted_prob, target=label)
        accuracy = (torch.argmax(predicted_prob, 1) == label).float().mean()
        return xe_loss, accuracy

    def loss(self, res):
        (res.prior_within_sparsity_loss,
         res.prior_between_sparsity_loss) = sparsity_loss(
            self._prior_sparsity_loss_type,
            res.caps_presence_prob,
            num_classes=self._n_classes,
            within_example_constant=self._prior_within_example_constant)

        mass_explained_by_capsule = res.posterior_mixing_prob.sum(1)
        n_points = res.posterior_mixing_prob.shape[1]
        (res.posterior_within_sparsity_loss,
         res.posterior_between_sparsity_loss) = sparsity_loss(
            self._posterior_sparsity_loss_type,
            mass_explained_by_capsule / n_points,
            num_classes=self._n_classes)

        loss = (
                -res.rec_ll
                - self._caps_ll_weight * res.log_prob
                + self._dynamic_l2_weight * res.dynamic_weights_l2
                + self._part_caps_sparsity_weight * res.part_caps_l1
                + self._posterior_within_example_sparsity_weight * res.posterior_within_sparsity_loss
                + self._posterior_between_example_sparsity_weight * res.posterior_between_sparsity_loss
                + self._prior_within_example_sparsity_weight * res.prior_within_sparsity_loss
                + self._prior_between_example_sparsity_weight * res.prior_between_sparsity_loss
        )

        try:
            loss += res.posterior_cls_xe + res.prior_cls_xe
        except AttributeError:
            pass

        return loss
