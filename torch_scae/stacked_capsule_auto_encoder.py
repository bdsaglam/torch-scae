import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scae.object_decoder import sparsity_loss
from torch_scae.probes import ClassificationProbe


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
            cpr_dynamic_reg_weight=0.,
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
            self._posterior_cls_probe = ClassificationProbe(
                obj_decoder.n_obj_capsules, n_classes)
            self._prior_cls_probe = ClassificationProbe(
                obj_decoder.n_obj_capsules, n_classes)
        else:
            self._posterior_cls_probe = None
            self._prior_cls_probe = None

        self._cpr_dynamic_reg_weight = cpr_dynamic_reg_weight
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

        # Encode parts from the image
        part_encoding = self._part_encoder(image)

        # Generate templates
        template_res = self._template_generator(feature=part_encoding.feature,
                                                batch_size=batch_size)
        templates = template_res.templates

        # Encode objects from templates and part instantiation parameters
        input_part_param = torch.cat(
            [part_encoding.pose, 1. - part_encoding.presence.unsqueeze(-1)],
            -1
        )
        input_presence = part_encoding.presence

        if self._stop_grad_caps_input:
            input_part_param = input_part_param.detach()
            input_presence = input_presence.detach()

        # Skip connection from the image to the higher level capsule
        if part_encoding.feature is not None:
            input_part_param = torch.cat([input_part_param, part_encoding.feature], -1)

        input_templates = templates
        if self._stop_grad_caps_input:
            input_templates = templates.detach()
        input_templates = input_templates.view(*input_templates.shape[:2], -1)

        parts_with_templates = torch.cat([input_part_param, input_templates], -1)

        obj_encoding = self._obj_encoder(parts_with_templates, input_presence)
        del input_part_param, input_templates, parts_with_templates, input_presence

        # Decode parts poses and presences from object encoding
        target_pose, target_presence = part_encoding.pose, part_encoding.presence
        if self._stop_grad_caps_target:
            target_pose = target_pose.detach()
            target_presence = target_presence.detach()

        res = self._obj_decoder(obj_encoding, target_pose, target_presence)
        del obj_encoding, target_pose, target_presence

        res.part_presence = part_encoding.presence

        # Decode parts into reconstructions. START
        if self._vote_type == 'enc':
            part_dec_vote = part_encoding.pose
        elif self._vote_type == 'soft':
            part_dec_vote = res.soft_winner
        elif self._vote_type == 'hard':
            part_dec_vote = res.winner
        else:
            raise ValueError(f'Invalid vote_type: {self._vote_type}')

        if self._presence_type == 'enc':
            part_dec_presence = part_encoding.presence
        elif self._presence_type == 'soft':
            part_dec_presence = res.soft_winner_presence
        elif self._presence_type == 'hard':
            part_dec_presence = res.winner_presence
        else:
            raise ValueError(f'Invalid presence_type: {self._presence_type}')

        res.bottom_up_rec = self._part_decoder(
            templates=templates,
            pose=part_encoding.pose,
            presence=part_encoding.presence)

        res.top_down_rec = self._part_decoder(
            templates=templates,
            pose=res.winner,
            presence=part_encoding.presence)

        rec = self._part_decoder(
            templates=templates,
            pose=part_dec_vote,
            presence=part_dec_presence)

        #
        n_obj_caps = res.vote.shape[1]

        td_feature = part_encoding.feature
        if td_feature is not None:
            td_feature = td_feature.repeat(n_obj_caps, 1, 1)

        td_templates = self._template_generator(feature=td_feature).templates
        td_pose = res.vote.view(-1, *res.vote.shape[2:])
        td_presence = res.vote_presence.view(-1, *res.vote_presence.shape[2:]) \
                      * part_encoding.presence.repeat(n_obj_caps, 1)
        res.top_down_per_caps_rec = self._part_decoder(
            templates=td_templates,
            pose=td_pose,
            presence=td_presence)
        del td_feature, td_templates, td_pose, td_presence

        # Decode parts into reconstructions. END

        res.templates = templates
        res.template_presence = part_encoding.presence
        res.used_templates = rec.transformed_templates

        res.rec_mode = rec.pdf.mode()
        res.rec_mean = rec.pdf.mean

        res.mse_per_pixel = (reconstruction_target - res.rec_mode) ** 2
        res.mse = res.mse_per_pixel.view(
            res.mse_per_pixel.shape[0], -1).sum(-1).mean()

        res.rec_ll_per_pixel = rec.pdf.log_prob(reconstruction_target)
        res.rec_ll = res.rec_ll_per_pixel.view(
            res.rec_ll_per_pixel.shape[0], -1).sum(-1).mean()

        if label is not None:
            assert self._prior_cls_probe is not None
            assert self._posterior_cls_probe is not None

            res.prior_cls_xe, res.prior_cls_acc = self._prior_cls_probe(
                res.caps_presence_prob.detach(), label)

            mass_explained_by_capsule = res.posterior_mixing_prob.sum(1)
            res.posterior_cls_xe, res.posterior_cls_acc = self._posterior_cls_probe(
                mass_explained_by_capsule.detach(), label)
            del mass_explained_by_capsule

            res.best_cls_acc = torch.max(res.prior_cls_acc,
                                         res.posterior_cls_acc)

        res.part_caps_l1 = res.part_presence.sum(-1).mean()

        return res

    def _classify(self, model, x, label):  # (B, O), (B, )
        predicted_prob = model(x)
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
                + self._cpr_dynamic_reg_weight * res.cpr_dynamic_reg_loss
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
