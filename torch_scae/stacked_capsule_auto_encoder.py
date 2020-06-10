# Copyright 2020 Barış Deniz Sağlam.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
            recon_mse_weight=0,
            part_caps_sparsity_weight=0.,
            cpr_dynamic_reg_weight=0.,
            caps_ll_weight=0.,
            prior_sparsity_loss_type='l2',
            prior_within_example_sparsity_weight=0.,
            prior_between_example_sparsity_weight=0.,
            prior_within_example_constant=None,
            posterior_sparsity_loss_type='entropy',
            posterior_within_example_sparsity_weight=0.,
            posterior_between_example_sparsity_weight=0.,
            reconstruct_alternatives=True,
    ):
        super().__init__()

        self.part_encoder = part_encoder
        self.template_generator = template_generator
        self.part_decoder = part_decoder
        self.obj_encoder = obj_encoder
        self.obj_decoder = obj_decoder

        self.n_classes = n_classes

        self.vote_type = vote_type
        self.presence_type = presence_type

        self.stop_grad_caps_input = stop_grad_caps_input
        self.stop_grad_caps_target = stop_grad_caps_target

        if n_classes:
            self.prior_classifier = nn.Sequential(
                nn.Linear(obj_decoder.n_obj_capsules, n_classes),
                nn.Softmax(-1),
            )
            self.posterior_classifier = nn.Sequential(
                nn.Linear(obj_decoder.n_obj_capsules, n_classes),
                nn.Softmax(-1),
            )
        else:
            self.prior_classifier = None
            self.posterior_classifier = None

        self.cpr_dynamic_reg_weight = cpr_dynamic_reg_weight
        self.caps_ll_weight = caps_ll_weight
        self.recon_mse_weight = recon_mse_weight
        self.prior_sparsity_loss_type = prior_sparsity_loss_type
        self.prior_within_example_sparsity_weight = prior_within_example_sparsity_weight
        self.prior_between_example_sparsity_weight = prior_between_example_sparsity_weight
        self.prior_within_example_constant = prior_within_example_constant
        self.posterior_sparsity_loss_type = posterior_sparsity_loss_type
        self.posterior_within_example_sparsity_weight = posterior_within_example_sparsity_weight
        self.posterior_between_example_sparsity_weight = posterior_between_example_sparsity_weight
        self.part_caps_sparsity_weight = part_caps_sparsity_weight
        self.reconstruct_alternatives = reconstruct_alternatives

    def forward(self, image):
        batch_size = image.shape[0]

        # Encode parts from the image
        part_enc_res = self.part_encoder(image)

        # Generate templates
        template_res = self.template_generator(feature=part_enc_res.feature,
                                               batch_size=batch_size)
        templates = template_res.templates
        del template_res.raw_templates, template_res

        # Encode objects from templates and part instantiation parameters
        input_part_param = torch.cat(
            [part_enc_res.pose, 1. - part_enc_res.presence.unsqueeze(-1)],
            -1
        )
        input_presence = part_enc_res.presence

        if self.stop_grad_caps_input:
            input_part_param = input_part_param.detach()
            input_presence = input_presence.detach()

        # Skip connection from the image to the higher level capsule
        if part_enc_res.feature is not None:
            input_part_param = torch.cat([input_part_param, part_enc_res.feature], -1)

        input_templates = templates
        if self.stop_grad_caps_input:
            input_templates = templates.detach()
        input_templates = input_templates.view(*input_templates.shape[:2], -1)

        parts_with_templates = torch.cat([input_part_param, input_templates], -1)

        obj_encoding = self.obj_encoder(parts_with_templates, input_presence)
        del input_part_param, input_templates, parts_with_templates, input_presence

        # Decode parts poses and presences from object encoding
        target_pose, target_presence = part_enc_res.pose, part_enc_res.presence
        if self.stop_grad_caps_target:
            target_pose = target_pose.detach()
            target_presence = target_presence.detach()

        res = self.obj_decoder(obj_encoding, target_pose, target_presence)
        del obj_encoding, target_pose, target_presence

        res.part_presence = part_enc_res.presence

        # Decode parts into reconstructions. START
        if self.vote_type == 'enc':
            part_dec_vote = part_enc_res.pose
        elif self.vote_type == 'soft':
            part_dec_vote = res.soft_winner
        elif self.vote_type == 'hard':
            part_dec_vote = res.winner
        else:
            raise ValueError(f'Invalid vote_type: {self.vote_type}')

        if self.presence_type == 'enc':
            part_dec_presence = part_enc_res.presence
        elif self.presence_type == 'soft':
            part_dec_presence = res.soft_winner_presence
        elif self.presence_type == 'hard':
            part_dec_presence = res.winner_presence
        else:
            raise ValueError(f'Invalid presence_type: {self.presence_type}')

        res.rec = self.part_decoder(
            templates=templates,
            pose=part_dec_vote,
            presence=part_dec_presence)

        if self.reconstruct_alternatives:
            with torch.no_grad():
                # bottom-up caps image reconstruction
                res.bottom_up_rec = self.part_decoder(
                    templates=templates,
                    pose=part_enc_res.pose,
                    presence=part_enc_res.presence)

                # top-down winner caps image reconstruction
                res.top_down_rec = self.part_decoder(
                    templates=templates,
                    pose=res.winner,
                    presence=part_enc_res.presence)

                # top-down per caps image reconstruction
                n_obj_caps = res.vote.shape[1]

                # tile tensors per object capsule
                # (B, M, C, H, W) -> (B*O, M, C, H, W)
                td_templates = templates.repeat_interleave(repeats=n_obj_caps, dim=0)
                # (B, O, M, 6) -> (B*O, M, 6)
                td_pose = res.vote.view(-1, *res.vote.shape[2:])
                # (B, M) -> (B*O, M)
                td_enc_presence = part_enc_res.presence.repeat_interleave(repeats=n_obj_caps, dim=0)
                # (B, O, M) -> (B*O, M)
                td_dec_presence = res.vote_presence_binary.view(-1, *res.vote_presence.shape[2:])
                td_presence = td_enc_presence * td_dec_presence
                res.top_down_per_caps_rec = self.part_decoder(
                    templates=td_templates,
                    pose=td_pose,
                    presence=td_presence)
                del td_templates, td_pose, td_enc_presence, td_dec_presence, td_presence

        # Decode parts into reconstructions. END

        res.templates = templates
        res.template_presence = part_enc_res.presence
        res.transformed_templates = res.rec.transformed_templates

        if self.n_classes is not None:
            assert self.prior_classifier is not None
            assert self.posterior_classifier is not None

            res.prior_cls_prob = self.prior_classifier(
                res.caps_presence.detach())

            mass_explained_by_capsule = res.posterior_mixing_prob.sum(-1)
            res.posterior_cls_prob = self.prior_classifier(
                mass_explained_by_capsule.detach())
            del mass_explained_by_capsule

        return res

    def loss(self, res, reconstruction_target, label=None):
        log = dict()
        # image reconstruction likelihood
        rec_ll_per_pixel = res.rec.pdf.log_prob(reconstruction_target)
        rec_ll = rec_ll_per_pixel.view(rec_ll_per_pixel.shape[0], -1).sum(-1).mean()
        loss = -rec_ll
        log.update(rec_ll_loss=-rec_ll)

        # image reconstruction mse loss
        if self.recon_mse_weight > 0:
            mse_per_pixel = (reconstruction_target - res.rec.pdf.mode()) ** 2
            mse = mse_per_pixel.view(mse_per_pixel.shape[0], -1).sum(-1).mean()
            loss += self.recon_mse_weight * mse
            log.update(mse=mse)

        # part capsule sparsity loss
        if self.part_caps_sparsity_weight > 0:
            part_caps_l1 = res.part_presence.sum(-1).mean()
            loss += self.part_caps_sparsity_weight * part_caps_l1
            log.update(part_caps_loss=part_caps_l1)

        # capsule likelihood
        loss += -self.caps_ll_weight * res.log_prob
        log.update(log_prob_loss=-res.log_prob)

        # prior sparsity loss
        if self.prior_within_example_sparsity_weight > 0 \
                or self.prior_between_example_sparsity_weight > 0:
            (prior_within_sparsity_loss,
             prior_between_sparsity_loss) = sparsity_loss(
                self.prior_sparsity_loss_type,
                res.caps_presence,
                n_classes=self.n_classes,
                within_example_constant=self.prior_within_example_constant)

            loss += (self.prior_within_example_sparsity_weight * prior_within_sparsity_loss
                     + self.prior_between_example_sparsity_weight * prior_between_sparsity_loss)
            log.update(prior_within_sparsity_loss=prior_within_sparsity_loss,
                       prior_between_sparsity_loss=prior_between_sparsity_loss)

        # posterior sparsity loss
        if self.prior_within_example_sparsity_weight > 0 \
                or self.prior_between_example_sparsity_weight > 0:
            n_points = res.posterior_mixing_prob.shape[-1]
            mass_explained_by_capsule = res.posterior_mixing_prob.sum(-1)
            (posterior_within_sparsity_loss,
             posterior_between_sparsity_loss) = sparsity_loss(
                self.posterior_sparsity_loss_type,
                mass_explained_by_capsule / n_points,
                n_classes=self.n_classes)

            loss += (self.posterior_within_example_sparsity_weight * posterior_within_sparsity_loss
                     + self.posterior_between_example_sparsity_weight * posterior_between_sparsity_loss)
            log.update(posterior_within_sparsity_loss=posterior_within_sparsity_loss,
                       posterior_between_sparsity_loss=posterior_between_sparsity_loss)

        # object capsule regularization losses
        loss += self.cpr_dynamic_reg_weight * res.cpr_dynamic_reg_loss
        log.update(cpr_dynamic_reg_loss=res.cpr_dynamic_reg_loss)

        # classification losses
        if label is not None:
            assert self.n_classes is not None

            prior_cls_xe = F.cross_entropy(res.prior_cls_prob, target=label)
            posterior_cls_xe = F.cross_entropy(res.posterior_cls_prob, target=label)

            loss += prior_cls_xe + posterior_cls_xe
            log.update(prior_cls_xe=prior_cls_xe, posterior_cls_xe=posterior_cls_xe)

        return loss, log

    def calculate_accuracy(self, res, label: torch.Tensor):
        prior_pred = res.prior_cls_prob.argmax(-1)
        prior_cls_acc = (prior_pred == label).float().mean()

        posterior_pred = res.posterior_cls_prob.argmax(-1)
        posterior_cls_acc = (posterior_pred == label).float().mean()

        best_cls_acc = torch.max(prior_cls_acc, posterior_cls_acc)
        return best_cls_acc
