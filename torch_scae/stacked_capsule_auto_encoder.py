import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scae.object_decoder import sparsity_loss


class SCAE(nn.Module):
    """Capsule autoencoder."""

    def make_reconstruction_target(self, image, sobel=False):
        if sobel:
            # TODO
            image = preprocess.normalized_sobel_edges(image)
        return image

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
            stop_grad_caps_input=False,
            stop_grad_caps_target=False,
            dynamic_l2_weight=0.,
            caps_ll_weight=0.,
            prior_sparsity_loss_type='kl',
            prior_within_example_sparsity_weight=0.,
            prior_between_example_sparsity_weight=0.,
            prior_within_example_constant=0.,
            posterior_sparsity_loss_type='kl',
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

    def forward(self, image, reconstruction_target=None, label=None):
        batch_size = image.shape[0]

        if reconstruction_target is None:
            reconstruction_target = image

        part_encoding = self._part_encoder(image)
        pose = part_encoding.pose
        presence = part_encoding.presence

        expanded_pres = presence.unsqueeze(-1)
        input_pose = torch.cat([pose, 1. - expanded_pres], -1)
        input_presence = presence

        if self._stop_grad_caps_input:
            input_pose = input_pose.detach()
            input_presence = input_presence.detach()

        target_pose, target_presence = pose, presence
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

        res = self._obj_decoder(h, target_pose, target_presence)
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

        res.templates = templates
        res.template_presence = presence
        res.used_templates = part_decoding.transformed_templates

        res.rec_mode = part_decoding.pdf.mode()
        res.rec_mean = part_decoding.pdf.mean()

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
        n_points = res.posterior_mixing_prob.shape[1]
        mass_explained_by_capsule = res.posterior_mixing_prob.sum(1)

        (res.posterior_within_sparsity_loss,
         res.posterior_between_sparsity_loss) = sparsity_loss(
            self._posterior_sparsity_loss_type,
            mass_explained_by_capsule / n_points,
            num_classes=self._n_classes)

        (res.prior_within_sparsity_loss,
         res.prior_between_sparsity_loss) = sparsity_loss(
            self._prior_sparsity_loss_type,
            res.caps_presence_prob,
            num_classes=self._n_classes,
            within_example_constant=self._prior_within_example_constant)

        loss = (-res.rec_ll - self._caps_ll_weight * res.log_prob +
                self._dynamic_l2_weight * res.dynamic_weights_l2 +
                self._part_caps_sparsity_weight * res.part_caps_l1 +
                self._posterior_within_example_sparsity_weight *
                res.posterior_within_sparsity_loss -
                self._posterior_between_example_sparsity_weight *
                res.posterior_between_sparsity_loss +
                self._prior_within_example_sparsity_weight *
                res.prior_within_sparsity_loss -
                self._prior_between_example_sparsity_weight *
                res.prior_between_sparsity_loss +
                self._weight_decay * res.weight_decay_loss
                )

        try:
            loss += res.posterior_cls_xe + res.prior_cls_xe
        except AttributeError:
            pass

        return loss

    # def _report(self, data, res):
    #     reports = super(ImageAutoencoder, self)._report(data, res)
    #
    #     n_caps = self._obj_decoder._n_caps  # pylint:disable=protected-access
    #
    #     is_from_capsule = res.is_from_capsule
    #     ones = tf.ones_like(is_from_capsule)
    #     capsule_one_hot = tf.one_hot((is_from_capsule + ones),
    #                                  depth=n_caps + 1)[Ellipsis, 1:]
    #
    #     num_per_group = tf.reduce_sum(capsule_one_hot, 1)
    #     num_per_group_per_batch = tf.reduce_mean(tf.to_float(num_per_group), 0)
    #
    #     reports.update({
    #         'votes_per_capsule_{}'.format(k): v
    #         for k, v in enumerate(tf.unstack(num_per_group_per_batch))
    #     })
    #
    #     label = self._label(data)
    #
    #     return reports

    # def _plot(self, data, res, name=None):
    #
    #     img = self._img(data)
    #     label = self._label(data)
    #     if label is not None:
    #         label_one_hot = tf.one_hot(label, depth=self._n_classes)
    #
    #     _render_activations = functools.partial(  # pylint:disable=invalid-name
    #         plot.render_activations,
    #         height=int(img.shape[1]),
    #         pixels_per_caps=3,
    #         cmap='viridis')
    #
    #     mass_explained_by_capsule = tf.reduce_sum(res.posterior_mixing_probs, 1)
    #     normalized_mass_expplained_by_capsule = mass_explained_by_capsule / tf.reduce_max(
    #         mass_explained_by_capsule, -1, keepdims=True)  # pylint:disable=line-too-long
    #
    #     posterior_caps_activation = _render_activations(
    #         normalized_mass_expplained_by_capsule)  # pylint:disable=line-too-long
    #     prior_caps_activation = _render_activations(res.caps_presence_prob)
    #
    #     is_from_capsule = snt.BatchApply(_render_activations)(
    #         res.posterior_mixing_probs)
    #
    #     green = res.top_down_rec
    #     rec_red = res.rec_mode
    #     rec_green = green.pdf.mode()
    #
    #     flat_per_caps_rec = res.top_down_per_caps_rec.pdf.mode()
    #     shape = res.vote.shape[:2].concatenate(flat_per_caps_rec.shape[1:])
    #     per_caps_rec = tf.reshape(flat_per_caps_rec, shape)
    #     per_caps_rec = plot.concat_images(
    #         tf.unstack(per_caps_rec, axis=1), 1, vertical=False)
    #     one_image = tf.reduce_mean(
    #         self._img(data, self._prep), axis=-1, keepdims=True)
    #     one_rec = tf.reduce_mean(rec_red, axis=-1, keepdims=True)
    #     diff = tf.concat([one_image, one_rec, tf.zeros_like(one_image)], -1)
    #
    #     used_templates = tf.reduce_mean(res.used_templates, axis=-1, keepdims=True)
    #     green_templates = tf.reduce_mean(
    #         green.transformed_templates, axis=-1, keepdims=True)
    #     templates = tf.concat(
    #         [used_templates, green_templates,
    #          tf.zeros_like(used_templates)], -1)
    #
    #     templates = tf.concat(
    #         [templates,
    #          tf.ones_like(templates[:, :, :, :1]), is_from_capsule], 3)
    #
    #     all_imgs = [
    #                    img, rec_red, rec_green, diff, prior_caps_activation,
    #                    tf.zeros_like(rec_red[:, :, :1]), posterior_caps_activation,
    #                    per_caps_rec
    #                ] + list(tf.unstack(templates, axis=1))
    #
    #     for i, img in enumerate(all_imgs):
    #         if img.shape[-1] == 1:
    #             all_imgs[i] = tf.image.grayscale_to_rgb(img)
    #
    #     img_with_templates = plot.concat_images(all_imgs, 1, vertical=False)
    #
    #     def render_corr(x, y):
    #         corr = abs(plot.correlation(x, y))
    #         rendered_corr = tf.expand_dims(_render_activations(corr), 0)
    #         return plot.concat_images(
    #             tf.unstack(rendered_corr, axis=1), 3, vertical=False)
    #
    #     if label is not None:
    #
    #         posterior_label_corr = render_corr(normalized_mass_expplained_by_capsule,
    #                                            label_one_hot)
    #         prior_label_corr = render_corr(res.caps_presence_prob, label_one_hot)
    #         label_corr = plot.concat_images([prior_label_corr, posterior_label_corr],
    #                                         3,
    #                                         vertical=True)
    #     else:
    #         label_corr = tf.zeros_like(img)
    #
    #     n_examples = min(int(shape[0]), 16)
    #     plot_params = dict(
    #         img_with_templates=dict(
    #             grid_height=n_examples,
    #             zoom=3.,
    #         ))
    #
    #     templates = res.templates
    #     if len(templates.shape) == 5:
    #         if templates.shape[0] == 1:
    #             templates = tf.squeeze(templates, 0)
    #
    #         else:
    #             templates = templates[:n_examples]
    #             templates = plot.concat_images(
    #                 tf.unstack(templates, axis=1), 1, vertical=False)
    #             plot_params['templates'] = dict(grid_height=n_examples)
    #
    #     plot_dict = dict(
    #         templates=templates,
    #         img_with_templates=img_with_templates[:n_examples],
    #         label_corr=label_corr,
    #     )
    #
    #     return plot_dict, plot_params
