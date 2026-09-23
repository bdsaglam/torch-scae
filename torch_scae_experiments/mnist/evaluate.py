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

"""Evaluates a trained model like `eval_mnist_model.py` of the reference code.

Reports the accuracy of the linear probes and the unsupervised accuracy:
k-means with 10 clusters on object capsule presences of the training set,
then clusters are matched to labels by bipartite matching.

    python -m torch_scae_experiments.mnist.evaluate path/to/last.ckpt
"""

import argparse

import numpy as np
import sklearn.cluster
import torch
from omegaconf import OmegaConf
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from torch_scae_experiments.mnist.experiment import MNISTExperiment


def bipartite_match_accuracy(pred, label, n_classes=10):
    counts = np.zeros([n_classes, n_classes])
    np.add.at(counts, (label, pred), 1)
    row, col = linear_sum_assignment(-counts)
    return counts[row, col].sum() / len(label)


@torch.no_grad()
def collect(model, dataset, batch_size, device):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=8)
    out = dict(prior_pres=[], posterior_pres=[], label=[],
               prior_pred=[], posterior_pred=[])
    for image, label in loader:
        res = model.scae(image=image.to(device))
        out['prior_pres'].append(res.caps_presence.cpu())
        out['posterior_pres'].append(res.posterior_mixing_prob.sum(-1).cpu())
        out['prior_pred'].append(res.prior_cls_logits.argmax(-1).cpu())
        out['posterior_pred'].append(res.posterior_cls_logits.argmax(-1).cpu())
        out['label'].append(label)
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('--data-dir', default='./data')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = OmegaConf.create(ckpt['hyper_parameters'])
    cfg.compile = False
    model = MNISTExperiment(cfg)
    state = {k.replace('_orig_mod.', ''): v
             for k, v in ckpt['state_dict'].items()}
    model.load_state_dict(state)
    model.to(device).eval()

    transform = model.make_transforms()
    results = {
        subset: collect(
            model,
            MNIST(args.data_dir, train=subset == 'train', download=True,
                  transform=transform),
            cfg.data_loader.batch_size, device)
        for subset in ('train', 'test')
    }

    print('Linear classification accuracy:')
    for subset, r in results.items():
        prior = (r['prior_pred'] == r['label']).mean()
        posterior = (r['posterior_pred'] == r['label']).mean()
        print(f'\t{subset}: prior={prior:.4f}, posterior={posterior:.4f}')

    print('Bipartite matching classification accuracy:')
    for field in ('posterior_pres', 'prior_pres'):
        kmeans = sklearn.cluster.KMeans(n_clusters=10, max_iter=1000,
                                        n_init=10, random_state=args.seed)
        kmeans.fit(results['train'][field])
        accs = {
            subset: bipartite_match_accuracy(kmeans.predict(r[field]),
                                             r['label'])
            for subset, r in results.items()
        }
        print(f"\t{field}: train_acc={accs['train']:.4f}, "
              f"test_acc={accs['test']:.4f}")


if __name__ == '__main__':
    main()
