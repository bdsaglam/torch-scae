

# torch-scae 

PyTorch implementation of [Stacked Capsule Auto-Encoders](http://arxiv.org/abs/1906.06818) \[1\].

Ported from [official implementation](https://github.com/akosiorek/stacked_capsule_autoencoders) with TensorFlow v1. 
The architecture of model and hyper-parameters are kept same. 
However, some parts are refactored for ease of use. 

Please, open an issue for bugs and inconsistencies with original implementation.

---
## Installation   
```bash
# clone project   
git clone https://github.com/bdsaglam/torch-scae   

# install project   
cd torch-scae
pip install -e .
 ```
 
## Train with MNIST [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bdsaglam/torch-scae/blob/master/torch_scae_experiments/mnist/train.ipynb)

It uses [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) 
for training and [Hydra](https://hydra.cc) for configuration management.

```bash
# CPU
python -m torch_scae_experiments.mnist.train

# GPU
python -m torch_scae_experiments.mnist.train +trainer.accelerator=gpu +trainer.devices=1
```

You can customize model hyperparameters and training with Hydra syntax.
```bash
python -m torch_scae_experiments.mnist.train \
    data_loader.batch_size=32 \
    optimizer.learning_rate=1e-4 \
    model.n_part_caps=16 \
    trainer.max_steps=100000
```

Evaluate a checkpoint with linear probes and with k-means clustering plus
bipartite matching, as in the paper:
```bash
python -m torch_scae_experiments.mnist.evaluate path/to/last.ckpt
```

### Results
Unsupervised classification accuracy (%) on the 40x40 MNIST test set after
300k steps with the default configuration, which follows the reference
`run_mnist.sh` (40 part capsules, 32 object capsules).

| | LIN-MATCH | LIN-PRED |
|---|---|---|
| Paper [1], Table 1 (5 runs) | 98.7 (0.35) | 99.0 (0.07) |
| This implementation, seed 42 | 98.6 | 98.9 |
| This implementation, seed 43 | 98.7 | 98.8 |

LIN-MATCH fits k-means with 10 clusters on the object capsule presences
of the training set and matches clusters to labels; the table reports the
posterior presences. LIN-PRED is the better of the two linear probes
trained alongside the model. Training takes about 4 hours on an A100.

#### Image reconstructions
![reconstructions](https://raw.githubusercontent.com/bdsaglam/torch-scae/master/.resources/mnist-recons.png)

*Fig 1. Rows: test images and their reconstructions*

## References

1. Kosiorek, A. R., Sabour, S., Teh, Y. W., & Hinton, G. E. (2019). 
Stacked Capsule Autoencoders. NeurIPS. 
http://arxiv.org/abs/1906.06818
