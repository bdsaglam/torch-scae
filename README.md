

# torch-scae 

PyTorch implementation of [Stacked Capsule Auto-Encoders](http://arxiv.org/abs/1906.06818) \[1\].

Ported from [official implementation](https://github.com/akosiorek/stacked_capsule_autoencoders) with TensorFlow v1. 
The architecture of model and hyper-parameters are kept same. 
However, some parts are refactored for ease of use. 

Please, open an issue for bugs and inconsistencies with original implementation.
> ⚠️: The performance of this implementation is inferior than the original due to an unknown bug.
There is already an open issue for this, but it has been resolved yet.

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
python -m torch_scae_experiments.mnist.train +trainer.gpus=1
```

You can customize model hyperparameters and training with Hydra syntax.
```bash
python -m torch_scae_experiments.mnist.train \
    data_loader.batch_size=32 \
    optimizer.learning_rate=1e-4 \
    model.n_part_caps=16 \
    trainer.max_epochs=100 
```

### Results
#### Image reconstructions
After training for 5 epochs

![logo](https://raw.githubusercontent.com/bdsaglam/torch-scae/master/.resources/mnist-recons.png)

*Fig 1. Rows: original image, bottom-up reconstructions and top-down reconstructions*

## References

1. Kosiorek, A. R., Sabour, S., Teh, Y. W., & Hinton, G. E. (2019). 
Stacked Capsule Autoencoders. NeurIPS. 
http://arxiv.org/abs/1906.06818
