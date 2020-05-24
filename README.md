

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

[PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) is used for training.

```bash
python -m torch_scae_experiments.mnist.train --batch_size 32 --learning_rate 1e-4
```

### Results
#### Image reconstructions
After training for 5 epochs

![logo](https://raw.githubusercontent.com/bdsaglam/torch-scae/master/.resources/mnist-recons.png)

*Fig 1. Rows: original image, bottom-up reconstructions and top-down reconstructions*


## Custom model

For a custom model, create a parameter dictionary similar to the one at 
```torch_scae_experiments.mnist.hparams.model_params``` .

Then, create full config with ```torch_scae.factory.make_config```.
```python
from torch_scae import factory

custom_params = dict(...)
config = factory.make_config(**custom_params)
model = factory.make_scae(your_custom_config)
```


### References

1. Kosiorek, A. R., Sabour, S., Teh, Y. W., & Hinton, G. E. (2019). 
Stacked Capsule Autoencoders. NeurIPS. 
http://arxiv.org/abs/1906.06818
