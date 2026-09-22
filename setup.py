from setuptools import setup, find_packages

setup(
    name='torch_scae',
    version=1.0,
    packages=find_packages(),
    install_requires=[
        'monty',
        'numpy',
        'hydra-core>=1.3',
        'torch>=2.1',
        'torchvision',
        'lightning>=2.0',
        'tensorboard',
        'scikit-learn',
        'scipy',
    ],
    package_data={
        'torch_scae_experiments': ['configs/*.yaml', 'configs/**/*.yaml']
    },
)
