from setuptools import setup, find_packages

setup(
    name='torch_scae',
    version=1.0,
    packages=find_packages(),
    install_requires=[
        'monty==3.0.2',
        'numpy==1.17.2',
        'hydra-core==1.0.0',
        'torch==1.4.0',
        'torchvision==0.5.0',
        'pytorch-lightning==0.9.0',
    ],
    package_data={
        'torch_scae_experiments': ['configs/*.yaml', 'configs/**/*.yaml']
    },
)
