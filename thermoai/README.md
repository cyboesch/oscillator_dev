# ThermoAI

A package for research into thermodynamic AI design.

## Set up environment

```bash
conda create -n thermoai python=3.12 && conda activate thermoai && conda install 'jaxlib=0.4.28=*cuda*' 'jax=0.4.28' cuda-nvcc -c conda-forge -c nvidia
```

## Installation

Install the package in development mode:

```bash
cd thermoai
conda activate thermoai
python -m pip install -e .
```

For CUDA support on LPC, install the appropriate version of JAX and JAXLIB:

```bash
python -m pip install -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html jax==0.4.28 jaxlib==0.4.28+cuda12.cudnn89
```
i