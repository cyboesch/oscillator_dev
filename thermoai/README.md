# ThermoAI

A package for research into thermodynamic AI design.

## Set up environment

`conda create -n thermoai python=3.12 && conda activate thermoai`
`conda install 'jaxlib=*=*cuda*' jax cuda-nvcc -c conda-forge -c nvidia`


## Installation

Install the package in development mode:

```bash
cd thermoai
conda activate thermoai
python -m pip install -e .
```