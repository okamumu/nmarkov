# nmarkov

Python module for numerical computation of Markov chains

## Installation

### Install with conda

The recommendation is to use Anaconda/Miniconda with your own environment for nmarkov.

```sh
conda create -n nmarkov python=3.6 jupyter numpy scipy pybind11
conda activate nmarkov
pip install git+https://github.com/okamumu/nmarkov.git
```

For Jupyter, make the kernel for the environment `nmarkov`
```
conda activate nmarkov
ipython kernel install --user --name nmarkov
```

### Install with pip

```sh
pip install git+https://github.com/okamumu/nmarkov.git
```

Requriements:
- pybind11
- numpy
- scipy

