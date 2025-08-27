# HRM-minimal
A minimal implementation of Hierarchical Reasoning Models (HRM)

## Quick Start Guide

### Prerequisites

Ensure PyTorch and CUDA are installed. The repo needs CUDA extensions to be built. If not present, run the following commands:

```bash
# Install CUDA 12.6
CUDA_URL=https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda_12.6.3_560.35.05_linux.run

wget -q --show-progress --progress=bar:force:noscroll -O cuda_installer.run $CUDA_URL
sudo sh cuda_installer.run --silent --toolkit --override

export CUDA_HOME=/usr/local/cuda-12.6

# Install PyTorch with CUDA 12.6
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu126

pip3 install torch torchvision torchaudio --index-url $PYTORCH_INDEX_URL

# Additional packages for building extensions
pip3 install packaging ninja wheel setuptools setuptools-scm
```

Then install FlashAttention. For Hopper GPUs, install FlashAttention 3

```bash
git clone git@github.com:Dao-AILab/flash-attention.git
cd flash-attention/hopper
python setup.py install
```

For Ampere or earlier GPUs, install FlashAttention 2

```bash
pip3 install flash-attn
```

## Install Python Dependencies 🐍

```bash
pip install -r requirements.txt
```

## W&B Integration 📈

This project uses [Weights & Biases](https://wandb.ai/) for experiment tracking and metric visualization. Ensure you're logged in:

```bash
wandb login
```

## Run Experiments

### Quickstart: Solving Sudoku with HRM

Getting started with training an HRM is as simple as editing the config [config](config.py) and running the following command:

```bash
python main.py
```

## Results

Coming soon!
- Eval script
- Streamlit application
- Trained checkpoints
- Benchmarks on a variety of datasets and architectures
- Comparisons with [the OG repo - sapientinc/HRM](https://github.com/sapientinc/HRM)
- Mech-Interp
- Ablations
- Blog
- Tips and tricks, analysis, important takeaways


# Collaborators
- [Harshvardhan Aditya](https://github.com/harshvardhan2707)
- [Ishaan Shrivastava](https://github.com/shrivastava95)
