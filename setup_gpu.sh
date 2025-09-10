#!/bin/bash

# DipGNNome GPU Environment Setup Script
# This script creates a conda environment with GPU support for DipGNNome

set -e  # Exit on any error

echo "=== DipGNNome GPU Environment Setup ==="
echo "This script will create a conda environment with GPU support."
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed or not in PATH."
    echo "Please install Anaconda or Miniconda first."
    echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. GPU support may not work properly."
    echo "Make sure you have NVIDIA drivers and CUDA toolkit installed."
    echo ""
fi

# Create the conda environment
echo "Creating conda environment 'dipgnnome_gpu'..."
conda env create -f environment_gpu.yml --channel conda-forge

# Activate the environment and install additional dependencies
echo ""
echo "Activating environment and installing additional dependencies..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate dipgnnome_gpu

# Install external tools using the Python script
echo ""
echo "Installing external bioinformatics tools..."
python install_tools.py

echo ""
echo "=== GPU Environment Setup Complete ==="
echo "To activate the environment, run:"
echo "  conda activate dipgnnome_gpu"
echo ""
echo "To test GPU support, run:"
echo "  python -c \"import torch; print(f'CUDA available: {torch.cuda.is_available()}')\""
echo ""
echo "To test the DipGNNome installation, run:"
echo "  python test_debug.py"
