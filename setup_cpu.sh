#!/bin/bash

# DipGNNome CPU Environment Setup Script
# This script creates a conda environment with CPU-only support for DipGNNome

set -e  # Exit on any error

echo "=== DipGNNome CPU Environment Setup ==="
echo "This script will create a conda environment with CPU-only support."
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed or not in PATH."
    echo "Please install Anaconda or Miniconda first."
    echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create the conda environment
echo "Creating conda environment 'dipgnnome_cpu'..."
conda env create -f environment_cpu.yml

# Activate the environment and install additional dependencies
echo ""
echo "Activating environment and installing additional dependencies..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate dipgnnome_cpu

# Install external tools using the Python script
echo ""
echo "Installing external bioinformatics tools..."
python install_tools.py

echo ""
echo "=== CPU Environment Setup Complete ==="
echo "To activate the environment, run:"
echo "  conda activate dipgnnome_cpu"
echo ""
echo "To test the DipGNNome installation, run:"
echo "  python test_debug.py"
echo ""
echo "Note: This environment is optimized for CPU-only execution."
echo "For GPU support, use setup_gpu.sh instead."
