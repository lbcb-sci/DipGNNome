# DipGNNome Installation Guide

This guide provides multiple ways to install DipGNNome and all its dependencies.

## Prerequisites

### System Dependencies
Before installing DipGNNome, ensure you have the following system dependencies:

- **Git** - for cloning repositories
- **Make** - for building external tools
- **GCC or compatible C compiler** - for compiling external tools
- **Python 3.8+** - for running the Python code
- **Conda or Miniconda** (recommended) - for environment management

### Optional GPU Support
For GPU acceleration, you'll also need:
- **NVIDIA GPU** with CUDA support
- **NVIDIA drivers** (latest version)
- **CUDA toolkit** (version 11.8 or compatible)

## Installation Methods

### Method 1: Automated Installation (Recommended)

The easiest way to install DipGNNome is using our automated installation script:

```bash
python install_tools.py
```

This script will:
- Install all required Python packages via pip
- Download and compile external bioinformatics tools (hifiasm, PBSIM3, yak)
- Set up the vendor directory with compiled tools

### Method 2: Conda Environment Setup

#### For GPU Support:
```bash
# Create and activate GPU environment
./setup_gpu.sh

# Or manually:
conda env create -f environment_gpu.yml
conda activate dipgnnome_gpu
python install_tools.py
```

#### For CPU-Only:
```bash
# Create and activate CPU environment
./setup_cpu.sh

# Or manually:
conda env create -f environment_cpu.yml
conda activate dipgnnome_cpu
python install_tools.py
```

### Method 3: Manual Installation

#### Python Dependencies
```bash
# Install Python packages
pip install -r requirements.txt

# Or install individually:
pip install torch torchvision torchaudio dgl
pip install numpy scipy pandas scikit-learn
pip install networkx biopython edlib pyliftover
pip install tqdm pyyaml wandb
```

#### External Tools
The external bioinformatics tools need to be compiled from source:

1. **hifiasm** (version 0.25.0):
   ```bash
   git clone https://github.com/chhylp123/hifiasm.git --branch 0.25.0 --single-branch
   cd hifiasm
   make
   ```

2. **PBSIM3**:
   ```bash
   git clone https://github.com/yukiteruono/pbsim3.git
   cd pbsim3
   ./configure && make && make install
   ```

3. **yak**:
   ```bash
   git clone https://github.com/lh3/yak.git
   cd yak
   make
   ```

## Verification

After installation, verify that everything is working:

### Test Python Dependencies
```bash
python -c "import torch, dgl, networkx, Bio; print('All packages imported successfully')"
```

### Test GPU Support (if applicable)
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Test DipGNNome Installation
```bash
# On macOS, you may need to set this environment variable to avoid OpenMP conflicts
export KMP_DUPLICATE_LIB_OK=TRUE
python test_debug.py
```

## Environment Files

### environment_cpu.yml
- CPU-optimized PyTorch installation
- All required Python packages
- System tools (git, make, gcc)
- Environment variables for CPU optimization

### environment_gpu.yml
- GPU-enabled PyTorch with CUDA support
- All required Python packages
- NVIDIA CUDA toolkit
- Environment variables for GPU optimization

### requirements.txt
- Complete list of Python package dependencies
- Version constraints for compatibility
- Can be used with pip for manual installation

## Troubleshooting

### Common Issues

1. **Compilation Errors for External Tools**:
   - Ensure you have gcc/g++ installed
   - On macOS: `xcode-select --install`
   - On Ubuntu: `sudo apt-get install build-essential`

2. **CUDA/GPU Issues**:
   - Verify NVIDIA drivers: `nvidia-smi`
   - Check CUDA installation: `nvcc --version`
   - Ensure PyTorch CUDA version matches your CUDA toolkit

3. **Missing Dependencies**:
   - Run `python install_tools.py` to install missing packages
   - Check that all system dependencies are installed

4. **OpenMP Library Conflicts (macOS)**:
   - If you get "OMP: Error #15" when running test_debug.py, set: `export KMP_DUPLICATE_LIB_OK=TRUE`
   - This is a common issue on macOS with multiple OpenMP installations

5. **Permission Errors**:
   - Some tools may need to be installed with sudo
   - Consider using conda environments to avoid permission issues

### Getting Help

If you encounter issues:
1. Check the error messages carefully
2. Ensure all system dependencies are installed
3. Try the conda environment approach
4. Check the DipGNNome documentation or GitHub issues

## File Structure After Installation

```
DipGNNome/
├── vendor/                    # External tools directory
│   ├── hifiasm_025/          # hifiasm installation
│   ├── pbsim3/               # PBSIM3 installation
│   └── yak/                  # yak installation
├── environment_cpu.yml       # CPU conda environment
├── environment_gpu.yml       # GPU conda environment
├── requirements.txt          # Python dependencies
├── install_tools.py          # Installation script
├── setup_cpu.sh             # CPU setup script
├── setup_gpu.sh             # GPU setup script
└── INSTALLATION.md          # This file
```

## Next Steps

After successful installation:
1. Activate your environment: `conda activate dipgnnome_gpu` (or `dipgnnome_cpu`)
2. Run the test script: `python test_debug.py`
3. Follow the main DipGNNome documentation for usage instructions
4. Configure your datasets and run the pipeline

## Notes

- The installation process may take 10-30 minutes depending on your system
- GPU environments require more disk space due to CUDA dependencies
- Some external tools may fail to compile on certain systems - this is normal and the pipeline can still work with available tools
- Always use the same environment (CPU or GPU) for consistency
