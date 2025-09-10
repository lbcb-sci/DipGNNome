import os
import subprocess
import sys


def install_python_packages():
    """Install required Python packages via pip."""
    print("\nInstalling Python packages...")
    
    # Core scientific computing packages
    packages = [
        "numpy>=1.21.0",
        "scipy>=1.7.0", 
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
    ]
    
    # Machine learning and deep learning
    packages.extend([
        "torch>=1.12.0",
        "torchvision>=0.13.0", 
        "torchaudio>=0.12.0",
        "dgl>=1.0.0",
    ])
    
    # Graph processing and network analysis
    packages.extend([
        "networkx>=2.6.0",
    ])
    
    # Bioinformatics packages
    packages.extend([
        "biopython>=1.79",
        "edlib>=1.3.9",
        "pyliftover>=0.4",
    ])
    
    # Progress bars and utilities
    packages.extend([
        "tqdm>=4.62.0",
        "pyyaml>=6.0",
        "argparse",
    ])
    
    # Experiment tracking (optional)
    packages.extend([
        "wandb>=0.12.0",
    ])
    
    # Install packages
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to install {package}: {e}")
            print("You may need to install this package manually.")


def install_external_tools():
    """Install external bioinformatics tools."""
    save_dir = 'vendor'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # Install hifiasm
    hifiasm_dir_name = f'hifiasm_025'
    if os.path.isfile(os.path.join(save_dir, hifiasm_dir_name, 'hifiasm')):
        print(f'\nFound hifiasm! Skipping installation...\n')
    else:
        print(f'\nInstalling hifiasm...')
        try:
            subprocess.run(f'git clone https://github.com/chhylp123/hifiasm.git --branch 0.25.0 --single-branch {hifiasm_dir_name}', shell=True, cwd=save_dir, check=True)
            hifiasm_dir = os.path.join(save_dir, hifiasm_dir_name)
            subprocess.run(f'make', shell=True, cwd=hifiasm_dir, check=True)
            print("hifiasm installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to install hifiasm: {e}")
            print("You may need to install hifiasm manually.")

    # Install PBSIM3
    pbsim_dir_name = f'pbsim3'
    if os.path.isfile(os.path.join(save_dir, pbsim_dir_name, 'src', 'pbsim')):
        print(f'\nFound PBSIM3! Skipping installation...\n')
    else:
        print(f'\nInstalling PBSIM3...')
        try:
            subprocess.run(f'git clone https://github.com/yukiteruono/pbsim3.git {pbsim_dir_name}', shell=True, cwd=save_dir, check=True)
            pbsim_dir = os.path.join(save_dir, pbsim_dir_name)
            subprocess.run(f'./configure; make; make install', shell=True, cwd=pbsim_dir, check=True)
            print("PBSIM3 installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to install PBSIM3: {e}")
            print("You may need to install PBSIM3 manually.")

    # Install yak
    tool_name = f'yak'
    if os.path.isfile(os.path.join(save_dir, tool_name, 'yak')):
        print(f'\nFound {tool_name}! Skipping installation...\n')
    else:
        print(f'\nInstalling {tool_name}...')
        try:
            subprocess.run(f'git clone https://github.com/lh3/yak.git', shell=True, cwd=save_dir, check=True)
            tool_dir = os.path.join(save_dir, tool_name)
            subprocess.run(f'make', shell=True, cwd=tool_dir, check=True)
            print("yak installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to install yak: {e}")
            print("You may need to install yak manually.")


def install():
    """Main installation function."""
    print("=== DipGNNome Installation ===")
    print("This script will install all required dependencies for DipGNNome.")
    print("Make sure you have the following system dependencies installed:")
    print("- Git")
    print("- Make")
    print("- GCC or compatible C compiler")
    print("- Python 3.8+")
    print()
    
    # Install Python packages first
    install_python_packages()
    
    # Install external tools
    install_external_tools()
    
    print("\n=== Installation Complete ===")
    print("All dependencies have been installed.")
    print("You can now run the DipGNNome pipeline.")
    print()
    print("Note: If you encounter any issues, you may need to:")
    print("1. Install system dependencies (git, make, gcc)")
    print("2. Install CUDA toolkit for GPU support")
    print("3. Manually install any failed packages")

if __name__ == '__main__':
    install()

