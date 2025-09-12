#!/usr/bin/env bash
#
# setup_conda_env.sh - Create conda environment for MLX VLM Check
#
# This script creates a conda environment with all dependencies needed to run
# the MLX Vision Language Model checker script. It reads dependencies from
# pyproject.toml and installs them appropriately.
#
# Key Features:
# - Validates macOS/Apple Silicon compatibility
# - Creates conda environment with Python 3.12
# - Installs all runtime and optional dependencies  
# - Optionally installs development tools
# - Verifies installation and provides usage instructions
#
# Usage:
#   ./setup_conda_env.sh [ENV_NAME]
#
# Environment name defaults to 'mlx-vlm' if not provided.
#

set -euo pipefail

# Default environment name
DEFAULT_ENV_NAME="mlx-vlm"

# Help function
show_help() {
    cat << EOF
setup_conda_env.sh - Create conda environment for MLX VLM Check

USAGE:
    ./setup_conda_env.sh [ENV_NAME]
    ./setup_conda_env.sh --help

ARGUMENTS:
    ENV_NAME    Name for the conda environment (default: mlx-vlm)

OPTIONS:
    --help, -h  Show this help message

DESCRIPTION:
    This script creates a conda environment with all dependencies needed to run
    the MLX Vision Language Model checker script. It reads dependencies from
    pyproject.toml and installs them appropriately.

REQUIREMENTS:
    - macOS (preferably with Apple Silicon)
    - conda or miniconda installed
    - pyproject.toml file in current directory (for development install)

EXAMPLES:
    ./setup_conda_env.sh                 # Create environment named 'mlx-vlm'
    ./setup_conda_env.sh my-vlm-env      # Create environment named 'my-vlm-env'

EOF
}

# Parse arguments
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    show_help
    exit 0
fi

ENV_NAME="${1:-$DEFAULT_ENV_NAME}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if conda is available
check_conda() {
    if ! command -v conda &> /dev/null; then
        log_error "conda is not installed or not in PATH"
        log_info "Please install Miniconda or Anaconda first:"
        log_info "  https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    log_info "Found conda: $(conda --version)"
}

# Check if we're on macOS (required for MLX)
check_platform() {
    if [[ "$(uname)" != "Darwin" ]]; then
        log_error "MLX requires macOS with Apple Silicon"
        log_info "This script is designed for macOS only"
        exit 1
    fi
    
    # Check for Apple Silicon
    if [[ "$(uname -m)" != "arm64" ]]; then
        log_warn "MLX is optimized for Apple Silicon (arm64)"
        log_warn "You may encounter performance issues on Intel Macs"
    fi
}

# Create conda environment
create_environment() {
    log_info "Creating conda environment: $ENV_NAME"
    
    # Check if environment already exists
    if conda env list | grep -q "^$ENV_NAME "; then
        log_warn "Environment '$ENV_NAME' already exists"
        read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Removing existing environment..."
            conda env remove -n "$ENV_NAME" -y
        else
            log_info "Updating existing environment instead..."
            return 0
        fi
    fi
    
    # Create new environment with Python 3.12
    log_info "Creating new environment with Python 3.12..."
    conda create -n "$ENV_NAME" python=3.12 -y
    
    log_success "Environment '$ENV_NAME' created successfully"
}

# Install dependencies
install_dependencies() {
    log_info "Installing dependencies in environment: $ENV_NAME"
    
    # Activate environment
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
    
    # Core runtime dependencies (from pyproject.toml)
    log_info "Installing core MLX dependencies..."
    pip install \
        "mlx>=0.29.1" \
        "mlx-vlm>=0.0.9" \
        "mlx-lm>=0.23.0"
    
    log_info "Installing core Python data libraries..."
    pip install \
        "numpy"
    
    log_info "Installing image processing dependencies..."
    pip install \
        "Pillow>=10.3.0" \
        "opencv-python>=4.12.0.88"
    
    log_info "Installing ML/AI frameworks..."
    pip install \
        "transformers>=4.53.0" \
        "datasets>=2.19.1"
    
    log_info "Installing model and caching utilities..."
    pip install \
        "huggingface-hub>=0.23.0"
    
    log_info "Installing serialization and configuration..."
    pip install \
        "protobuf" \
        "pyyaml" \
        "jinja2"
    
    log_info "Installing network and API dependencies..."
    pip install \
        "requests>=2.31.0" \
        "fastapi>=0.95.1" \
        "uvicorn"
    
    log_info "Installing audio processing..."
    pip install \
        "soundfile>=0.13.1"
    
    log_info "Installing reporting and system utilities..."
    pip install \
        "tabulate>=0.9.0" \
        "tqdm>=4.66.2" \
        "tzlocal>=5.0"
    
    # Optional extras
    log_info "Installing optional extras for enhanced functionality..."
    pip install \
        "psutil>=5.9.0" \
        "tokenizers>=0.15.0"
    
    # Development dependencies (optional)
    read -p "Do you want to install development dependencies (ruff, mypy, pytest)? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Installing development dependencies..."
        pip install \
            "ruff>=0.1.0" \
            "mypy>=1.8.0" \
            "pytest>=8.0.0" \
            "pytest-cov>=4.0.0"
    fi
    
    # Install this package in development mode
    if [[ -f "pyproject.toml" ]]; then
        log_info "Installing mlx-vlm-check in development mode..."
        pip install -e .
    else
        log_warn "pyproject.toml not found in current directory"
        log_warn "Run this script from the vlm directory to install the package"
    fi
    
    log_success "All dependencies installed successfully"
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."
    
    # Check Python version
    python_version=$(python --version)
    log_info "Python version: $python_version"
    
    # Check key packages
    log_info "Checking key packages..."
    python -c "
import mlx.core as mx
import mlx_vlm
from PIL import Image
import huggingface_hub
import tabulate
import tzlocal
print('✓ All core packages imported successfully')
print(f'✓ MLX version: {mx.__version__}')
"
    
    # Check if CLI is available
    if command -v mlx-vlm-check &> /dev/null; then
        log_success "CLI command 'mlx-vlm-check' is available"
    else
        log_warn "CLI command not found. You may need to reinstall with 'pip install -e .'"
    fi
    
    log_success "Installation verification complete"
}

# Print usage instructions
print_usage() {
    cat << EOF

${GREEN}Environment setup complete!${NC}

To use the MLX VLM environment:

1. Activate the environment:
   ${BLUE}conda activate $ENV_NAME${NC}

2. Run the script directly:
   ${BLUE}python check_models.py --help${NC}

3. Or use the CLI command:
   ${BLUE}mlx-vlm-check --help${NC}

4. Example usage:
   ${BLUE}mlx-vlm-check --image /path/to/image.jpg${NC}
   ${BLUE}mlx-vlm-check --models "microsoft/Florence-2-large"${NC}

5. Deactivate when done:
   ${BLUE}conda deactivate${NC}

For more information, see README.md

EOF
}

# Main execution
main() {
    log_info "Setting up MLX VLM conda environment: $ENV_NAME"
    log_info "Script location: $(pwd)"
    
    check_conda
    check_platform
    create_environment
    install_dependencies
    verify_installation
    print_usage
    
    log_success "Setup complete! Environment '$ENV_NAME' is ready to use."
}

# Run main function
main "$@"