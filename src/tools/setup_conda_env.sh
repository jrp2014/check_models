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
# - Creates conda environment with Python 3.13
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

# Ensure we are in the project root (src directory)
cd "$(dirname "$0")/.." || exit 1

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
    # Try to find conda if not in PATH
    if ! command -v conda &> /dev/null; then
        if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
            # shellcheck disable=SC1091
            source "$HOME/miniconda3/etc/profile.d/conda.sh"
        elif [ -f "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh" ]; then
            # shellcheck disable=SC1091
            source "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh"
        elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
            # shellcheck disable=SC1091
            source "$HOME/anaconda3/etc/profile.d/conda.sh"
        fi
    fi

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

    # Create new environment with Python 3.13
    log_info "Creating new environment with Python 3.13..."
    conda create -n "$ENV_NAME" python=3.13 -y

    log_success "Environment '$ENV_NAME' created successfully"
}

# Install dependencies
install_dependencies() {
    log_info "Installing dependencies in environment: $ENV_NAME"

    # Activate environment
    # Initialize conda for this shell session
    echo "Initializing conda for bash..."
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"

    # Check for pyproject.toml
    if [[ ! -f "pyproject.toml" ]]; then
        log_warn "pyproject.toml not found in current directory"
        log_warn "Please run this script from the src directory"
        exit 1
    fi

    log_info "Installing project in editable mode (installs core dependencies)..."
    pip install -e .

    # Development dependencies (optional)
    read -p "Do you want to install development dependencies (ruff, mypy, pytest)? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Installing development dependencies..."
        # Try to install via extras if available, otherwise fallback to manual
        if grep -q "dev =" pyproject.toml; then
             pip install -e ".[dev]"
        else
             pip install "ruff>=0.1.0" "mypy>=1.8.0" "pytest>=8.0.0" "pytest-cov>=4.0.0"
        fi
    fi

    # Optional: PyTorch stack (torch, torchvision, torchaudio)
    read -p "Install PyTorch stack (torch, torchvision, torchaudio)? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Installing PyTorch packages..."
        # On Apple Silicon, standard PyPI wheels provide MPS acceleration.
        if grep -q "torch =" pyproject.toml; then
            pip install -e ".[torch]"
        else
            pip install \
                "torch>=2.2.0" \
                "torchvision>=0.17.0" \
                "torchaudio>=2.2.0"
        fi
        log_success "Installed PyTorch packages"
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
try:
    import torch, torchvision, torchaudio
    print(f'✓ PyTorch version: {torch.__version__}')
except Exception:
    pass
print('✓ All core packages imported successfully')
print(f'✓ MLX version: {mx.__version__}')
"

    # Install huggingface_hub CLI tools
    log_info "Installing huggingface_hub CLI tools..."
    pip install "huggingface_hub[cli]"

    # Check if CLI is available
    if command -v check_models &> /dev/null; then
        log_success "CLI command 'check_models' is available"
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
   ${BLUE}check_models --help${NC}

4. Example usage:
   ${BLUE}check_models --image /path/to/image.jpg${NC}
   ${BLUE}check_models --models "microsoft/Florence-2-large"${NC}

Optional installs:
    - To include PyTorch stack during setup, answer 'y' when prompted, or later run:
    ${BLUE}pip install -e ".[torch]"${NC}

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
