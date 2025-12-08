#!/bin/bash
# Setup a restricted Python environment for child LM code execution.
#
# This creates a minimal virtual environment with only numpy and scipy,
# preventing child LM generated code from accessing matplotlib, anthropic,
# or other packages that shouldn't be available during evaluation.
#
# Usage:
#   ./scripts/setup_restricted_env.sh
#   # or with custom packages:
#   ./scripts/setup_restricted_env.sh numpy scipy pandas
#
# Then configure your experiment yaml:
#   evaluation:
#     evaluator_kwargs:
#       python_executable: ".venv-restricted/bin/python"

set -e

VENV_DIR=".venv-restricted"

# Default packages for circle packing (only numpy and scipy needed)
DEFAULT_PACKAGES="numpy scipy"

# Use provided packages or defaults
PACKAGES="${@:-$DEFAULT_PACKAGES}"

echo "Setting up restricted environment in $VENV_DIR..."
echo "Packages to install: $PACKAGES"

# Check if uv is available
if command -v uv &> /dev/null; then
    echo "Using uv for environment setup..."

    # Remove existing environment if present
    if [ -d "$VENV_DIR" ]; then
        echo "Removing existing $VENV_DIR..."
        rm -rf "$VENV_DIR"
    fi

    # Create new virtual environment
    uv venv "$VENV_DIR"

    # Install packages
    uv pip install $PACKAGES --python "$VENV_DIR/bin/python"

else
    echo "uv not found, falling back to standard venv..."

    # Remove existing environment if present
    if [ -d "$VENV_DIR" ]; then
        echo "Removing existing $VENV_DIR..."
        rm -rf "$VENV_DIR"
    fi

    # Create new virtual environment
    python -m venv "$VENV_DIR"

    # Install packages
    "$VENV_DIR/bin/pip" install --upgrade pip
    "$VENV_DIR/bin/pip" install $PACKAGES
fi

echo ""
echo "Restricted environment created at: $VENV_DIR"
echo ""
echo "Packages available in restricted environment:"
if command -v uv &> /dev/null; then
    uv pip list --python "$VENV_DIR/bin/python"
else
    "$VENV_DIR/bin/pip" list
fi

echo ""
echo "Verifying matplotlib is NOT available..."
if "$VENV_DIR/bin/python" -c 'import matplotlib' 2>/dev/null; then
    echo "ERROR: matplotlib should not be available!"
    exit 1
else
    echo "OK: matplotlib correctly unavailable"
fi

echo ""
echo "Verifying numpy IS available..."
if "$VENV_DIR/bin/python" -c 'import numpy; print(f"numpy version: {numpy.__version__}")'; then
    echo "OK: numpy available"
else
    echo "ERROR: numpy should be available!"
    exit 1
fi

echo ""
echo "To use this environment, add to your config yaml:"
echo "  evaluation:"
echo "    evaluator_kwargs:"
echo "      python_executable: \"$VENV_DIR/bin/python\""
