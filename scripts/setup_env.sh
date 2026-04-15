#!/bin/bash

echo "Setting up Farmland Crack Detection environment..."

if command -v conda &> /dev/null; then
    echo "Creating conda environment..."
    conda create -n farmland_crack python=3.10 -y
    echo "Activating environment..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate farmland_crack
else
    echo "Conda not found, using pip directly..."
fi

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Checking CUDA..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

echo "Creating necessary directories..."
mkdir -p data/raw data/processed data/annotations
mkdir -p weights outputs

echo "Setup complete!"
