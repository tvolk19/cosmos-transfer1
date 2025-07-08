#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting setup..."

# Optionally ensure pip is up to date
echo "Upgrading pip..."
# python3 -m pip install --upgrade pip --break-system-packages

# Install gradio
echo "Installing gradio..."
python3 -m pip install gradio --break-system-packages
echo "Finished Installing gradio..."

# echo "Upgrading transformers..."
# python3 -m pip install --upgrade transformers
# echo "Finished Upgrading transformers..."

echo "Installing libmagic system library..."
apt-get update
apt-get install -y libmagic1
echo "Finished Installing libmagic system library..."

# Install python magic
echo "Installing python magic..."
python3 -m pip install python-magic --break-system-packages
echo "Finished Installing python magic..."


echo "Setup complete!"
