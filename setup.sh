#!/bin/bash

# Art Trend Classifier - Setup Script
# This script sets up the project on a new device

set -e  # Exit on any error

echo "🎨 Art Trend Classifier - Setup Script"
echo "======================================"

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "Python version: $python_version"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
    echo "❌ Error: Python 3.8+ is required. Current version: $python_version"
    exit 1
fi
echo "✅ Python version is compatible"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Verify installation
echo "🔍 Verifying installation..."
python -c "
import fastapi
import torch
import torchvision
import albumentations
import cv2
import numpy as np
import PIL
import sklearn
import matplotlib
import mangum
import boto3
print('✅ All core dependencies imported successfully')
"

# Check if model files exist
echo "🤖 Checking model files..."
if [ -f "ml_model/model/model_best_82_73.pth" ]; then
    echo "✅ Model file found"
else
    echo "⚠️ Warning: Model file not found. You may need to train a model first."
    echo "   Run: cd ml_model && python train.py"
fi

# Test backend import
echo "🔧 Testing backend import..."
cd backend
python -c "import main; print('✅ Backend imports successfully')"
cd ..

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📖 Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Start backend server: cd backend && python main.py"
echo "3. Open frontend: open frontend/index.html or frontend/landing.html"
echo ""
echo "🐳 Alternative - Docker setup:"
echo "1. Build and run: docker-compose up --build"
echo ""
echo "📚 For more information, see README.md"
