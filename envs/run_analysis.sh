#!/bin/bash

# AI Image Analysis Agent Runner Script
# This script runs the LangGraph-based AI agent to analyze dashboard images

set -e  # Exit on any error

echo "=========================================="
echo "AI Image Analysis Agent"
echo "=========================================="

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Error: .env file not found in the project root"
    echo "Please create a .env file with your OPENAI_API_KEY"
    exit 1
fi

# Check if requirements are installed
echo "Checking dependencies..."
if ! python -c "import langgraph, langchain, openai" 2>/dev/null; then
    echo "Installing required dependencies..."
    pip install -r requirements.txt
fi

# Create necessary directories
echo "Setting up directories..."
mkdir -p assets/images/processed
mkdir -p envs/data

# Run the analysis
echo "Starting image analysis..."
python src/main.py

echo "=========================================="
echo "Analysis completed!"
echo "==========================================" 