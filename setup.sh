#!/bin/bash
# Quick setup script for local development

echo "🚀 Setting up Grocery Stock Predictor development environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Install development dependencies
echo "🛠️  Installing development dependencies..."
pip install pytest pytest-cov black isort flake8 pylint bandit safety

# Install package in editable mode
echo "📦 Installing package in editable mode..."
pip install -e .

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run tests: pytest tests/ -v"
echo "3. Format code: black src/"
echo "4. Check linting: flake8 src/"
echo ""
echo "Happy coding! 🎉"
