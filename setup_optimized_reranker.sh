#!/bin/bash

# Local Reranker Optimization Setup Script
# This script ensures all dependencies are installed for optimal local reranking

echo "🚀 Setting up optimized local reranker..."

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]] || [[ "$CONDA_DEFAULT_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: ${VIRTUAL_ENV:-$CONDA_DEFAULT_ENV}"
else
    echo "⚠️ Warning: No virtual environment detected. Consider using one for better dependency management."
fi

# Install core dependencies
echo "📦 Installing core dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 2>/dev/null || \
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install sentence-transformers transformers

# Optional: Install additional reranker models
echo "🤖 Installing additional model dependencies..."
pip install FlagEmbedding  # For BAAI rerankers

# Test installation
echo "🧪 Testing installation..."
python3 -c "
try:
    import torch
    print(f'✅ PyTorch {torch.__version__} installed')
    print(f'✅ CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'   GPU: {torch.cuda.get_device_name(0)}')
        print(f'   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    
    import sentence_transformers
    print(f'✅ Sentence Transformers {sentence_transformers.__version__} installed')
    
    from transformers import AutoTokenizer
    print('✅ Transformers library working')
    
    print('🎉 All dependencies successfully installed!')
    
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

echo ""
echo "📋 Next Steps:"
echo "1. Update your configuration to use optimized reranker:"
echo "   export CONFIG_PATH=run_configs/optimized_local.json"
echo ""
echo "2. Restart your application:"
echo "   docker-compose up --build"
echo ""
echo "3. Run performance tests:"
echo "   python api/test_reranker_performance.py"
echo ""
echo "🎯 For more options, see: docs/LOCAL_RERANKER_OPTIMIZATION.md"
