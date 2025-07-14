#!/usr/bin/env python3
"""
Quick test to verify the optimized reranker is working properly
"""

import sys
import time
import logging
from typing import List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_reranker_import():
    """Test if we can import the optimized reranker"""
    try:
        from scholarqa.rag.reranker.optimized_local_reranker import OptimizedLocalReranker
        logger.info("✅ Successfully imported OptimizedLocalReranker")
        return True
    except ImportError as e:
        logger.error(f"❌ Failed to import OptimizedLocalReranker: {e}")
        return False

def test_dependencies():
    """Test if all required dependencies are available"""
    deps = ['torch', 'sentence_transformers', 'pandas', 'numpy']
    results = {}
    
    for dep in deps:
        try:
            __import__(dep)
            results[dep] = "✅ Available"
            logger.info(f"✅ {dep} is available")
        except ImportError:
            results[dep] = "❌ Missing"
            logger.error(f"❌ {dep} is missing")
    
    return results

def test_basic_reranking():
    """Test basic reranking functionality"""
    try:
        from scholarqa.rag.reranker.optimized_local_reranker import OptimizedLocalReranker
        
        # Create a test query and documents
        query = "What are transformer architectures?"
        docs = [
            "Transformers are neural network architectures that use attention mechanisms.",
            "Convolutional neural networks are used for image processing.",
            "The transformer architecture was introduced in the paper 'Attention is All You Need'.",
            "Recurrent neural networks process sequences of data.",
            "BERT is a transformer-based model for natural language understanding."
        ]
        
        logger.info("🔧 Initializing OptimizedLocalReranker...")
        start_time = time.time()
        
        reranker = OptimizedLocalReranker()
        init_time = time.time() - start_time
        logger.info(f"⚡ Reranker initialized in {init_time:.2f}s")
        
        logger.info("🔄 Running reranking test...")
        start_time = time.time()
        
        scores = reranker.rerank(query, docs)
        rerank_time = time.time() - start_time
        
        logger.info(f"🏎️ Reranking completed in {rerank_time:.3f}s")
        logger.info(f"📊 Scores: {scores}")
        
        # Verify we got reasonable results
        if len(scores) == len(docs) and all(isinstance(s, (int, float)) for s in scores):
            logger.info("✅ Reranking test passed!")
            return True, init_time, rerank_time
        else:
            logger.error("❌ Reranking test failed - invalid scores")
            return False, init_time, rerank_time
            
    except Exception as e:
        logger.error(f"❌ Reranking test failed: {e}")
        return False, 0, 0

def main():
    """Run all tests"""
    logger.info("🚀 Starting Quick Reranker Tests")
    logger.info("=" * 50)
    
    # Test 1: Dependencies
    logger.info("1. Testing Dependencies...")
    deps = test_dependencies()
    all_deps_ok = all("✅" in status for status in deps.values())
    
    # Test 2: Import
    logger.info("\n2. Testing Import...")
    import_ok = test_reranker_import()
    
    # Test 3: Basic functionality (only if import works)
    logger.info("\n3. Testing Basic Functionality...")
    if import_ok:
        rerank_ok, init_time, rerank_time = test_basic_reranking()
    else:
        rerank_ok, init_time, rerank_time = False, 0, 0
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📋 TEST SUMMARY")
    logger.info("=" * 50)
    
    for dep, status in deps.items():
        logger.info(f"  {dep}: {status}")
    
    logger.info(f"  Import: {'✅ Success' if import_ok else '❌ Failed'}")
    logger.info(f"  Reranking: {'✅ Success' if rerank_ok else '❌ Failed'}")
    
    if rerank_ok:
        logger.info(f"  Initialization Time: {init_time:.2f}s")
        logger.info(f"  Reranking Time: {rerank_time:.3f}s")
        logger.info("🎉 All tests passed! Optimized reranker is working properly.")
    else:
        logger.error("⚠️ Some tests failed. Check logs above for details.")
    
    return all_deps_ok and import_ok and rerank_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
