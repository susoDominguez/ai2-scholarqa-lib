#!/usr/bin/env python3
"""
Simple Reranker Test - Direct test of optimized reranker functionality
"""

import sys
import os
import time

# Add the current directory to Python path
sys.path.insert(0, "/Users/w1214757/Dev/ai2-scholarqa-lib/api")


def test_basic_reranker():
    """Test basic reranker functionality"""
    print(" Testing Basic Reranker Functionality")
    print("=" * 50)

    try:
        # Test data
        query = "What are machine learning advances?"
        documents = [
            "Machine learning has advanced significantly with deep learning.",
            "Natural language processing uses transformer models.",
            "Computer vision has improved with convolutional networks.",
            "Reinforcement learning enables autonomous agents.",
            "Transfer learning reduces training time.",
        ]

        print(f" Testing with {len(documents)} documents")
        print(f" Query: {query}")

        # Test 1: Standard Cross-Encoder
        print("\n Testing Standard Cross-Encoder...")
        try:
            from scholarqa.rag.reranker.reranker_base import CrossEncoderScores

            start_time = time.time()
            reranker = CrossEncoderScores(
                model_name_or_path="mixedbread-ai/mxbai-rerank-large-v1"
            )
            init_time = time.time() - start_time

            start_time = time.time()
            scores = reranker.get_scores(query, documents)
            score_time = time.time() - start_time

            print(f" Standard Cross-Encoder Success!")
            print(f"    Init: {init_time:.2f}s, Scoring: {score_time:.2f}s")
            print(f"    Scores: {[round(s, 3) for s in scores[:3]]}...")

        except Exception as e:
            print(f" Standard Cross-Encoder Failed: {e}")

        # Test 2: Optimized Cross-Encoder
        print("\n⚡ Testing Optimized Cross-Encoder...")
        try:
            from scholarqa.rag.reranker.optimized_local_reranker import (
                OptimizedCrossEncoderReranker,
            )

            start_time = time.time()
            reranker = OptimizedCrossEncoderReranker(
                model_name_or_path="mixedbread-ai/mxbai-rerank-large-v1",
                warm_up=True,
                compile_model=True,
            )
            init_time = time.time() - start_time

            start_time = time.time()
            scores = reranker.get_scores(query, documents)
            score_time = time.time() - start_time

            print(f" Optimized Cross-Encoder Success!")
            print(f"    Init: {init_time:.2f}s, Scoring: {score_time:.2f}s")
            print(f"    Scores: {[round(s, 3) for s in scores[:3]]}...")

        except Exception as e:
            print(f" Optimized Cross-Encoder Failed: {e}")

        # Test 3: Fast Bi-Encoder
        print("\n Testing Fast Bi-Encoder...")
        try:
            from scholarqa.rag.reranker.optimized_local_reranker import (
                FastBiEncoderReranker,
            )

            start_time = time.time()
            reranker = FastBiEncoderReranker(model_name_or_path="BAAI/bge-base-en-v1.5")
            init_time = time.time() - start_time

            start_time = time.time()
            scores = reranker.get_scores(query, documents)
            score_time = time.time() - start_time

            print(f" Fast Bi-Encoder Success!")
            print(f"    Init: {init_time:.2f}s, Scoring: {score_time:.2f}s")
            print(f"    Scores: {[round(s, 3) for s in scores[:3]]}...")

        except Exception as e:
            print(f" Fast Bi-Encoder Failed: {e}")

    except Exception as e:
        print(f" Overall test failed: {e}")


def test_device_detection():
    """Test device detection capabilities"""
    print("\n Testing Device Detection")
    print("=" * 30)

    try:
        import torch

        print(f" PyTorch version: {torch.__version__}")
        print(f" CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(
                f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print(" Apple Silicon MPS available")

        print(f" CPU cores: {os.cpu_count()}")

    except Exception as e:
        print(f" Device detection failed: {e}")


def main():
    """Main test function"""
    print(" Ai2 Scholar QA - Optimized Reranker Test")
    print("=" * 60)

    test_device_detection()
    test_basic_reranker()

    print("\n Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
