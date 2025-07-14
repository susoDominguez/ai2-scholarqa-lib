#!/usr/bin/env python3
"""
Reranker Performance Test Script

This script tests different reranker configurations and provides performance benchmarks.
"""

import time
import json
import logging
from typing import List, Dict

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Test data
TEST_QUERY = (
    "What are the latest advances in machine learning for natural language processing?"
)

TEST_DOCUMENTS = [
    "Recent advances in transformer architectures have significantly improved natural language understanding tasks.",
    "Deep learning models like BERT and GPT have revolutionized how we approach text processing and generation.",
    "Attention mechanisms have become fundamental building blocks in modern neural network architectures.",
    "Pre-trained language models have shown remarkable performance across various NLP benchmarks.",
    "Fine-tuning large language models on specific tasks has become a standard practice in machine learning.",
    "The emergence of few-shot learning capabilities in large language models has opened new research directions.",
    "Multimodal models that combine text and image understanding are gaining increasing attention.",
    "Efficient training techniques like gradient accumulation and mixed precision have made large model training feasible.",
    "The development of more efficient architectures like MobileBERT has enabled deployment on edge devices.",
    "Recent work on interpretability has helped understand what language models learn during training.",
] * 5  # 50 documents total


def test_reranker_performance(reranker_class, reranker_args: Dict, name: str) -> Dict:
    """Test a specific reranker configuration"""
    logger.info(f"\n🧪 Testing {name}")
    logger.info("=" * 50)

    try:
        # Initialize reranker
        start_time = time.time()
        reranker = reranker_class(**reranker_args)
        init_time = time.time() - start_time

        # Test scoring
        start_time = time.time()
        scores = reranker.get_scores(TEST_QUERY, TEST_DOCUMENTS)
        scoring_time = time.time() - start_time

        # Calculate metrics
        docs_per_second = len(TEST_DOCUMENTS) / scoring_time if scoring_time > 0 else 0
        avg_score = sum(scores) / len(scores) if scores else 0
        max_score = max(scores) if scores else 0
        min_score = min(scores) if scores else 0

        results = {
            "name": name,
            "success": True,
            "init_time": round(init_time, 3),
            "scoring_time": round(scoring_time, 3),
            "docs_per_second": round(docs_per_second, 1),
            "num_documents": len(TEST_DOCUMENTS),
            "avg_score": round(avg_score, 4),
            "max_score": round(max_score, 4),
            "min_score": round(min_score, 4),
            "score_range": round(max_score - min_score, 4),
        }

        logger.info(f"✅ {name} - Success!")
        logger.info(f"   📊 Init time: {init_time:.3f}s")
        logger.info(f"   ⚡ Scoring time: {scoring_time:.3f}s")
        logger.info(f"   🎯 Speed: {docs_per_second:.1f} docs/sec")
        logger.info(f"   📈 Score range: {min_score:.4f} to {max_score:.4f}")

        return results

    except Exception as e:
        logger.error(f"❌ {name} - Failed: {str(e)}")
        return {
            "name": name,
            "success": False,
            "error": str(e),
            "init_time": 0,
            "scoring_time": 0,
            "docs_per_second": 0,
        }


def main():
    """Run performance tests for all available rerankers"""
    logger.info("🚀 Starting Reranker Performance Tests")
    logger.info(f"📄 Testing with {len(TEST_DOCUMENTS)} documents")

    results = []

    # Test configurations
    test_configs = [
        {
            "name": "Optimized Cross-Encoder",
            "import_path": "scholarqa.rag.reranker.optimized_local_reranker",
            "class_name": "OptimizedCrossEncoderReranker",
            "args": {
                "model_name_or_path": "mixedbread-ai/mxbai-rerank-large-v1",
                "warm_up": True,
                "compile_model": True,
            },
        },
        {
            "name": "Fast Bi-Encoder",
            "import_path": "scholarqa.rag.reranker.optimized_local_reranker",
            "class_name": "FastBiEncoderReranker",
            "args": {"model_name_or_path": "BAAI/bge-base-en-v1.5"},
        },
        {
            "name": "Standard Cross-Encoder",
            "import_path": "scholarqa.rag.reranker.reranker_base",
            "class_name": "CrossEncoderScores",
            "args": {"model_name_or_path": "mixedbread-ai/mxbai-rerank-large-v1"},
        },
    ]

    for config in test_configs:
        try:
            # Dynamic import
            module = __import__(config["import_path"], fromlist=[config["class_name"]])
            reranker_class = getattr(module, config["class_name"])

            # Test the reranker
            result = test_reranker_performance(
                reranker_class, config["args"], config["name"]
            )
            results.append(result)

        except ImportError as e:
            logger.warning(f"⚠️ Could not import {config['name']}: {e}")
            results.append(
                {
                    "name": config["name"],
                    "success": False,
                    "error": f"Import error: {e}",
                }
            )
        except Exception as e:
            logger.error(f"❌ Unexpected error testing {config['name']}: {e}")
            results.append(
                {
                    "name": config["name"],
                    "success": False,
                    "error": f"Unexpected error: {e}",
                }
            )

    # Summary
    logger.info("\n📊 PERFORMANCE SUMMARY")
    logger.info("=" * 60)

    successful_results = [r for r in results if r.get("success", False)]

    if successful_results:
        # Sort by speed
        successful_results.sort(key=lambda x: x.get("docs_per_second", 0), reverse=True)

        logger.info(
            f"{'Name':<25} {'Speed (docs/s)':<15} {'Init (s)':<10} {'Score (s)':<10}"
        )
        logger.info("-" * 60)

        for result in successful_results:
            logger.info(
                f"{result['name']:<25} "
                f"{result['docs_per_second']:<15.1f} "
                f"{result['init_time']:<10.3f} "
                f"{result['scoring_time']:<10.3f}"
            )

        # Recommendations
        logger.info("\n💡 RECOMMENDATIONS")
        logger.info("-" * 30)

        fastest = successful_results[0]
        logger.info(
            f"🏃 Fastest: {fastest['name']} ({fastest['docs_per_second']:.1f} docs/s)"
        )

        if len(successful_results) > 1:
            most_accurate = max(
                successful_results, key=lambda x: x.get("score_range", 0)
            )
            logger.info(
                f"🎯 Most Accurate: {most_accurate['name']} (score range: {most_accurate['score_range']:.4f})"
            )

    else:
        logger.error("❌ No rerankers were successfully tested!")

    # Save results
    with open("reranker_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n💾 Results saved to reranker_benchmark_results.json")

    return results


if __name__ == "__main__":
    main()
