"""
Optimized Local Reranker Implementation
Provides performance-tuned local reranking with memory management and caching.
"""

import logging
import os
import time
from typing import List, Dict, Any, Optional
import threading
from functools import lru_cache

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn.functional as F
    from sentence_transformers import CrossEncoder

    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("torch not found, optimized local rerankers will not work.")
    TORCH_AVAILABLE = False

from scholarqa.rag.reranker.reranker_base import AbstractReranker, RERANKER_MAPPING


class OptimizedCrossEncoderReranker(AbstractReranker):
    """
    Optimized Cross-Encoder Reranker with:
    - Automatic GPU/CPU detection and optimization
    - Model compilation for faster inference
    - Memory-efficient batch processing
    - Model caching and warm-up
    """

    def __init__(
        self,
        model_name_or_path: str = "mixedbread-ai/mxbai-rerank-large-v1",
        batch_size: int = None,
        max_length: int = 512,
        compile_model: bool = True,
        cache_size: int = 1000,
        warm_up: bool = True,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "torch and sentence-transformers are required for optimized reranker"
            )

        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.compile_model = compile_model
        self.cache_size = cache_size

        # Detect optimal device and configuration
        self.device = self._detect_optimal_device()
        self.batch_size = batch_size or self._get_optimal_batch_size()

        logger.info(
            f"🚀 Initializing optimized reranker with device: {self.device}, batch_size: {self.batch_size}"
        )

        # Initialize model
        self._init_model()

        # Warm up model
        if warm_up:
            self._warm_up_model()

    def _detect_optimal_device(self) -> str:
        """Detect the best available device and configure optimally"""
        if torch.cuda.is_available():
            device = "cuda"
            # Set optimal CUDA settings
            if hasattr(torch.backends.cudnn, "benchmark"):
                torch.backends.cudnn.benchmark = True
            if hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = True
            logger.info(f"🔥 Using GPU: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"  # Apple Silicon
            logger.info("🍎 Using Apple Silicon MPS")
        else:
            device = "cpu"
            # Optimize CPU usage
            torch.set_num_threads(os.cpu_count())
            logger.info(f"🖥️ Using CPU with {os.cpu_count()} threads")

        return device

    def _get_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on device"""
        if self.device == "cuda":
            # Get GPU memory and calculate optimal batch size
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory_gb >= 24:  # High-end GPU
                return 128
            elif gpu_memory_gb >= 12:  # Mid-range GPU
                return 64
            else:  # Lower-end GPU
                return 32
        elif self.device == "mps":
            return 64  # Conservative for Apple Silicon
        else:
            return 16  # CPU

    def _init_model(self):
        """Initialize the cross-encoder model with optimizations"""
        start_time = time.time()

        # Model initialization arguments
        model_args = {
            "automodel_args": {
                "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32
            },
            "trust_remote_code": True,
            "device": self.device,
        }

        # Add max_length if supported
        try:
            self.model = CrossEncoder(
                self.model_name_or_path, max_length=self.max_length, **model_args
            )
        except TypeError:
            # Fallback if max_length not supported
            self.model = CrossEncoder(self.model_name_or_path, **model_args)

        # Model compilation for faster inference (PyTorch 2.0+)
        if self.compile_model and hasattr(torch, "compile"):
            try:
                logger.info("🔧 Compiling model for faster inference...")
                self.model.model = torch.compile(
                    self.model.model, mode="reduce-overhead"
                )
                self.compiled = True
                logger.info("✅ Model compilation successful")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
                self.compiled = False
        else:
            self.compiled = False

        init_time = time.time() - start_time
        logger.info(f"⚡ Model initialized in {init_time:.2f}s")

    def _warm_up_model(self):
        """Warm up the model with dummy data for faster first inference"""
        logger.info("🔥 Warming up model...")
        start_time = time.time()

        # Create dummy data
        dummy_query = "This is a sample query for warming up the model"
        dummy_passages = [
            "This is a sample passage to warm up the reranking model",
            "Another sample passage for model initialization",
        ]

        # Run dummy inference
        try:
            self._score_batch(dummy_query, dummy_passages)
            warmup_time = time.time() - start_time
            logger.info(f"🏎️ Model warmed up in {warmup_time:.2f}s")
        except Exception as e:
            logger.warning(f"Model warm-up failed: {e}")

    @lru_cache(maxsize=1000)
    def _cached_score(self, query: str, passage: str) -> float:
        """Cache individual scores for repeated query-passage pairs"""
        # This is mainly for development/testing where same pairs might be scored multiple times
        return self._score_batch(query, [passage])[0]

    def _score_batch(self, query: str, passages: List[str]) -> List[float]:
        """Score a batch of passages efficiently"""
        sentence_pairs = [[query, passage] for passage in passages]

        try:
            scores = self.model.predict(
                sentence_pairs,
                convert_to_tensor=True,
                show_progress_bar=len(passages)
                > 50,  # Only show progress for large batches
                batch_size=self.batch_size,
            )

            # Convert to list of floats
            if hasattr(scores, "tolist"):
                return scores.tolist()
            else:
                return [float(s) for s in scores]

        except Exception as e:
            logger.error(f"Scoring failed: {e}")
            # Fallback to smaller batch size
            if self.batch_size > 1:
                logger.info(f"Retrying with smaller batch size: {self.batch_size // 2}")
                return self._score_with_fallback(query, passages)
            else:
                raise e

    def _score_with_fallback(self, query: str, passages: List[str]) -> List[float]:
        """Fallback scoring with reduced batch size"""
        fallback_batch_size = max(1, self.batch_size // 2)
        all_scores = []

        for i in range(0, len(passages), fallback_batch_size):
            batch_passages = passages[i : i + fallback_batch_size]
            sentence_pairs = [[query, passage] for passage in batch_passages]

            scores = self.model.predict(
                sentence_pairs,
                convert_to_tensor=True,
                show_progress_bar=False,
                batch_size=fallback_batch_size,
            )

            if hasattr(scores, "tolist"):
                all_scores.extend(scores.tolist())
            else:
                all_scores.extend([float(s) for s in scores])

        return all_scores

    def get_scores(self, query: str, documents: List[str]) -> List[float]:
        """Main scoring function with optimizations"""
        if not documents:
            return []

        start_time = time.time()

        # Process in batches if documents list is very large
        max_docs_per_call = 1000  # Prevent memory issues

        if len(documents) <= max_docs_per_call:
            scores = self._score_batch(query, documents)
        else:
            # Process in chunks
            logger.info(f"Processing {len(documents)} documents in chunks...")
            scores = []
            for i in range(0, len(documents), max_docs_per_call):
                chunk = documents[i : i + max_docs_per_call]
                chunk_scores = self._score_batch(query, chunk)
                scores.extend(chunk_scores)

        processing_time = time.time() - start_time
        logger.info(
            f"🎯 Scored {len(documents)} documents in {processing_time:.2f}s "
            f"({len(documents)/processing_time:.1f} docs/sec)"
        )

        return scores

    def get_tokenizer(self):
        """Get the tokenizer for debugging/analysis"""
        return self.model.tokenizer


class FastBiEncoderReranker(AbstractReranker):
    """
    Optimized Bi-Encoder for faster but potentially less accurate reranking
    Good for initial filtering of large document sets
    """

    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-base-en-v1.5",
        batch_size: int = None,
        normalize_embeddings: bool = True,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("torch and sentence-transformers are required")

        from sentence_transformers import SentenceTransformer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size or (256 if self.device == "cuda" else 64)
        self.normalize_embeddings = normalize_embeddings

        logger.info(f"🚀 Initializing fast bi-encoder on {self.device}")

        self.model = SentenceTransformer(model_name_or_path, device=self.device)

        # Enable optimizations
        if hasattr(self.model, "max_seq_length"):
            self.model.max_seq_length = 512

    def get_scores(self, query: str, documents: List[str]) -> List[float]:
        """Fast similarity scoring using bi-encoder"""
        start_time = time.time()

        # Encode query
        query_embedding = self.model.encode(
            [query],
            convert_to_tensor=True,
            batch_size=1,
            normalize_embeddings=self.normalize_embeddings,
        )[0]

        # Encode documents
        doc_embeddings = self.model.encode(
            documents,
            convert_to_tensor=True,
            batch_size=self.batch_size,
            show_progress_bar=len(documents) > 100,
            normalize_embeddings=self.normalize_embeddings,
        )

        # Calculate cosine similarities
        scores = (
            F.cosine_similarity(query_embedding.unsqueeze(0), doc_embeddings)
            .cpu()
            .numpy()
        )

        processing_time = time.time() - start_time
        logger.info(
            f"⚡ Fast-scored {len(documents)} documents in {processing_time:.2f}s"
        )

        return [float(s) for s in scores]


# Register optimized rerankers
RERANKER_MAPPING["optimized_crossencoder"] = OptimizedCrossEncoderReranker
RERANKER_MAPPING["fast_biencoder"] = FastBiEncoderReranker
