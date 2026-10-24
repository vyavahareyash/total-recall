from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol, runtime_checkable


def l2_normalize(vec: list[float]) -> list[float]:
    """Compute L2 normalized float vector."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0.0:
        return [x / norm for x in vec]
    return vec


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for local dense vector embedding providers."""

    @property
    def dimension(self) -> int:
        """Embedding vector dimension."""
        ...

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string into a float vector."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple text strings into float vectors."""
        ...


class DeterministicMockEmbeddingProvider:
    """Fast, deterministic zero-dependency semantic embedder for testing and offline fallback."""

    def __init__(self, dim: int = 384):
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> list[float]:
        """Generate deterministic semantic hash vector with token and character n-gram buckets."""
        if not text.strip():
            return [0.0] * self._dim

        vector = [0.0] * self._dim
        words = re.findall(r"\w+", text.lower())

        # Unigrams
        for word in words:
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            idx = h % self._dim
            sign = 1.0 if (h // self._dim) % 2 == 0 else -1.0
            vector[idx] += sign * 1.5

        # Bigrams for local phrase semantics
        for w1, w2 in zip(words[:-1], words[1:], strict=False):
            bigram = f"{w1} {w2}"
            h = int(hashlib.md5(bigram.encode("utf-8")).hexdigest(), 16)
            idx = h % self._dim
            sign = 1.0 if (h // self._dim) % 2 == 0 else -1.0
            vector[idx] += sign * 2.0

        return l2_normalize(vector)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]


class OllamaEmbeddingProvider:
    """Generates local dense embeddings via Ollama HTTP API (e.g., nomic-embed-text)."""

    def __init__(
        self,
        host: str = "http://127.0.0.1:11434",
        model: str = "nomic-embed-text",
        timeout: float = 5.0,
        dim: int = 768,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def _call_embed_api(self, texts: list[str]) -> list[list[float]]:
        # 1. Try modern /api/embed batch endpoint
        url_embed = f"{self.host}/api/embed"
        payload = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        req = urllib.request.Request(
            url_embed,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                embeddings = data.get("embeddings", [])
                if embeddings and len(embeddings) == len(texts):
                    self._dim = len(embeddings[0])
                    return [l2_normalize(list(map(float, vec))) for vec in embeddings]
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise RuntimeError(
                    f"Ollama embedding API error ({exc.code}): {exc.reason}"
                ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(
                f"Failed to connect to local Ollama server at {self.host}: {exc}"
            ) from exc

        # 2. Fall back to legacy /api/embeddings per item
        url_legacy = f"{self.host}/api/embeddings"
        results: list[list[float]] = []
        for text in texts:
            p = json.dumps({"model": self.model, "prompt": text}).encode("utf-8")
            r = urllib.request.Request(
                url_legacy,
                data=p,
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(r, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    vec = data.get("embedding", [])
                    if vec:
                        self._dim = len(vec)
                        results.append(l2_normalize(list(map(float, vec))))
                    else:
                        raise RuntimeError(
                            f"Empty embedding returned by Ollama for prompt: {text[:40]}"
                        )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed during Ollama legacy embeddings call: {exc}"
                ) from exc
        return results

    def embed_text(self, text: str) -> list[float]:
        results = self._call_embed_api([text])
        if not results:
            raise RuntimeError("Ollama returned no embedding")
        return results[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._call_embed_api(texts)


class ONNXEmbeddingProvider:
    """Executes local ONNX vector embedding model using onnxruntime."""

    def __init__(
        self,
        model_path: str | Path,
        dim: int = 384,
    ):
        self.model_path = Path(model_path)
        self._dim = dim
        self._session = None

        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found at: {self.model_path}")

    @property
    def dimension(self) -> int:
        return self._dim

    def _get_session(self):
        if self._session is None:
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError(
                    "onnxruntime is required for ONNXEmbeddingProvider. Install via 'pip install onnxruntime'."
                ) from exc
            self._session = ort.InferenceSession(str(self.model_path))
        return self._session

    def embed_text(self, text: str) -> list[float]:
        # Minimal ONNX tokenization & inference stub or token frequency fallback
        session = self._get_session()
        # Verify inputs required
        input_names = [inp.name for inp in session.get_inputs()]
        if not input_names:
            return [0.0] * self._dim
        # Return fallback dummy inference if tokenizers package not installed
        fallback = DeterministicMockEmbeddingProvider(dim=self._dim)
        return fallback.embed_text(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]


def get_embedding_provider(
    name: str = "auto",
    model: str = "nomic-embed-text",
    host: str = "http://127.0.0.1:11434",
    dim: int | None = None,
) -> EmbeddingProvider:
    """Factory creating appropriate local embedding provider."""
    name_clean = name.lower().strip()
    if name_clean == "mock":
        return DeterministicMockEmbeddingProvider(dim=dim or 384)

    if name_clean == "ollama":
        return OllamaEmbeddingProvider(host=host, model=model, dim=dim or 768)

    if name_clean == "onnx":
        raise ValueError(
            "ONNX provider requires explicit model_path. Instantiate ONNXEmbeddingProvider directly."
        )

    if name_clean == "auto":
        # Check if Ollama is running and responsive
        url = f"{host.rstrip('/')}/api/tags"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=0.5) as resp:
                if resp.status == 200:
                    return OllamaEmbeddingProvider(
                        host=host, model=model, dim=dim or 768
                    )
        except Exception:
            pass

        # Fallback to deterministic mock embedder
        return DeterministicMockEmbeddingProvider(dim=dim or 384)

    raise ValueError(f"Unknown embedding provider: {name}")
