"""Total Recall: A Zero-Cost, Privacy-First Conversational Intelligence Engine."""

from __future__ import annotations

from total_recall.chunking import (
    DIRECT_BURST_GAP_SECONDS,
    GROUP_BURST_GAP_SECONDS,
    BurstChunker,
    chunk_into_episodes,
)
from total_recall.dag import ReplyDAG
from total_recall.discovery import WhatsAppContainers, discover_whatsapp_containers
from total_recall.embeddings import (
    DeterministicMockEmbeddingProvider,
    EmbeddingProvider,
    OllamaEmbeddingProvider,
    ONNXEmbeddingProvider,
    get_embedding_provider,
)
from total_recall.ingestion import IngestionEngine
from total_recall.search import SearchIndex
from total_recall.summarizer import (
    DeterministicMockLLMAdapter,
    LLMAdapter,
    MiniTruePrefixesVerifier,
    OllamaLLMAdapter,
    SentimentNormalizer,
    Summarizer,
    format_citation,
    parse_citation,
)
from total_recall.types import (
    Citation,
    ConversationalFlowEpisode,
    CrossBurstQuoteContext,
    IngestionStats,
    NormalizedMessage,
    PerspectiveSplit,
    PurgeReport,
    ScoredEpisode,
    SearchResult,
    VerifiedClaim,
    VerifiedSummary,
)
from total_recall.vault import DiscoveredChat, IngestionReport, VaultConfig

__all__ = [
    "DIRECT_BURST_GAP_SECONDS",
    "GROUP_BURST_GAP_SECONDS",
    "BurstChunker",
    "Citation",
    "ConversationalFlowEpisode",
    "CrossBurstQuoteContext",
    "DeterministicMockEmbeddingProvider",
    "DeterministicMockLLMAdapter",
    "DiscoveredChat",
    "EmbeddingProvider",
    "IngestionEngine",
    "IngestionReport",
    "IngestionStats",
    "LLMAdapter",
    "MiniTruePrefixesVerifier",
    "NormalizedMessage",
    "OllamaEmbeddingProvider",
    "OllamaLLMAdapter",
    "ONNXEmbeddingProvider",
    "PerspectiveSplit",
    "PurgeReport",
    "ReplyDAG",
    "ScoredEpisode",
    "SearchIndex",
    "SearchResult",
    "SentimentNormalizer",
    "Summarizer",
    "VaultConfig",
    "VerifiedClaim",
    "VerifiedSummary",
    "WhatsAppContainers",
    "chunk_into_episodes",
    "discover_whatsapp_containers",
    "format_citation",
    "get_embedding_provider",
    "parse_citation",
]
