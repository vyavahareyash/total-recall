"""Total Recall: A Zero-Cost, Privacy-First Conversational Intelligence Engine."""

from __future__ import annotations

from total_recall.chunking import (
    DIRECT_BURST_GAP_SECONDS,
    GROUP_BURST_GAP_SECONDS,
    BurstChunker,
    chunk_into_episodes,
)
from total_recall.config import DEFAULT_CONFIG_PATH, AppConfig
from total_recall.dag import ReplyDAG
from total_recall.discovery import WhatsAppContainers, discover_whatsapp_containers
from total_recall.drafter import (
    AutonomousSendingProhibitedError,
    DraftReviewBox,
    IntentVerifier,
    ReplyDrafter,
    ToneAnalyzer,
    detect_unanswered_questions,
)
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
    DraftStatus,
    IngestionStats,
    IntentVerificationResult,
    NormalizedMessage,
    PerspectiveSplit,
    PurgeReport,
    ScoredEpisode,
    SearchResult,
    StagedDraft,
    ToneFormality,
    ToneProfile,
    UnansweredQuestion,
    VerifiedClaim,
    VerifiedSummary,
)
from total_recall.vault import DiscoveredChat, IngestionReport, VaultConfig

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DIRECT_BURST_GAP_SECONDS",
    "GROUP_BURST_GAP_SECONDS",
    "AppConfig",
    "AutonomousSendingProhibitedError",
    "BurstChunker",
    "Citation",
    "ConversationalFlowEpisode",
    "CrossBurstQuoteContext",
    "DeterministicMockEmbeddingProvider",
    "DeterministicMockLLMAdapter",
    "DiscoveredChat",
    "DraftReviewBox",
    "DraftStatus",
    "EmbeddingProvider",
    "IngestionEngine",
    "IngestionReport",
    "IngestionStats",
    "IntentVerificationResult",
    "IntentVerifier",
    "LLMAdapter",
    "MiniTruePrefixesVerifier",
    "NormalizedMessage",
    "OllamaEmbeddingProvider",
    "OllamaLLMAdapter",
    "ONNXEmbeddingProvider",
    "PerspectiveSplit",
    "PurgeReport",
    "ReplyDAG",
    "ReplyDrafter",
    "ScoredEpisode",
    "SearchIndex",
    "SearchResult",
    "SentimentNormalizer",
    "StagedDraft",
    "Summarizer",
    "ToneAnalyzer",
    "ToneFormality",
    "ToneProfile",
    "UnansweredQuestion",
    "VaultConfig",
    "VerifiedClaim",
    "VerifiedSummary",
    "WhatsAppContainers",
    "chunk_into_episodes",
    "detect_unanswered_questions",
    "discover_whatsapp_containers",
    "format_citation",
    "get_embedding_provider",
    "parse_citation",
]
