"""Total Recall: A Zero-Cost, Privacy-First Conversational Intelligence Engine."""

from __future__ import annotations

from whatsapp_total_recall.chunking import (
    DIRECT_BURST_GAP_SECONDS,
    GROUP_BURST_GAP_SECONDS,
    BurstChunker,
    chunk_into_episodes,
)
from whatsapp_total_recall.config import DEFAULT_CONFIG_PATH, AppConfig
from whatsapp_total_recall.dag import ReplyDAG
from whatsapp_total_recall.discovery import (
    WhatsAppContainers,
    discover_whatsapp_containers,
)
from whatsapp_total_recall.drafter import (
    AutonomousSendingProhibitedError,
    DraftReviewBox,
    IntentVerifier,
    ReplyDrafter,
    ToneAnalyzer,
    detect_unanswered_questions,
)
from whatsapp_total_recall.embeddings import (
    DeterministicMockEmbeddingProvider,
    EmbeddingProvider,
    OllamaEmbeddingProvider,
    ONNXEmbeddingProvider,
    get_embedding_provider,
)
from whatsapp_total_recall.ingestion import IngestionEngine
from whatsapp_total_recall.mcp import (
    DEFAULT_BRIDGE_TOKEN_PATH,
    MCPServer,
    MCPSSEServer,
    MCPStdioTransport,
    load_or_create_bridge_token,
    validate_loopback_host,
)
from whatsapp_total_recall.search import SearchIndex
from whatsapp_total_recall.summarizer import (
    DeterministicMockLLMAdapter,
    LLMAdapter,
    MiniTruePrefixesVerifier,
    OllamaLLMAdapter,
    SentimentNormalizer,
    Summarizer,
    format_citation,
    parse_citation,
)
from whatsapp_total_recall.types import (
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
from whatsapp_total_recall.ui import (
    UIServer,
    UIServerContext,
    build_native_menu_bar_app,
    run_ui_server,
)
from whatsapp_total_recall.vault import DiscoveredChat, IngestionReport, VaultConfig

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_BRIDGE_TOKEN_PATH",
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
    "MCPServer",
    "MCPSSEServer",
    "MCPStdioTransport",
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
    "UIServer",
    "UIServerContext",
    "UnansweredQuestion",
    "VaultConfig",
    "VerifiedClaim",
    "VerifiedSummary",
    "WhatsAppContainers",
    "build_native_menu_bar_app",
    "chunk_into_episodes",
    "detect_unanswered_questions",
    "discover_whatsapp_containers",
    "format_citation",
    "get_embedding_provider",
    "load_or_create_bridge_token",
    "parse_citation",
    "run_ui_server",
    "validate_loopback_host",
]
