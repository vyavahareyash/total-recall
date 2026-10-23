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
from total_recall.ingestion import IngestionEngine
from total_recall.search import SearchIndex
from total_recall.types import (
    ConversationalFlowEpisode,
    CrossBurstQuoteContext,
    IngestionStats,
    NormalizedMessage,
    ScoredEpisode,
    SearchResult,
)
from total_recall.vault import DiscoveredChat, IngestionReport, VaultConfig

__all__ = [
    "DIRECT_BURST_GAP_SECONDS",
    "GROUP_BURST_GAP_SECONDS",
    "BurstChunker",
    "ConversationalFlowEpisode",
    "CrossBurstQuoteContext",
    "DiscoveredChat",
    "IngestionEngine",
    "IngestionReport",
    "IngestionStats",
    "NormalizedMessage",
    "ReplyDAG",
    "ScoredEpisode",
    "SearchIndex",
    "SearchResult",
    "VaultConfig",
    "WhatsAppContainers",
    "chunk_into_episodes",
    "discover_whatsapp_containers",
]
