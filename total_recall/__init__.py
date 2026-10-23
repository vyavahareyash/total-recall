"""Total Recall: A Zero-Cost, Privacy-First Conversational Intelligence Engine."""

from __future__ import annotations

from total_recall.dag import ReplyDAG
from total_recall.discovery import WhatsAppContainers, discover_whatsapp_containers
from total_recall.ingestion import IngestionEngine
from total_recall.search import SearchIndex
from total_recall.types import IngestionStats, NormalizedMessage, SearchResult

__all__ = [
    "IngestionEngine",
    "IngestionStats",
    "NormalizedMessage",
    "ReplyDAG",
    "SearchIndex",
    "SearchResult",
    "WhatsAppContainers",
    "discover_whatsapp_containers",
]
