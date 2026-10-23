from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizedMessage:
    pk: int
    chat_session_id: int
    chat_jid: str
    stanza_id: str
    is_from_me: bool
    sender_name: str
    sender_jid: str
    apple_date: float
    unix_timestamp: float
    iso_date: str
    text: str
    parent_pk: int | None = None


@dataclass(frozen=True)
class SearchResult:
    message_pk: int
    chat_jid: str
    sender_name: str
    iso_date: str
    text: str
    bm25_score: float


@dataclass(frozen=True)
class CrossBurstQuoteContext:
    child_pk: int
    parent_pk: int
    parent_sender_name: str
    parent_text: str
    parent_iso_date: str


@dataclass(frozen=True)
class ConversationalFlowEpisode:
    episode_id: str
    chat_jid: str
    chat_name: str
    start_timestamp: float
    end_timestamp: float
    start_iso: str
    end_iso: str
    participants: tuple[str, ...]
    messages: tuple[NormalizedMessage, ...]
    cross_burst_quotes: tuple[CrossBurstQuoteContext, ...]
    header: str
    content: str


@dataclass(frozen=True)
class ScoredEpisode:
    episode: ConversationalFlowEpisode
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    fts_rank: int | None = None
    vec_rank: int | None = None
    vector_distance: float | None = None


@dataclass(frozen=True)
class PurgeReport:
    message_pk: int
    episodes_updated: int
    episodes_deleted: int
    raw_messages_purged: int
    secure_delete_executed: bool


@dataclass(frozen=True)
class IngestionStats:
    total_sessions_scanned: int
    direct_sessions_ingested: int
    group_sessions_ingested: int = 0
    messages_ingested: int = 0
    episodes_generated: int = 0
