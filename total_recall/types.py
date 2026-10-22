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


@dataclass(frozen=True)
class SearchResult:
    message_pk: int
    chat_jid: str
    sender_name: str
    iso_date: str
    text: str
    bm25_score: float


@dataclass(frozen=True)
class IngestionStats:
    total_sessions_scanned: int
    direct_sessions_ingested: int
    messages_ingested: int
