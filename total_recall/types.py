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
    expire_timer: int | None = None
    expires_at: float | None = None

    def __post_init__(self) -> None:
        if (
            self.expire_timer is not None
            and self.expire_timer > 0
            and self.expires_at is None
        ):
            object.__setattr__(
                self, "expires_at", self.unix_timestamp + float(self.expire_timer)
            )


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
    min_expires_at: float | None = None


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
    stanza_id: str | None = None


@dataclass(frozen=True)
class IngestionStats:
    total_sessions_scanned: int
    direct_sessions_ingested: int
    group_sessions_ingested: int = 0
    messages_ingested: int = 0
    episodes_generated: int = 0
    revocations_detected: int = 0
    expired_purged: int = 0


@dataclass(frozen=True)
class Citation:
    chat_name: str
    iso_date: str
    sender_name: str
    message_pk: int
    raw: str = ""

    def __str__(self) -> str:
        return (
            self.raw
            or f"[{self.chat_name} | {self.iso_date} | {self.sender_name} | {self.message_pk}]"
        )


@dataclass(frozen=True)
class VerifiedClaim:
    text: str
    citation: Citation | None
    entailed: bool
    entailment_score: float = 1.0
    discard_reason: str | None = None


@dataclass(frozen=True)
class PerspectiveSplit:
    topic: str
    perspectives: tuple[tuple[str, tuple[VerifiedClaim, ...]], ...]


@dataclass(frozen=True)
class VerifiedSummary:
    chat_jid: str
    title: str
    claims: tuple[VerifiedClaim, ...]
    discarded_claims: tuple[VerifiedClaim, ...]
    splits: tuple[PerspectiveSplit, ...]
    summary_text: str
    episodes_analyzed: int
    messages_analyzed: int
