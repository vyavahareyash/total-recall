from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

from total_recall.types import (
    ConversationalFlowEpisode,
    CrossBurstQuoteContext,
    NormalizedMessage,
)

# Inactivity thresholds in seconds
GROUP_BURST_GAP_SECONDS = 900.0  # 15 minutes for @g.us group chats
DIRECT_BURST_GAP_SECONDS = 2700.0  # 45 minutes for 1:1 direct chats

MIN_EPISODE_MESSAGES = 5
MAX_EPISODE_MESSAGES = 25


def get_burst_threshold_seconds(chat_jid: str) -> float:
    """Return inactivity gap threshold for chat based on chat type."""
    if chat_jid.endswith("@g.us"):
        return GROUP_BURST_GAP_SECONDS
    return DIRECT_BURST_GAP_SECONDS


def truncate_text(text: str, max_length: int = 60) -> str:
    """Normalize whitespace and truncate text for header/quote snippets."""
    clean = " ".join(text.split())
    if len(clean) > max_length:
        return clean[: max_length - 3] + "..."
    return clean


class BurstChunker:
    """Segments continuous message streams into Conversational Flow Episodes."""

    def __init__(
        self,
        group_gap_seconds: float = GROUP_BURST_GAP_SECONDS,
        direct_gap_seconds: float = DIRECT_BURST_GAP_SECONDS,
        min_messages: int = MIN_EPISODE_MESSAGES,
        max_messages: int = MAX_EPISODE_MESSAGES,
    ):
        self.group_gap_seconds = group_gap_seconds
        self.direct_gap_seconds = direct_gap_seconds
        self.min_messages = min_messages
        self.max_messages = max_messages

    def _get_threshold(self, chat_jid: str) -> float:
        if chat_jid.endswith("@g.us"):
            return self.group_gap_seconds
        return self.direct_gap_seconds

    def _split_into_bursts(
        self, messages: list[NormalizedMessage], threshold_seconds: float
    ) -> list[list[NormalizedMessage]]:
        """Split chronologically ordered messages into bursts based on inactivity gaps."""
        if not messages:
            return []

        bursts: list[list[NormalizedMessage]] = []
        current_burst: list[NormalizedMessage] = [messages[0]]

        for msg in messages[1:]:
            prev_msg = current_burst[-1]
            gap = msg.unix_timestamp - prev_msg.unix_timestamp
            if gap > threshold_seconds:
                bursts.append(current_burst)
                current_burst = [msg]
            else:
                current_burst.append(msg)

        if current_burst:
            bursts.append(current_burst)

        return bursts

    def _subdivide_burst(
        self, burst: list[NormalizedMessage]
    ) -> list[list[NormalizedMessage]]:
        """Subdivide an oversized burst (> max_messages) into balanced chunks <= max_messages."""
        n = len(burst)
        if n <= self.max_messages:
            return [burst]

        # Calculate number of chunks needed
        num_chunks = math.ceil(n / self.max_messages)
        base_size = n // num_chunks
        remainder = n % num_chunks

        chunks: list[list[NormalizedMessage]] = []
        start_idx = 0
        for i in range(num_chunks):
            chunk_size = base_size + (1 if i < remainder else 0)
            end_idx = start_idx + chunk_size
            chunks.append(burst[start_idx:end_idx])
            start_idx = end_idx

        return chunks

    def chunk_messages(
        self,
        messages: Sequence[NormalizedMessage],
        chat_names: dict[str, str] | None = None,
    ) -> list[ConversationalFlowEpisode]:
        """Chunk a stream of normalized messages into Conversational Flow Episodes."""
        if not messages:
            return []

        chat_names = chat_names or {}
        # Global message lookup by PK for cross-burst quote resolution
        msg_lookup: dict[int, NormalizedMessage] = {m.pk: m for m in messages}

        # Partition messages by chat_jid, ensuring chronological ordering
        chats: dict[str, list[NormalizedMessage]] = defaultdict(list)
        for m in sorted(messages, key=lambda msg: (msg.unix_timestamp, msg.pk)):
            chats[m.chat_jid].append(m)

        episodes: list[ConversationalFlowEpisode] = []

        for chat_jid, chat_msgs in chats.items():
            threshold = self._get_threshold(chat_jid)
            bursts = self._split_into_bursts(chat_msgs, threshold)

            episode_chunks: list[list[NormalizedMessage]] = []
            for burst in bursts:
                episode_chunks.extend(self._subdivide_burst(burst))

            for chunk_idx, chunk in enumerate(episode_chunks, start=1):
                episode = self._assemble_episode(
                    chat_jid=chat_jid,
                    chat_name=chat_names.get(chat_jid, chat_jid),
                    messages=chunk,
                    chunk_idx=chunk_idx,
                    msg_lookup=msg_lookup,
                )
                episodes.append(episode)

        return episodes

    def _assemble_episode(
        self,
        chat_jid: str,
        chat_name: str,
        messages: list[NormalizedMessage],
        chunk_idx: int,
        msg_lookup: dict[int, NormalizedMessage],
    ) -> ConversationalFlowEpisode:
        """Construct a ConversationalFlowEpisode from a message chunk with quote stitching."""
        chunk_pks = {m.pk for m in messages}
        first_msg = messages[0]
        last_msg = messages[-1]

        # Identify unique participants in chronological order of first appearance
        participants_seen: dict[str, None] = {}
        for m in messages:
            participants_seen[m.sender_name] = None
        participants = tuple(participants_seen.keys())

        # Resolve cross-burst quotes
        cross_burst_quotes: list[CrossBurstQuoteContext] = []
        for m in messages:
            if m.parent_pk is not None and m.parent_pk not in chunk_pks:
                parent = msg_lookup.get(m.parent_pk)
                if parent:
                    cross_burst_quotes.append(
                        CrossBurstQuoteContext(
                            child_pk=m.pk,
                            parent_pk=parent.pk,
                            parent_sender_name=parent.sender_name,
                            parent_text=parent.text,
                            parent_iso_date=parent.iso_date,
                        )
                    )

        # Build Metadata Header
        header_parts = [
            f"[Chat: {chat_name} ({chat_jid})]",
            f"[Date: {first_msg.iso_date} - {last_msg.iso_date}]",
            f"[Participants: {', '.join(participants)}]",
        ]

        if cross_burst_quotes:
            quote_snippets = [
                f"In reply to msg_id {q.parent_pk} from {q.parent_sender_name}: '{truncate_text(q.parent_text)}'"
                for q in cross_burst_quotes
            ]
            header_parts.append(f"[Cross-Burst Quotes: {'; '.join(quote_snippets)}]")

        header = "Header: " + " | ".join(header_parts)

        # Build Formatted Content
        cb_map: dict[int, CrossBurstQuoteContext] = {
            q.child_pk: q for q in cross_burst_quotes
        }
        content_lines: list[str] = []

        for m in messages:
            if m.pk in cb_map:
                cb = cb_map[m.pk]
                snip = truncate_text(cb.parent_text)
                turn = (
                    f"[{m.iso_date}] {m.sender_name} (in reply to earlier msg_id {cb.parent_pk} "
                    f"from {cb.parent_sender_name}: '{snip}'): {m.text} (msg_id: {m.pk})"
                )
            elif m.parent_pk is not None and m.parent_pk in chunk_pks:
                turn = (
                    f"[{m.iso_date}] {m.sender_name} (reply to #{m.parent_pk}): "
                    f"{m.text} (msg_id: {m.pk})"
                )
            else:
                turn = f"[{m.iso_date}] {m.sender_name}: {m.text} (msg_id: {m.pk})"
            content_lines.append(turn)

        content = "\n".join(content_lines)
        episode_id = f"ep_{chat_jid}_{first_msg.pk}_{last_msg.pk}_{chunk_idx}"

        return ConversationalFlowEpisode(
            episode_id=episode_id,
            chat_jid=chat_jid,
            chat_name=chat_name,
            start_timestamp=first_msg.unix_timestamp,
            end_timestamp=last_msg.unix_timestamp,
            start_iso=first_msg.iso_date,
            end_iso=last_msg.iso_date,
            participants=participants,
            messages=tuple(messages),
            cross_burst_quotes=tuple(cross_burst_quotes),
            header=header,
            content=content,
        )


def chunk_into_episodes(
    messages: Sequence[NormalizedMessage],
    chat_names: dict[str, str] | None = None,
    group_gap_seconds: float = GROUP_BURST_GAP_SECONDS,
    direct_gap_seconds: float = DIRECT_BURST_GAP_SECONDS,
    min_messages: int = MIN_EPISODE_MESSAGES,
    max_messages: int = MAX_EPISODE_MESSAGES,
) -> list[ConversationalFlowEpisode]:
    """Convenience helper to chunk normalized messages into Conversational Flow Episodes."""
    chunker = BurstChunker(
        group_gap_seconds=group_gap_seconds,
        direct_gap_seconds=direct_gap_seconds,
        min_messages=min_messages,
        max_messages=max_messages,
    )
    return chunker.chunk_messages(messages, chat_names=chat_names)
