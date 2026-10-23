from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from total_recall.types import (
    ConversationalFlowEpisode,
    CrossBurstQuoteContext,
    NormalizedMessage,
    ScoredEpisode,
    SearchResult,
)


class SearchIndex:
    """Manages SQLite FTS5 virtual tables and BM25 lexical retrieval for messages and episodes."""

    def __init__(self, db_path: Path | str = ":memory:"):
        self.db_path = str(db_path) if isinstance(db_path, Path) else db_path
        self.con = sqlite3.connect(self.db_path)
        self.con.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        """Create normalized messages and episodes tables along with FTS5 virtual tables."""
        cur = self.con.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
                pk INTEGER PRIMARY KEY,
                chat_session_id INTEGER,
                chat_jid TEXT,
                stanza_id TEXT,
                is_from_me INTEGER,
                sender_name TEXT,
                sender_jid TEXT,
                apple_date REAL,
                unix_timestamp REAL,
                iso_date TEXT,
                text TEXT,
                parent_pk INTEGER
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                pk UNINDEXED,
                chat_jid,
                sender_name,
                text,
                tokenize = 'porter unicode61'
            );

            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY,
                chat_jid TEXT,
                chat_name TEXT,
                start_timestamp REAL,
                end_timestamp REAL,
                start_iso TEXT,
                end_iso TEXT,
                header TEXT,
                content TEXT,
                message_count INTEGER,
                messages_json TEXT,
                cross_burst_quotes_json TEXT
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
                episode_id UNINDEXED,
                chat_jid,
                chat_name,
                header,
                content,
                tokenize = 'porter unicode61'
            );
            """
        )

        # Migrate existing schema if parent_pk column is missing
        cur.execute("PRAGMA table_info(messages)")
        columns = [row["name"] for row in cur.fetchall()]
        if "parent_pk" not in columns:
            cur.execute("ALTER TABLE messages ADD COLUMN parent_pk INTEGER")

        self.con.commit()

    def index_messages(self, messages: list[NormalizedMessage]) -> int:
        """Batch index normalized messages into both storage and FTS5 tables."""
        if not messages:
            return 0

        cur = self.con.cursor()
        for msg in messages:
            cur.execute(
                """
                INSERT OR REPLACE INTO messages (
                    pk, chat_session_id, chat_jid, stanza_id, is_from_me,
                    sender_name, sender_jid, apple_date, unix_timestamp,
                    iso_date, text, parent_pk
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    msg.pk,
                    msg.chat_session_id,
                    msg.chat_jid,
                    msg.stanza_id,
                    1 if msg.is_from_me else 0,
                    msg.sender_name,
                    msg.sender_jid,
                    msg.apple_date,
                    msg.unix_timestamp,
                    msg.iso_date,
                    msg.text,
                    msg.parent_pk,
                ),
            )

            cur.execute("DELETE FROM messages_fts WHERE pk = ?", (str(msg.pk),))
            cur.execute(
                """
                INSERT INTO messages_fts (pk, chat_jid, sender_name, text)
                VALUES (?, ?, ?, ?)
                """,
                (str(msg.pk), msg.chat_jid, msg.sender_name, msg.text),
            )

        self.con.commit()
        return len(messages)

    def index_episodes(
        self,
        episodes: list[ConversationalFlowEpisode],
        index_constituent_messages: bool = True,
    ) -> int:
        """Batch index Conversational Flow Episodes and their constituent messages."""
        if not episodes:
            return 0

        if index_constituent_messages:
            all_msgs: dict[int, NormalizedMessage] = {}
            for ep in episodes:
                for m in ep.messages:
                    all_msgs[m.pk] = m
            self.index_messages(list(all_msgs.values()))

        cur = self.con.cursor()
        for ep in episodes:
            msgs_payload = [
                {
                    "pk": m.pk,
                    "chat_session_id": m.chat_session_id,
                    "chat_jid": m.chat_jid,
                    "stanza_id": m.stanza_id,
                    "is_from_me": m.is_from_me,
                    "sender_name": m.sender_name,
                    "sender_jid": m.sender_jid,
                    "apple_date": m.apple_date,
                    "unix_timestamp": m.unix_timestamp,
                    "iso_date": m.iso_date,
                    "text": m.text,
                    "parent_pk": m.parent_pk,
                }
                for m in ep.messages
            ]

            quotes_payload = [
                {
                    "child_pk": q.child_pk,
                    "parent_pk": q.parent_pk,
                    "parent_sender_name": q.parent_sender_name,
                    "parent_text": q.parent_text,
                    "parent_iso_date": q.parent_iso_date,
                }
                for q in ep.cross_burst_quotes
            ]

            cur.execute(
                """
                INSERT OR REPLACE INTO episodes (
                    episode_id, chat_jid, chat_name, start_timestamp,
                    end_timestamp, start_iso, end_iso, header, content,
                    message_count, messages_json, cross_burst_quotes_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ep.episode_id,
                    ep.chat_jid,
                    ep.chat_name,
                    ep.start_timestamp,
                    ep.end_timestamp,
                    ep.start_iso,
                    ep.end_iso,
                    ep.header,
                    ep.content,
                    len(ep.messages),
                    json.dumps(msgs_payload),
                    json.dumps(quotes_payload),
                ),
            )

            cur.execute(
                "DELETE FROM episodes_fts WHERE episode_id = ?", (ep.episode_id,)
            )
            cur.execute(
                """
                INSERT INTO episodes_fts (episode_id, chat_jid, chat_name, header, content)
                VALUES (?, ?, ?, ?, ?)
                """,
                (ep.episode_id, ep.chat_jid, ep.chat_name, ep.header, ep.content),
            )

        self.con.commit()
        return len(episodes)

    @staticmethod
    def sanitize_fts_query(query: str) -> str:
        """Sanitize raw user query into safe FTS5 query expression."""
        tokens = re.findall(r"\w+", query)
        if not tokens:
            return ""
        return " OR ".join(f'"{token}"' for token in tokens)

    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Perform BM25 ranked lexical search over indexed messages."""
        fts_query = self.sanitize_fts_query(query)
        if not fts_query:
            return []

        cur = self.con.cursor()
        cur.execute(
            """
            SELECT
                m.pk,
                m.chat_jid,
                m.sender_name,
                m.iso_date,
                m.text,
                bm25(messages_fts) AS bm25_score
            FROM messages_fts f
            JOIN messages m ON m.pk = CAST(f.pk AS INTEGER)
            WHERE messages_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (fts_query, limit),
        )

        results: list[SearchResult] = []
        for row in cur.fetchall():
            results.append(
                SearchResult(
                    message_pk=int(row["pk"]),
                    chat_jid=str(row["chat_jid"]),
                    sender_name=str(row["sender_name"]),
                    iso_date=str(row["iso_date"]),
                    text=str(row["text"]),
                    bm25_score=float(row["bm25_score"]),
                )
            )
        return results

    def _row_to_episode(self, row: sqlite3.Row) -> ConversationalFlowEpisode:
        raw_msgs = json.loads(row["messages_json"] or "[]")
        msgs = tuple(
            NormalizedMessage(
                pk=int(d["pk"]),
                chat_session_id=int(d["chat_session_id"]),
                chat_jid=str(d["chat_jid"]),
                stanza_id=str(d["stanza_id"]),
                is_from_me=bool(d["is_from_me"]),
                sender_name=str(d["sender_name"]),
                sender_jid=str(d["sender_jid"]),
                apple_date=float(d["apple_date"]),
                unix_timestamp=float(d["unix_timestamp"]),
                iso_date=str(d["iso_date"]),
                text=str(d["text"]),
                parent_pk=int(d["parent_pk"])
                if d.get("parent_pk") is not None
                else None,
            )
            for d in raw_msgs
        )

        raw_quotes = json.loads(row["cross_burst_quotes_json"] or "[]")
        quotes = tuple(
            CrossBurstQuoteContext(
                child_pk=int(q["child_pk"]),
                parent_pk=int(q["parent_pk"]),
                parent_sender_name=str(q["parent_sender_name"]),
                parent_text=str(q["parent_text"]),
                parent_iso_date=str(q["parent_iso_date"]),
            )
            for q in raw_quotes
        )

        seen_participants: dict[str, None] = {}
        for m in msgs:
            seen_participants[m.sender_name] = None

        return ConversationalFlowEpisode(
            episode_id=str(row["episode_id"]),
            chat_jid=str(row["chat_jid"]),
            chat_name=str(row["chat_name"]),
            start_timestamp=float(row["start_timestamp"]),
            end_timestamp=float(row["end_timestamp"]),
            start_iso=str(row["start_iso"]),
            end_iso=str(row["end_iso"]),
            participants=tuple(seen_participants.keys()),
            messages=msgs,
            cross_burst_quotes=quotes,
            header=str(row["header"]),
            content=str(row["content"]),
        )

    def search_episodes(self, query: str, limit: int = 10) -> list[ScoredEpisode]:
        """Perform BM25 ranked lexical search over indexed Conversational Flow Episodes."""
        fts_query = self.sanitize_fts_query(query)
        if not fts_query:
            return []

        cur = self.con.cursor()
        cur.execute(
            """
            SELECT
                e.*,
                bm25(episodes_fts) AS bm25_score
            FROM episodes_fts f
            JOIN episodes e ON e.episode_id = f.episode_id
            WHERE episodes_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (fts_query, limit),
        )

        results: list[ScoredEpisode] = []
        for row in cur.fetchall():
            episode = self._row_to_episode(row)
            results.append(
                ScoredEpisode(
                    episode=episode,
                    bm25_score=float(row["bm25_score"]),
                )
            )
        return results

    def get_episode(self, episode_id: str) -> ConversationalFlowEpisode | None:
        """Fetch a single episode by episode ID."""
        cur = self.con.cursor()
        cur.execute("SELECT * FROM episodes WHERE episode_id = ?", (episode_id,))
        row = cur.fetchone()
        return self._row_to_episode(row) if row else None

    def get_episodes_for_chat(self, chat_jid: str) -> list[ConversationalFlowEpisode]:
        """Fetch all episodes for a specific chat ordered chronologically."""
        cur = self.con.cursor()
        cur.execute(
            "SELECT * FROM episodes WHERE chat_jid = ? ORDER BY start_timestamp ASC",
            (chat_jid,),
        )
        return [self._row_to_episode(row) for row in cur.fetchall()]

    def get_all_episodes(self) -> list[ConversationalFlowEpisode]:
        """Fetch all indexed episodes ordered chronologically."""
        cur = self.con.cursor()
        cur.execute("SELECT * FROM episodes ORDER BY start_timestamp ASC")
        return [self._row_to_episode(row) for row in cur.fetchall()]

    @staticmethod
    def _row_to_normalized_message(row: sqlite3.Row) -> NormalizedMessage:
        return NormalizedMessage(
            pk=int(row["pk"]),
            chat_session_id=int(row["chat_session_id"]),
            chat_jid=str(row["chat_jid"]),
            stanza_id=str(row["stanza_id"] or ""),
            is_from_me=bool(row["is_from_me"]),
            sender_name=str(row["sender_name"]),
            sender_jid=str(row["sender_jid"]),
            apple_date=float(row["apple_date"]),
            unix_timestamp=float(row["unix_timestamp"]),
            iso_date=str(row["iso_date"]),
            text=str(row["text"]),
            parent_pk=int(row["parent_pk"]) if row["parent_pk"] is not None else None,
        )

    def get_message(self, pk: int) -> NormalizedMessage | None:
        """Fetch a single message by PK."""
        cur = self.con.cursor()
        cur.execute("SELECT * FROM messages WHERE pk = ?", (pk,))
        row = cur.fetchone()
        return self._row_to_normalized_message(row) if row else None

    def get_messages_for_chat(self, chat_jid: str) -> list[NormalizedMessage]:
        """Fetch all messages for a specific chat JID ordered chronologically."""
        cur = self.con.cursor()
        cur.execute(
            "SELECT * FROM messages WHERE chat_jid = ? ORDER BY unix_timestamp ASC, pk ASC",
            (chat_jid,),
        )
        return [self._row_to_normalized_message(row) for row in cur.fetchall()]

    def get_all_messages(self) -> list[NormalizedMessage]:
        """Fetch all indexed messages ordered chronologically."""
        cur = self.con.cursor()
        cur.execute("SELECT * FROM messages ORDER BY unix_timestamp ASC, pk ASC")
        return [self._row_to_normalized_message(row) for row in cur.fetchall()]

    def close(self) -> None:
        """Close database connection."""
        self.con.close()

    def __enter__(self) -> SearchIndex:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
