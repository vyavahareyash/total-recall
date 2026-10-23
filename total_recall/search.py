from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from total_recall.types import NormalizedMessage, SearchResult


class SearchIndex:
    """Manages SQLite FTS5 virtual tables and BM25 lexical message retrieval."""

    def __init__(self, db_path: Path | str = ":memory:"):
        self.db_path = str(db_path) if isinstance(db_path, Path) else db_path
        self.con = sqlite3.connect(self.db_path)
        self.con.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        """Create normalized messages table and FTS5 virtual table with parent_pk."""
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
            # Upsert into base table
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

            # Ensure FTS5 does not contain duplicate entry for same message pk
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

    @staticmethod
    def sanitize_fts_query(query: str) -> str:
        """Sanitize raw user query into safe FTS5 query expression."""
        tokens = re.findall(r"\w+", query)
        if not tokens:
            return ""
        # Match any token or token phrase safely
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
