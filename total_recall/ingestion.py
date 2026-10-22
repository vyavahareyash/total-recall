from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from total_recall.types import IngestionStats, NormalizedMessage

# Apple Epoch offset: seconds between 1970-01-01 and 2001-01-01 UTC
APPLE_EPOCH_OFFSET = 978_307_200

# 32-bit Sentinel boundary constants
SENTINEL_SORT_THRESHOLD = 200_000_000
SENTINEL_COUNTER_VALUE = 2_147_473_649


def apple_epoch_to_unix(apple_timestamp: float) -> float:
    """Convert Apple CoreData timestamp (seconds since 2001-01-01 UTC) to Unix Epoch."""
    return apple_timestamp + APPLE_EPOCH_OFFSET


def format_iso_timestamp(unix_timestamp: float) -> str:
    """Format Unix timestamp as UTC ISO 8601 string."""
    dt = datetime.fromtimestamp(unix_timestamp, tz=UTC)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


class IngestionEngine:
    """Ingests 1:1 messages from macOS WhatsApp Catalyst SQLite storage."""

    def __init__(
        self,
        chat_storage_path: Path | str,
        contacts_path: Path | str | None = None,
    ):
        self.chat_storage_path = Path(chat_storage_path)
        self.contacts_path = Path(contacts_path) if contacts_path else None

    @staticmethod
    def open_lock_free_connection(path: Path) -> sqlite3.Connection:
        """Connect to SQLite via URI immutable read-only mode to prevent locks."""
        resolved = path.resolve()
        uri = f"file:{resolved.as_posix()}?mode=ro&immutable=1"
        return sqlite3.connect(uri, uri=True)

    def _load_address_book(self) -> dict[str, str]:
        """Load contact full names from ContactsV2.sqlite if available."""
        address_book: dict[str, str] = {}
        if not self.contacts_path or not self.contacts_path.exists():
            return address_book

        try:
            con = self.open_lock_free_connection(self.contacts_path)
            cur = con.cursor()
            cur.execute(
                "SELECT ZWHATSAPPID, ZFULLNAME FROM ZWAADDRESSBOOKCONTACT WHERE ZWHATSAPPID IS NOT NULL AND ZFULLNAME IS NOT NULL"
            )
            for jid, name in cur.fetchall():
                address_book[str(jid)] = str(name)
            con.close()
        except (sqlite3.Error, OSError):
            pass
        return address_book

    def _load_push_names(self, con: sqlite3.Connection) -> dict[str, str]:
        """Load push names from ChatStorage.sqlite."""
        push_names: dict[str, str] = {}
        try:
            cur = con.cursor()
            cur.execute(
                "SELECT ZJID, ZPUSHNAME FROM ZWAPROFILEPUSHNAME WHERE ZJID IS NOT NULL AND ZPUSHNAME IS NOT NULL"
            )
            for jid, name in cur.fetchall():
                push_names[str(jid)] = str(name)
        except sqlite3.Error:
            pass
        return push_names

    def extract_1to1_messages(
        self,
    ) -> tuple[list[NormalizedMessage], IngestionStats]:
        """Extract and normalize all 1:1 chat messages with lock-free snapshot reads."""
        address_book = self._load_address_book()
        con = self.open_lock_free_connection(self.chat_storage_path)
        con.row_factory = sqlite3.Row
        cur = con.cursor()

        push_names = self._load_push_names(con)

        # Count total vs direct sessions
        cur.execute("SELECT COUNT(*) AS total_count FROM ZWACHATSESSION")
        total_sessions = int(cur.fetchone()["total_count"])

        cur.execute(
            "SELECT COUNT(*) AS direct_count FROM ZWACHATSESSION WHERE ZCONTACTJID NOT LIKE '%@g.us'"
        )
        direct_sessions = int(cur.fetchone()["direct_count"])

        # Fetch 1:1 messages ordered by true timestamp to eliminate sentinel sort skew
        cur.execute(
            """
            SELECT
                m.Z_PK,
                m.ZCHATSESSION,
                s.ZCONTACTJID,
                s.ZPARTNERNAME,
                m.ZSTANZAID,
                m.ZISFROMME,
                m.ZMESSAGEDATE,
                m.ZSORT,
                m.ZTEXT
            FROM ZWAMESSAGE m
            JOIN ZWACHATSESSION s ON s.Z_PK = m.ZCHATSESSION
            WHERE s.ZCONTACTJID NOT LIKE '%@g.us'
              AND m.ZTEXT IS NOT NULL
              AND length(m.ZTEXT) > 0
            ORDER BY m.ZMESSAGEDATE ASC, m.Z_PK ASC
            """
        )

        messages: list[NormalizedMessage] = []
        for row in cur.fetchall():
            is_from_me = bool(row["ZISFROMME"])
            chat_jid = str(row["ZCONTACTJID"])
            partner_name = str(row["ZPARTNERNAME"]) if row["ZPARTNERNAME"] else ""

            if is_from_me:
                sender_name = "You"
                sender_jid = "you"
            else:
                sender_jid = chat_jid
                sender_name = (
                    address_book.get(chat_jid)
                    or partner_name
                    or push_names.get(chat_jid)
                    or chat_jid
                )

            apple_date = float(row["ZMESSAGEDATE"])
            unix_ts = apple_epoch_to_unix(apple_date)
            iso_str = format_iso_timestamp(unix_ts)

            messages.append(
                NormalizedMessage(
                    pk=int(row["Z_PK"]),
                    chat_session_id=int(row["ZCHATSESSION"]),
                    chat_jid=chat_jid,
                    stanza_id=str(row["ZSTANZAID"] or ""),
                    is_from_me=is_from_me,
                    sender_name=sender_name,
                    sender_jid=sender_jid,
                    apple_date=apple_date,
                    unix_timestamp=unix_ts,
                    iso_date=iso_str,
                    text=str(row["ZTEXT"]),
                )
            )

        con.close()
        stats = IngestionStats(
            total_sessions_scanned=total_sessions,
            direct_sessions_ingested=direct_sessions,
            messages_ingested=len(messages),
        )
        return messages, stats
