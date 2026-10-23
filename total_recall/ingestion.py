from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from total_recall.chunking import chunk_into_episodes
from total_recall.types import (
    ConversationalFlowEpisode,
    IngestionStats,
    NormalizedMessage,
)
from total_recall.vault import DiscoveredChat, IngestionReport, VaultConfig

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
    """Ingests messages from macOS WhatsApp Catalyst SQLite storage."""

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
                "SELECT ZWHATSAPPID, ZFULLNAME FROM ZWAADDRESSBOOKCONTACT "
                "WHERE ZWHATSAPPID IS NOT NULL AND ZFULLNAME IS NOT NULL"
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
                "SELECT ZJID, ZPUSHNAME FROM ZWAPROFILEPUSHNAME "
                "WHERE ZJID IS NOT NULL AND ZPUSHNAME IS NOT NULL"
            )
            for jid, name in cur.fetchall():
                push_names[str(jid)] = str(name)
        except sqlite3.Error:
            pass
        return push_names

    def _load_group_members(self, con: sqlite3.Connection) -> dict[int, str]:
        """Load group member JIDs from ZWAGROUPMEMBER if table exists."""
        group_members: dict[int, str] = {}
        try:
            cur = con.cursor()
            cur.execute(
                "SELECT Z_PK, ZMEMBERJID FROM ZWAGROUPMEMBER "
                "WHERE Z_PK IS NOT NULL AND ZMEMBERJID IS NOT NULL"
            )
            for pk, jid in cur.fetchall():
                group_members[int(pk)] = str(jid)
        except sqlite3.Error:
            pass
        return group_members

    def discover_chats(
        self, vault_config: VaultConfig | None = None
    ) -> IngestionReport:
        """Scan ChatStorage to inventory discovered chats and evaluate vault opt-in status."""
        address_book = self._load_address_book()
        con = self.open_lock_free_connection(self.chat_storage_path)
        con.row_factory = sqlite3.Row
        cur = con.cursor()

        push_names = self._load_push_names(con)

        # Message count per session
        msg_counts: dict[int, int] = {}
        try:
            cur.execute(
                "SELECT ZCHATSESSION, COUNT(*) AS cnt FROM ZWAMESSAGE "
                "WHERE ZTEXT IS NOT NULL AND length(ZTEXT) > 0 "
                "GROUP BY ZCHATSESSION"
            )
            for row in cur.fetchall():
                msg_counts[int(row["ZCHATSESSION"])] = int(row["cnt"])
        except sqlite3.Error:
            pass

        cur.execute(
            "SELECT Z_PK, ZCONTACTJID, ZPARTNERNAME FROM ZWACHATSESSION ORDER BY Z_PK ASC"
        )
        sessions = cur.fetchall()
        con.close()

        discovered: list[DiscoveredChat] = []
        direct_count = 0
        group_count = 0
        total_msgs = 0
        approved_count = 0

        for row in sessions:
            pk = int(row["Z_PK"])
            jid = str(row["ZCONTACTJID"])
            partner = str(row["ZPARTNERNAME"]) if row["ZPARTNERNAME"] else ""
            is_group = jid.endswith("@g.us")
            count = msg_counts.get(pk, 0)
            total_msgs += count

            if is_group:
                group_count += 1
                name = partner or jid
            else:
                direct_count += 1
                name = address_book.get(jid) or partner or push_names.get(jid) or jid

            is_appr = vault_config.is_chat_approved(jid) if vault_config else False
            if is_appr:
                approved_count += 1

            workspaces = (
                tuple(vault_config.get_workspaces_for_chat(jid)) if vault_config else ()
            )

            discovered.append(
                DiscoveredChat(
                    chat_session_id=pk,
                    chat_jid=jid,
                    chat_name=name,
                    is_group=is_group,
                    message_count=count,
                    is_approved=is_appr,
                    workspaces=workspaces,
                )
            )

        active_ws = (
            tuple(sorted(vault_config.active_workspaces)) if vault_config else ()
        )
        return IngestionReport(
            total_chats_discovered=len(discovered),
            direct_chats_discovered=direct_count,
            group_chats_discovered=group_count,
            total_messages_discovered=total_msgs,
            approved_chats_count=approved_count,
            active_workspaces=active_ws,
            chats=tuple(discovered),
        )

    def extract_messages(
        self,
        include_direct: bool = True,
        include_groups: bool = True,
        vault_config: VaultConfig | None = None,
    ) -> tuple[list[NormalizedMessage], IngestionStats]:
        """Extract and normalize chat messages with lock-free snapshot reads and vault filtering."""
        if not include_direct and not include_groups:
            return [], IngestionStats(
                total_sessions_scanned=0,
                direct_sessions_ingested=0,
                group_sessions_ingested=0,
                messages_ingested=0,
            )

        address_book = self._load_address_book()
        con = self.open_lock_free_connection(self.chat_storage_path)
        con.row_factory = sqlite3.Row
        cur = con.cursor()

        push_names = self._load_push_names(con)
        group_members = self._load_group_members(con)

        # Count total vs direct vs group sessions
        cur.execute("SELECT COUNT(*) AS total_count FROM ZWACHATSESSION")
        total_sessions = int(cur.fetchone()["total_count"])

        cur.execute(
            "SELECT COUNT(*) AS direct_count FROM ZWACHATSESSION WHERE ZCONTACTJID NOT LIKE '%@g.us'"
        )
        direct_sessions_total = int(cur.fetchone()["direct_count"])

        cur.execute(
            "SELECT COUNT(*) AS group_count FROM ZWACHATSESSION WHERE ZCONTACTJID LIKE '%@g.us'"
        )
        group_sessions_total = int(cur.fetchone()["group_count"])

        # Construct WHERE clause based on session inclusion
        where_conditions = ["m.ZTEXT IS NOT NULL", "length(m.ZTEXT) > 0"]
        if include_direct and not include_groups:
            where_conditions.append("s.ZCONTACTJID NOT LIKE '%@g.us'")
        elif include_groups and not include_direct:
            where_conditions.append("s.ZCONTACTJID LIKE '%@g.us'")

        where_clause = " AND ".join(where_conditions)

        query = f"""
            SELECT
                m.Z_PK,
                m.ZCHATSESSION,
                s.ZCONTACTJID,
                s.ZPARTNERNAME,
                m.ZSTANZAID,
                m.ZISFROMME,
                m.ZGROUPMEMBER,
                m.ZPARENTMESSAGE,
                m.ZMESSAGEDATE,
                m.ZSORT,
                m.ZTEXT
            FROM ZWAMESSAGE m
            JOIN ZWACHATSESSION s ON s.Z_PK = m.ZCHATSESSION
            WHERE {where_clause}
            ORDER BY m.ZMESSAGEDATE ASC, m.Z_PK ASC
        """

        cur.execute(query)

        messages: list[NormalizedMessage] = []
        ingested_session_jids: set[str] = set()

        for row in cur.fetchall():
            chat_jid = str(row["ZCONTACTJID"])

            # Opt-In Chat Vault filter: strictly exclude unapproved chats
            if vault_config is not None and not vault_config.is_chat_approved(chat_jid):
                continue

            ingested_session_jids.add(chat_jid)
            is_from_me = bool(row["ZISFROMME"])
            is_group = chat_jid.endswith("@g.us")
            partner_name = str(row["ZPARTNERNAME"]) if row["ZPARTNERNAME"] else ""

            if is_from_me:
                sender_name = "You"
                sender_jid = "you"
            elif is_group:
                # Group speaker resolution:
                # 1. ZWAGROUPMEMBER member JID
                # 2. ContactsV2 full name
                # 3. Push name fallback
                # 4. Member JID fallback
                raw_gm = row["ZGROUPMEMBER"]
                gm_pk = int(raw_gm) if raw_gm is not None else None
                member_jid = group_members.get(gm_pk) if gm_pk is not None else None

                if member_jid:
                    sender_jid = member_jid
                    sender_name = (
                        address_book.get(member_jid)
                        or push_names.get(member_jid)
                        or member_jid
                    )
                else:
                    sender_jid = "unknown"
                    sender_name = "Unknown"
            else:
                # 1:1 Direct chat speaker resolution
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
            parent_pk = (
                int(row["ZPARENTMESSAGE"])
                if row["ZPARENTMESSAGE"] is not None
                else None
            )

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
                    parent_pk=parent_pk,
                )
            )

        con.close()

        # Calculate actual sessions ingested
        if vault_config is not None:
            direct_ingested = sum(
                1 for jid in ingested_session_jids if not jid.endswith("@g.us")
            )
            group_ingested = sum(
                1 for jid in ingested_session_jids if jid.endswith("@g.us")
            )
        else:
            direct_ingested = direct_sessions_total if include_direct else 0
            group_ingested = group_sessions_total if include_groups else 0

        stats = IngestionStats(
            total_sessions_scanned=total_sessions,
            direct_sessions_ingested=direct_ingested,
            group_sessions_ingested=group_ingested,
            messages_ingested=len(messages),
        )
        return messages, stats

    def extract_1to1_messages(
        self,
        vault_config: VaultConfig | None = None,
    ) -> tuple[list[NormalizedMessage], IngestionStats]:
        """Extract and normalize 1:1 chat messages (backwards-compatible)."""
        return self.extract_messages(
            include_direct=True, include_groups=False, vault_config=vault_config
        )

    def extract_group_messages(
        self,
        vault_config: VaultConfig | None = None,
    ) -> tuple[list[NormalizedMessage], IngestionStats]:
        """Extract and normalize group chat messages."""
        return self.extract_messages(
            include_direct=False, include_groups=True, vault_config=vault_config
        )

    def sync(
        self,
        vault_config: VaultConfig,
        chunk_episodes: bool = True,
    ) -> tuple[list[ConversationalFlowEpisode], IngestionReport]:
        """Synchronize WhatsApp store using Opt-In Chat Vault perimeter and return episodes & report."""
        report = self.discover_chats(vault_config=vault_config)
        messages, _ = self.extract_messages(
            include_direct=True, include_groups=True, vault_config=vault_config
        )

        chat_names = {c.chat_jid: c.chat_name for c in report.chats}
        episodes = (
            chunk_into_episodes(messages, chat_names=chat_names)
            if chunk_episodes
            else []
        )

        ingested_jids = {m.chat_jid for m in messages}
        final_report = IngestionReport(
            total_chats_discovered=report.total_chats_discovered,
            direct_chats_discovered=report.direct_chats_discovered,
            group_chats_discovered=report.group_chats_discovered,
            total_messages_discovered=report.total_messages_discovered,
            approved_chats_count=report.approved_chats_count,
            active_workspaces=report.active_workspaces,
            chats=report.chats,
            ingested_chats_count=len(ingested_jids),
            ingested_messages_count=len(messages),
            episodes_generated_count=len(episodes),
        )

        return episodes, final_report
