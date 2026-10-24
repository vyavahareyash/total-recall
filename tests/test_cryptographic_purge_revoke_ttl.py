from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from whatsapp_total_recall.chunking import chunk_into_episodes
from whatsapp_total_recall.cli import main
from whatsapp_total_recall.embeddings import DeterministicMockEmbeddingProvider
from whatsapp_total_recall.ingestion import (
    APPLE_EPOCH_OFFSET,
    IngestionEngine,
    parse_disappearing_timer_string,
)
from whatsapp_total_recall.search import SearchIndex
from whatsapp_total_recall.types import NormalizedMessage
from whatsapp_total_recall.vault import VaultConfig


@pytest.fixture
def mock_embedder() -> DeterministicMockEmbeddingProvider:
    return DeterministicMockEmbeddingProvider(dim=64)


def make_test_message(
    pk: int,
    chat_jid: str,
    sender_name: str,
    text: str,
    unix_ts: float,
    stanza_id: str | None = None,
    parent_pk: int | None = None,
    expire_timer: int | None = None,
    expires_at: float | None = None,
) -> NormalizedMessage:
    dt = datetime.fromtimestamp(unix_ts, tz=UTC)
    iso_date = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    return NormalizedMessage(
        pk=pk,
        chat_session_id=1,
        chat_jid=chat_jid,
        stanza_id=stanza_id or f"STANZA_{pk}",
        is_from_me=(sender_name == "You"),
        sender_name=sender_name,
        sender_jid="you"
        if sender_name == "You"
        else f"{sender_name.lower()}@s.whatsapp.net",
        apple_date=unix_ts - APPLE_EPOCH_OFFSET,
        unix_timestamp=unix_ts,
        iso_date=iso_date,
        text=text,
        parent_pk=parent_pk,
        expire_timer=expire_timer,
        expires_at=expires_at,
    )


def create_fixture_chatstorage(
    db_path: Path,
    messages_data: list[tuple],
    session_expire_timer: int | None = None,
) -> None:
    """Create a synthetic ChatStorage.sqlite with ZWACHATSESSION and ZWAMESSAGE tables."""
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.executescript(
        """
        CREATE TABLE ZWACHATSESSION (
            Z_PK INTEGER PRIMARY KEY,
            ZCONTACTJID TEXT,
            ZPARTNERNAME TEXT,
            ZEXPIRETIMER INTEGER
        );

        CREATE TABLE ZWAMESSAGE (
            Z_PK INTEGER PRIMARY KEY,
            ZCHATSESSION INTEGER,
            ZSTANZAID TEXT,
            ZISFROMME INTEGER,
            ZGROUPMEMBER INTEGER,
            ZPARENTMESSAGE INTEGER,
            ZMESSAGEDATE REAL,
            ZSORT INTEGER,
            ZTEXT TEXT,
            ZMESSAGETYPE INTEGER,
            ZEXPIRETIMER INTEGER
        );
        """
    )
    cur.execute(
        "INSERT INTO ZWACHATSESSION (Z_PK, ZCONTACTJID, ZPARTNERNAME, ZEXPIRETIMER) VALUES (1, ?, ?, ?)",
        ("alice@s.whatsapp.net", "Alice", session_expire_timer),
    )
    for row in messages_data:
        cur.execute(
            """
            INSERT INTO ZWAMESSAGE (
                Z_PK, ZCHATSESSION, ZSTANZAID, ZISFROMME, ZGROUPMEMBER,
                ZPARENTMESSAGE, ZMESSAGEDATE, ZSORT, ZTEXT, ZMESSAGETYPE, ZEXPIRETIMER
            ) VALUES (?, 1, ?, ?, NULL, ?, ?, 10, ?, ?, ?)
            """,
            row,
        )
    con.commit()
    con.close()


# ==============================================================================
# 1. Revocation Detection & Extraction Tests
# ==============================================================================


def test_parse_disappearing_timer_string():
    """Verify parsing of human-readable disappearing message strings into seconds."""
    assert parse_disappearing_timer_string("24h") == 86400
    assert parse_disappearing_timer_string("24 hours") == 86400
    assert parse_disappearing_timer_string("1d") == 86400
    assert parse_disappearing_timer_string("7d") == 604800
    assert parse_disappearing_timer_string("1w") == 604800
    assert parse_disappearing_timer_string("90d") == 7776000
    assert parse_disappearing_timer_string("3m") == 7776000
    assert parse_disappearing_timer_string("60s") == 60
    assert parse_disappearing_timer_string("off") is None
    assert parse_disappearing_timer_string("0") is None
    assert parse_disappearing_timer_string("disabled") is None
    assert parse_disappearing_timer_string("") is None


def test_detect_revocation_events_cleared_ztext(tmp_path: Path):
    """Ingestion detects messages with NULL, empty, or tombstone ZTEXT as revocation events."""
    chat_db = tmp_path / "ChatStorage.sqlite"
    t0 = 1792659600.0 - APPLE_EPOCH_OFFSET

    # Message 1: Active
    # Message 2: Retracted (NULL ZTEXT)
    # Message 3: Retracted (empty ZTEXT)
    # Message 4: Retracted ("This message was deleted")
    # Message 5: Explicit ProtocolMessage_REVOKE
    msgs = [
        (1, "STZ_1", 0, None, t0, "Hello Alice", 0, None),
        (2, "STZ_2", 0, None, t0 + 10, None, 0, None),
        (3, "STZ_3", 0, None, t0 + 20, "   ", 0, None),
        (4, "STZ_4", 0, None, t0 + 30, "This message was deleted", 0, None),
        (5, "STZ_5", 0, None, t0 + 40, "ProtocolMessage_REVOKE", 15, None),
    ]
    create_fixture_chatstorage(chat_db, msgs)

    engine = IngestionEngine(chat_db)
    revoked = engine.detect_revocation_events()

    assert set(revoked) == {2, 3, 4, 5}


def test_detect_revocation_missing_from_snapshot(tmp_path: Path):
    """Ingestion identifies messages that were previously indexed but deleted from the SQLite snapshot."""
    chat_db = tmp_path / "ChatStorage.sqlite"
    index_db = tmp_path / "index.db"
    t0 = 1792659600.0

    m1 = make_test_message(1, "alice@s.whatsapp.net", "Alice", "Keep this message", t0)
    m2 = make_test_message(
        2, "alice@s.whatsapp.net", "Alice", "Deleted from WhatsApp", t0 + 60
    )

    with SearchIndex(index_db) as index:
        index.index_messages([m1, m2])
        assert index.get_all_message_pks() == {1, 2}

        # Snapshot now only contains message 1
        create_fixture_chatstorage(
            chat_db,
            [
                (
                    1,
                    "STZ_1",
                    0,
                    None,
                    t0 - APPLE_EPOCH_OFFSET,
                    "Keep this message",
                    0,
                    None,
                )
            ],
        )

        engine = IngestionEngine(chat_db)
        revoked = engine.detect_revocation_events(index=index)
        assert revoked == [2]


# ==============================================================================
# 2. Hard Cryptographic Purge & Vector / Lexical Removal Tests
# ==============================================================================


def test_purge_removes_raw_fts_and_vectors(
    mock_embedder: DeterministicMockEmbeddingProvider,
):
    """Purging a message removes raw rows, FTS5 tokens, updates parent episode, and recomputes vectors."""
    t0 = 1792659600.0
    chat_jid = "alice@s.whatsapp.net"
    m1 = make_test_message(
        1, chat_jid, "Alice", "Secret cryptographic key DEADBEEF99.", t0
    )
    m2 = make_test_message(
        2, chat_jid, "Bob", "Normal follow-up discussing the weather.", t0 + 60
    )

    episodes = chunk_into_episodes([m1, m2], {chat_jid: "Alice Chat"})

    with SearchIndex(":memory:", dimension=64, embedder=mock_embedder) as index:
        index.index_messages([m1, m2])
        index.index_episodes(
            episodes, index_constituent_messages=False, embedder=mock_embedder
        )

        # Pre-purge: both keyword and vector search find the secret
        assert len(index.search("DEADBEEF99")) == 1
        assert len(index.search_episodes("DEADBEEF99")) == 1
        hybrid_results = index.search_hybrid("DEADBEEF99", embedder=mock_embedder)
        assert len(hybrid_results) > 0
        assert "DEADBEEF99" in hybrid_results[0].episode.content

        # Execute hard purge
        report = index.purge(1, embedder=mock_embedder)
        assert report.message_pk == 1
        assert report.raw_messages_purged == 1
        assert report.episodes_updated == 1
        assert report.episodes_deleted == 0
        assert report.secure_delete_executed is True

        # Assert raw message and FTS are cleared
        assert index.get_message(1) is None
        assert len(index.search("DEADBEEF99")) == 0
        assert len(index.search_episodes("DEADBEEF99")) == 0

        # Assert parent episode content was updated and secret is gone
        ep = index.get_episode(episodes[0].episode_id)
        assert ep is not None
        assert "DEADBEEF99" not in ep.content
        assert "Normal follow-up" in ep.content

        # Assert hybrid search no longer surfaces the secret
        updated_hybrid = index.search_hybrid(
            "cryptographic key DEADBEEF99", embedder=mock_embedder
        )
        for res in updated_hybrid:
            assert "DEADBEEF99" not in res.episode.content


def test_purge_sole_message_deletes_entire_episode(
    mock_embedder: DeterministicMockEmbeddingProvider,
):
    """When the only message in an episode is purged, the episode is completely deleted across all tables."""
    t0 = 1792659600.0
    chat_jid = "confidential@s.whatsapp.net"
    m1 = make_test_message(10, chat_jid, "Eve", "Solo classified memo.", t0)
    episodes = chunk_into_episodes([m1], {chat_jid: "Classified"})
    ep_id = episodes[0].episode_id

    with SearchIndex(":memory:", dimension=64, embedder=mock_embedder) as index:
        index.index_messages([m1])
        index.index_episodes(
            episodes, index_constituent_messages=False, embedder=mock_embedder
        )

        assert index.get_episode(ep_id) is not None

        report = index.purge(10, embedder=mock_embedder)
        assert report.episodes_deleted == 1
        assert index.get_episode(ep_id) is None

        # Verify FTS5 virtual table
        cur = index.con.cursor()
        cur.execute("SELECT count(*) FROM episodes_fts WHERE episode_id = ?", (ep_id,))
        assert cur.fetchone()[0] == 0

        # Verify vector store
        cur.execute(
            "SELECT count(*) FROM episode_vectors WHERE episode_id = ?", (ep_id,)
        )
        assert cur.fetchone()[0] == 0


def test_purge_sanitizes_cross_burst_quotes(
    mock_embedder: DeterministicMockEmbeddingProvider,
):
    """When a quoted parent message is purged, cross-burst quote references in child episodes are sanitized."""
    t0 = 1792659600.0
    chat_jid = "team@s.whatsapp.net"

    m1 = make_test_message(1, chat_jid, "Alice", "Secret patent disclosure text.", t0)
    # Child message 2 hours later quoting m1
    m2 = make_test_message(
        2, chat_jid, "Bob", "I agree with your patent strategy.", t0 + 7200, parent_pk=1
    )

    episodes = chunk_into_episodes([m1, m2], {chat_jid: "Patent Team"})
    assert len(episodes) == 2  # Separated across bursts

    with SearchIndex(":memory:", dimension=64, embedder=mock_embedder) as index:
        index.index_messages([m1, m2])
        index.index_episodes(
            episodes, index_constituent_messages=False, embedder=mock_embedder
        )

        ep2_id = episodes[1].episode_id
        ep2_before = index.get_episode(ep2_id)
        assert "Secret patent disclosure" in ep2_before.header
        assert "Secret patent disclosure" in ep2_before.content

        # Purge message 1
        index.purge(1, embedder=mock_embedder)

        # Message 1 episode is deleted
        assert index.get_episode(episodes[0].episode_id) is None

        # Episode 2 is updated: the quote referencing m1 must be redacted/removed
        ep2_after = index.get_episode(ep2_id)
        assert ep2_after is not None
        assert "Secret patent disclosure" not in ep2_after.header
        assert "Secret patent disclosure" not in ep2_after.content
        assert "I agree with your patent strategy" in ep2_after.content


# ==============================================================================
# 3. Disk Zero-Filling via PRAGMA secure_delete Tests
# ==============================================================================


def test_pragma_secure_delete_disk_overwriting(tmp_path: Path):
    """Verify PRAGMA secure_delete = ON actively overwrites purged text with zeroes on disk pages."""
    db_file = tmp_path / "secure_purge.db"
    sensitive_token = "SUPER_SECRET_REDACTED_PASSWORD_987654321"

    m1 = make_test_message(
        100,
        "user@s.whatsapp.net",
        "User",
        f"Confidential payload: {sensitive_token}",
        1792659600.0,
    )

    # 1. Index sensitive message
    with SearchIndex(db_file) as index:
        index.index_messages([m1])

    # Assert raw string exists in binary database file on disk
    raw_disk_before = db_file.read_bytes()
    assert sensitive_token.encode("utf-8") in raw_disk_before

    # 2. Open index, purge message, and commit
    with SearchIndex(db_file) as index:
        report = index.purge(100)
        assert report.raw_messages_purged == 1
        assert report.secure_delete_executed is True

    # 3. Read raw binary file on disk again: verify the sensitive token has been zero-filled
    raw_disk_after = db_file.read_bytes()
    assert sensitive_token.encode("utf-8") not in raw_disk_after


# ==============================================================================
# 4. Ephemeral Disappearing Messages TTL Tests
# ==============================================================================


def test_ephemeral_ttl_automatic_purge(
    mock_embedder: DeterministicMockEmbeddingProvider,
):
    """Messages with TTL timer metadata automatically expire and are purged via purge_expired()."""
    t0 = 1792659600.0
    chat_jid = "ephemeral@s.whatsapp.net"

    # Message 1: 60s TTL (expires at t0 + 60)
    m1 = make_test_message(
        1, chat_jid, "Alice", "Self-destructing notice in 60s.", t0, expire_timer=60
    )
    assert m1.expires_at == t0 + 60.0

    # Message 2: 24h TTL (expires at t0 + 86400)
    m2 = make_test_message(
        2, chat_jid, "Alice", "Persistent for 24 hours.", t0, expire_timer=86400
    )
    assert m2.expires_at == t0 + 86400.0

    # Message 3: Standard permanent message (no TTL)
    m3 = make_test_message(3, chat_jid, "Bob", "Permanent agreement.", t0)
    assert m3.expires_at is None

    episodes = chunk_into_episodes([m1, m2, m3], {chat_jid: "Ephemeral Chat"})

    with SearchIndex(":memory:", dimension=64, embedder=mock_embedder) as index:
        index.index_messages([m1, m2, m3])
        index.index_episodes(
            episodes, index_constituent_messages=False, embedder=mock_embedder
        )

        # At t0 + 30: Nothing expired yet
        expired_30 = index.get_expired_messages(current_timestamp=t0 + 30.0)
        assert len(expired_30) == 0

        # At t0 + 100: Message 1 is expired (t0 + 60 <= t0 + 100)
        expired_100 = index.get_expired_messages(current_timestamp=t0 + 100.0)
        assert [m.pk for m in expired_100] == [1]

        # Trigger purge_expired at t0 + 100
        reports = index.purge_expired(
            current_timestamp=t0 + 100.0, embedder=mock_embedder
        )
        assert len(reports) == 1
        assert reports[0].message_pk == 1

        # Message 1 is purged
        assert index.get_message(1) is None
        assert index.get_message(2) is not None
        assert index.get_message(3) is not None

        # Episode no longer contains message 1
        ep = index.get_episode(episodes[0].episode_id)
        assert ep is not None
        assert "Self-destructing notice" not in ep.content
        assert "Persistent for 24 hours" in ep.content


def test_ingestion_sync_with_revocations_and_expired(
    tmp_path: Path, mock_embedder: DeterministicMockEmbeddingProvider
):
    """Ingestion sync automatically runs purge on detected revocations and expired messages."""
    chat_db = tmp_path / "ChatStorage.sqlite"
    index_db = tmp_path / "whatsapp_total_recall.db"
    t0_unix = 1792659600.0
    t0_apple = t0_unix - APPLE_EPOCH_OFFSET

    # Initial state: 3 messages in ChatStorage
    # Message 1: Normal
    # Message 2: Will be revoked later
    # Message 3: Disappearing with 30s timer
    initial_msgs = [
        (1, "STZ_1", 0, None, t0_apple, "Initial persistent message", 0, None),
        (
            2,
            "STZ_2",
            0,
            None,
            t0_apple + 10,
            "Secret error message to be deleted",
            0,
            None,
        ),
        (3, "STZ_3", 0, None, t0_apple + 20, "Temporary alert", 0, 30),
    ]
    create_fixture_chatstorage(chat_db, initial_msgs)

    vault = VaultConfig()
    vault.approve("alice@s.whatsapp.net")

    engine = IngestionEngine(chat_db)

    # First sync: index all 3
    with SearchIndex(index_db, embedder=mock_embedder) as index:
        episodes, report = engine.sync(
            vault, index=index, embedder=mock_embedder, current_timestamp=t0_unix
        )
        index.index_episodes(episodes, embedder=mock_embedder)
        assert len(index.get_all_message_pks()) == 3

    # Now update ChatStorage:
    # Message 2 has its ZTEXT cleared to NULL (Revocation event)
    con = sqlite3.connect(chat_db)
    con.execute("UPDATE ZWAMESSAGE SET ZTEXT = NULL WHERE Z_PK = 2")
    con.commit()
    con.close()

    # Second sync at t0_unix + 60s:
    # Message 2 should be purged (revoked)
    # Message 3 should be purged (expired at t0_unix + 20 + 30 = t0_unix + 50)
    with SearchIndex(index_db, embedder=mock_embedder) as index:
        episodes, report = engine.sync(
            vault,
            index=index,
            embedder=mock_embedder,
            current_timestamp=t0_unix + 60.0,
        )
        assert report.revocations_purged_count == 1
        assert report.expired_purged_count == 1

        # Only Message 1 remains
        assert index.get_all_message_pks() == {1}
        assert index.get_message(2) is None
        assert index.get_message(3) is None


# ==============================================================================
# 5. CLI Purge Command Tests
# ==============================================================================


def test_cli_purge_specific_message(tmp_path: Path, capsys: pytest.CaptureFixture):
    """Test 'total-recall purge <message_id>' subcommand."""
    db_file = tmp_path / "cli_test.db"
    m1 = make_test_message(
        42,
        "user@s.whatsapp.net",
        "User",
        "Retract me via CLI",
        1792659600.0,
        stanza_id="STZ_42",
    )
    episodes = chunk_into_episodes([m1], {"user@s.whatsapp.net": "User Chat"})

    with SearchIndex(db_file) as index:
        index.index_episodes(episodes)
        assert index.get_message(42) is not None

    # Run CLI purge
    ret = main(["purge", "42", "--db", str(db_file), "--embedder", "mock"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Cryptographic purge successful for message #42" in captured.out

    with SearchIndex(db_file) as index:
        assert index.get_message(42) is None


def test_cli_purge_expired(tmp_path: Path, capsys: pytest.CaptureFixture):
    """Test 'total-recall purge --expired' subcommand."""
    db_file = tmp_path / "cli_expired.db"
    # Create message with expired timestamp in the past
    past_ts = 100000.0
    m1 = make_test_message(
        77, "user@s.whatsapp.net", "User", "Expired long ago", past_ts, expire_timer=60
    )

    with SearchIndex(db_file) as index:
        index.index_messages([m1])
        assert index.get_message(77) is not None

    ret = main(["purge", "--expired", "--db", str(db_file), "--embedder", "mock"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Purged 1 expired disappearing message(s)" in captured.out

    with SearchIndex(db_file) as index:
        assert index.get_message(77) is None
