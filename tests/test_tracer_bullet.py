from __future__ import annotations

import io
import sqlite3
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from whatsapp_total_recall.cli import main as cli_main
from whatsapp_total_recall.ingestion import (
    APPLE_EPOCH_OFFSET,
    IngestionEngine,
    apple_epoch_to_unix,
    format_iso_timestamp,
)
from whatsapp_total_recall.search import SearchIndex


@pytest.fixture
def synthetic_coredata_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Create authentic CoreData SQLite tables matching macOS Catalyst WhatsApp."""
    chat_storage_path = tmp_path / "ChatStorage.sqlite"
    contacts_path = tmp_path / "ContactsV2.sqlite"

    # 1. ContactsV2.sqlite
    con_contacts = sqlite3.connect(contacts_path)
    cur_contacts = con_contacts.cursor()
    cur_contacts.executescript(
        """
        CREATE TABLE ZWAADDRESSBOOKCONTACT (
            Z_PK INTEGER PRIMARY KEY AUTOINCREMENT,
            ZWHATSAPPID TEXT,
            ZFULLNAME TEXT,
            ZPHONENUMBER TEXT
        );
        INSERT INTO ZWAADDRESSBOOKCONTACT (ZWHATSAPPID, ZFULLNAME, ZPHONENUMBER) VALUES
            ('14155550199@s.whatsapp.net', 'David Miller', '+1 415 555 0199'),
            ('14155550200@s.whatsapp.net', 'Elena Rostova', '+1 415 555 0200');
        """
    )
    con_contacts.commit()
    con_contacts.close()

    # 2. ChatStorage.sqlite
    con_chat = sqlite3.connect(chat_storage_path)
    cur_chat = con_chat.cursor()
    cur_chat.executescript(
        """
        CREATE TABLE ZWACHATSESSION (
            Z_PK INTEGER PRIMARY KEY AUTOINCREMENT,
            ZCONTACTJID TEXT UNIQUE,
            ZPARTNERNAME TEXT,
            ZMESSAGECOUNTER INTEGER,
            ZUNREADCOUNT INTEGER
        );

        CREATE TABLE ZWAPROFILEPUSHNAME (
            Z_PK INTEGER PRIMARY KEY AUTOINCREMENT,
            ZJID TEXT UNIQUE,
            ZPUSHNAME TEXT
        );

        CREATE TABLE ZWAMESSAGE (
            Z_PK INTEGER PRIMARY KEY AUTOINCREMENT,
            ZCHATSESSION INTEGER,
            ZSTANZAID TEXT,
            ZISFROMME INTEGER,
            ZGROUPMEMBER INTEGER,
            ZPARENTMESSAGE INTEGER,
            ZMESSAGEDATE REAL,
            ZSENTDATE REAL,
            ZSORT INTEGER,
            ZTEXT TEXT,
            ZMESSAGETYPE INTEGER
        );

        -- Session 1: Group Chat (MUST BE FILTERED OUT IN 1:1 INGESTION)
        INSERT INTO ZWACHATSESSION (Z_PK, ZCONTACTJID, ZPARTNERNAME, ZMESSAGECOUNTER, ZUNREADCOUNT)
        VALUES (1, '120363041234567890@g.us', 'Engineering Core Group', 2147473649, 0);

        -- Session 2: 1:1 Direct Chat with David Miller (Has sentinel ZMESSAGECOUNTER)
        INSERT INTO ZWACHATSESSION (Z_PK, ZCONTACTJID, ZPARTNERNAME, ZMESSAGECOUNTER, ZUNREADCOUNT)
        VALUES (2, '14155550199@s.whatsapp.net', 'David Partner Name', 2147473649, 0);

        -- Session 3: 1:1 Direct Chat with no contacts entry (Uses Push Name)
        INSERT INTO ZWACHATSESSION (Z_PK, ZCONTACTJID, ZPARTNERNAME, ZMESSAGECOUNTER, ZUNREADCOUNT)
        VALUES (3, '14155550300@s.whatsapp.net', NULL, 42, 0);

        INSERT INTO ZWAPROFILEPUSHNAME (ZJID, ZPUSHNAME) VALUES
            ('14155550300@s.whatsapp.net', 'PushName User');
        """
    )

    # Base timestamp: 2026-10-22 09:00:00 UTC
    base_unix = 1792659600.0
    t0 = base_unix - APPLE_EPOCH_OFFSET

    # Insert messages
    messages = [
        # Group messages (ChatSession 1) - should be ignored
        (1, 1, "STZ-GRP-1", 0, 1, None, t0, t0, 10, "Group announcement only", 0),
        # 1:1 Direct messages (ChatSession 2):
        # Notice: Message 2 has an artificially inflated sentinel ZSORT (214748364)!
        # If sorted by ZSORT, message 2 would appear AFTER message 3 & 4.
        # But chronological sequencing by ZMESSAGEDATE puts it at t0 + 60s.
        (
            2,
            2,
            "STZ-DIR-1",
            0,
            None,
            None,
            t0,
            t0,
            100,
            "Hey! Did you review the lock-free snapshot ingestion spec?",
            0,
        ),
        (
            3,
            2,
            "STZ-DIR-2",
            1,
            None,
            None,
            t0 + 60,
            t0 + 60,
            214748364,
            "Yes, immutable read mode prevents 0xDEAD10CC crashes.",
            0,
        ),
        (
            4,
            2,
            "STZ-DIR-3",
            0,
            None,
            None,
            t0 + 120,
            t0 + 120,
            102,
            "What about FTS5 lexical tokenization with Porter stemming?",
            0,
        ),
        (
            5,
            2,
            "STZ-DIR-4",
            1,
            None,
            None,
            t0 + 180,
            t0 + 180,
            103,
            "Porter stemmer handles plural terms like connections and searches.",
            0,
        ),
        # 1:1 Direct message (ChatSession 3): from partner with push name
        (
            6,
            3,
            "STZ-DIR-5",
            0,
            None,
            None,
            t0 + 240,
            t0 + 240,
            104,
            "Hello from push name contact!",
            0,
        ),
    ]

    cur_chat.executemany(
        """
        INSERT INTO ZWAMESSAGE (
            Z_PK, ZCHATSESSION, ZSTANZAID, ZISFROMME, ZGROUPMEMBER,
            ZPARENTMESSAGE, ZMESSAGEDATE, ZSENTDATE, ZSORT, ZTEXT, ZMESSAGETYPE
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        messages,
    )
    con_chat.commit()
    con_chat.close()

    return chat_storage_path, contacts_path


def test_lock_free_connection_prevents_write_locks(
    synthetic_coredata_fixture: tuple[Path, Path],
):
    """Verify connection opened via mode=ro&immutable=1 is read-only and acquires no write locks."""
    chat_storage, _ = synthetic_coredata_fixture
    con = IngestionEngine.open_lock_free_connection(chat_storage)

    # Read queries succeed
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM ZWAMESSAGE")
    count = cur.fetchone()[0]
    assert count == 6

    # Attempting to modify raises OperationalError: attempt to write a readonly database
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        cur.execute("DELETE FROM ZWAMESSAGE")

    con.close()


def test_apple_epoch_normalization():
    """Verify CoreData Apple Epoch timestamps convert accurately to Unix Epoch and UTC ISO."""
    apple_zero = 0.0
    unix_zero = apple_epoch_to_unix(apple_zero)
    assert unix_zero == 978_307_200.0
    assert format_iso_timestamp(unix_zero) == "2001-01-01 00:00:00 UTC"

    # Known modern timestamp: 2026-10-22 09:00:00 UTC
    unix_target = 1792659600.0
    apple_target = unix_target - APPLE_EPOCH_OFFSET
    assert apple_epoch_to_unix(apple_target) == unix_target
    assert format_iso_timestamp(unix_target) == "2026-10-22 09:00:00 UTC"


def test_sentinel_sort_filtering_and_chronology(
    synthetic_coredata_fixture: tuple[Path, Path],
):
    """Verify chronological sequencing by true timestamp bypasses 32-bit sentinel sort values."""
    chat_storage, contacts = synthetic_coredata_fixture
    engine = IngestionEngine(chat_storage, contacts)
    messages, _ = engine.extract_1to1_messages()

    # Verify messages in Session 2 (pk 2, 3, 4, 5) are strictly ascending by unix_timestamp
    session2_msgs = [m for m in messages if m.chat_session_id == 2]
    assert len(session2_msgs) == 4

    # Even though pk=3 had sentinel ZSORT = 214748364, it must be ordered at index 1 (second chronologically)
    assert [m.pk for m in session2_msgs] == [2, 3, 4, 5]
    for m1, m2 in zip(session2_msgs[:-1], session2_msgs[1:], strict=False):
        assert m1.unix_timestamp < m2.unix_timestamp


def test_1to1_filtering_excludes_groups(synthetic_coredata_fixture: tuple[Path, Path]):
    """Verify group chats (@g.us) are strictly excluded from 1:1 message extraction."""
    chat_storage, contacts = synthetic_coredata_fixture
    engine = IngestionEngine(chat_storage, contacts)
    messages, stats = engine.extract_1to1_messages()

    # Total 3 sessions: 1 group, 2 direct
    assert stats.total_sessions_scanned == 3
    assert stats.direct_sessions_ingested == 2

    # Group message pk=1 must not be present
    message_pks = [m.pk for m in messages]
    assert 1 not in message_pks
    assert set(message_pks) == {2, 3, 4, 5, 6}
    for m in messages:
        assert not m.chat_jid.endswith("@g.us")


def test_sender_resolution_1to1(synthetic_coredata_fixture: tuple[Path, Path]):
    """Verify sender identity resolution for 1:1 chats."""
    chat_storage, contacts = synthetic_coredata_fixture
    engine = IngestionEngine(chat_storage, contacts)
    messages, _ = engine.extract_1to1_messages()

    msg_by_pk = {m.pk: m for m in messages}

    # ZISFROMME=1 -> "You"
    assert msg_by_pk[3].is_from_me is True
    assert msg_by_pk[3].sender_name == "You"
    assert msg_by_pk[3].sender_jid == "you"

    # ZISFROMME=0 with entry in ContactsV2 -> "David Miller" (resolved from contacts, not partner name fallback)
    assert msg_by_pk[2].is_from_me is False
    assert msg_by_pk[2].sender_name == "David Miller"
    assert msg_by_pk[2].sender_jid == "14155550199@s.whatsapp.net"

    # ZISFROMME=0 with no contacts entry -> resolved from ZWAPROFILEPUSHNAME -> "PushName User"
    assert msg_by_pk[6].is_from_me is False
    assert msg_by_pk[6].sender_name == "PushName User"
    assert msg_by_pk[6].sender_jid == "14155550300@s.whatsapp.net"


def test_fts5_indexing_and_bm25_search(synthetic_coredata_fixture: tuple[Path, Path]):
    """Verify indexing into SQLite FTS5 table and lexical search with BM25 ranking."""
    chat_storage, contacts = synthetic_coredata_fixture
    engine = IngestionEngine(chat_storage, contacts)
    messages, _ = engine.extract_1to1_messages()

    with SearchIndex(":memory:") as index:
        indexed_count = index.index_messages(messages)
        assert indexed_count == 5

        # Query exact technical keyword
        results = index.search("0xDEAD10CC")
        assert len(results) == 1
        assert results[0].message_pk == 3
        assert results[0].sender_name == "You"
        assert "0xDEAD10CC" in results[0].text
        assert results[0].bm25_score < 0.0  # SQLite FTS5 BM25 returns negative score

        # Query stemmed term: "searching" should match "searches" via Porter tokenizer
        stemmed_results = index.search("searching")
        assert len(stemmed_results) >= 1
        assert any(r.message_pk == 5 for r in stemmed_results)

        # Query non-existent keyword
        empty_results = index.search("nonexistentterm12345")
        assert len(empty_results) == 0


def test_cli_ingest_and_search_end_to_end(
    synthetic_coredata_fixture: tuple[Path, Path], tmp_path: Path
):
    """Verify CLI subcommands 'ingest' and 'search' work end-to-end against synthetic store."""
    chat_storage, contacts = synthetic_coredata_fixture
    target_db = tmp_path / "test_total_recall.db"

    # 1. Run CLI ingest
    buf = io.StringIO()
    with redirect_stdout(buf):
        exit_code = cli_main(
            [
                "ingest",
                "--source",
                str(chat_storage),
                "--contacts",
                str(contacts),
                "--db",
                str(target_db),
            ]
        )
    assert exit_code == 0
    output = buf.getvalue()
    assert "Ingestion complete" in output
    assert "Direct 1:1 sessions: 2" in output
    assert "Messages indexed: 5" in output

    # 2. Run CLI search
    buf_search = io.StringIO()
    with redirect_stdout(buf_search):
        exit_code_search = cli_main(
            [
                "search",
                "snapshot ingestion",
                "--db",
                str(target_db),
            ]
        )
    assert exit_code_search == 0
    search_output = buf_search.getvalue()
    assert "David Miller:" in search_output
    assert "Did you review the lock-free snapshot ingestion spec?" in search_output
    assert "[2026-10-22" in search_output
