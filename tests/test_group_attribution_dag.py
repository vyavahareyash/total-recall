from __future__ import annotations

import io
import sqlite3
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from total_recall.cli import main as cli_main
from total_recall.dag import ReplyDAG
from total_recall.ingestion import (
    APPLE_EPOCH_OFFSET,
    IngestionEngine,
)
from total_recall.search import SearchIndex


@pytest.fixture
def synthetic_group_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Create authentic CoreData SQLite tables matching WhatsApp Catalyst group chats."""
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
            ('14155550101@s.whatsapp.net', 'Alice Smith', '+1 415 555 0101'),
            ('14155550102@s.whatsapp.net', 'Bob Jones', '+1 415 555 0102');
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

        CREATE TABLE ZWAGROUPMEMBER (
            Z_PK INTEGER PRIMARY KEY AUTOINCREMENT,
            ZCHATSESSION INTEGER,
            ZMEMBERJID TEXT
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

        -- Chat 1: Multi-party Group Chat
        INSERT INTO ZWACHATSESSION (Z_PK, ZCONTACTJID, ZPARTNERNAME, ZMESSAGECOUNTER, ZUNREADCOUNT)
        VALUES (1, '120363041234567890@g.us', 'AI Architecture Group', 100, 0);

        -- Chat 2: 1:1 Direct Chat
        INSERT INTO ZWACHATSESSION (Z_PK, ZCONTACTJID, ZPARTNERNAME, ZMESSAGECOUNTER, ZUNREADCOUNT)
        VALUES (2, '14155550101@s.whatsapp.net', 'Alice Direct', 50, 0);

        -- Group Members for Chat 1
        INSERT INTO ZWAGROUPMEMBER (Z_PK, ZCHATSESSION, ZMEMBERJID) VALUES
            (1, 1, '14155550101@s.whatsapp.net'),  -- In contacts -> Alice Smith
            (2, 1, '14155550102@s.whatsapp.net'),  -- In contacts -> Bob Jones (also has push name)
            (3, 1, '14155550103@s.whatsapp.net'),  -- Only push name -> Charlie Push
            (4, 1, '14155550104@s.whatsapp.net');  -- Neither -> Fallback to JID

        -- Push names
        INSERT INTO ZWAPROFILEPUSHNAME (ZJID, ZPUSHNAME) VALUES
            ('14155550102@s.whatsapp.net', 'Bob Push'),
            ('14155550103@s.whatsapp.net', 'Charlie Push');
        """
    )

    base_unix = 1792659600.0  # 2026-10-22 09:00:00 UTC
    t0 = base_unix - APPLE_EPOCH_OFFSET

    # Multi-turn quoted reply chain with intervening chatter
    messages = [
        # Msg 1: Root question from Alice
        (
            1,
            1,
            "STZ-1",
            0,
            1,
            None,
            t0,
            t0,
            10,
            "Should we use sqlite-vec or an external vector DB like Qdrant?",
            0,
        ),
        # Msg 2: Bob replies to Msg 1
        (
            2,
            1,
            "STZ-2",
            0,
            2,
            1,
            t0 + 60,
            t0 + 60,
            11,
            "sqlite-vec is zero-cost and embedded, no Docker needed.",
            0,
        ),
        # Msg 3: You reply to Bob (Msg 2)
        (
            3,
            1,
            "STZ-3",
            1,
            None,
            2,
            t0 + 120,
            t0 + 120,
            12,
            "Agreed, Simon Willison showed sqlite-vec + FTS5 works cleanly.",
            0,
        ),
        # Msg 4: Charlie asks unrelated question (root of separate thread)
        (
            4,
            1,
            "STZ-4",
            0,
            3,
            None,
            t0 + 180,
            t0 + 180,
            13,
            "Anyone free for lunch today?",
            0,
        ),
        # Msg 5: Member 4 replies to Charlie (Msg 4)
        (
            5,
            1,
            "STZ-5",
            0,
            4,
            4,
            t0 + 240,
            t0 + 240,
            14,
            "Tacos sound good to me.",
            0,
        ),
        # Msg 6: Alice replies to Bob (Msg 2) 1 hour later (intervening chatter!)
        (
            6,
            1,
            "STZ-6",
            0,
            1,
            2,
            t0 + 3600,
            t0 + 3600,
            15,
            "Going back to Bob's point on sqlite-vec, Porter tokenization handles plurals.",
            0,
        ),
        # Msg 7: 1:1 chat message (in ChatSession 2)
        (
            7,
            2,
            "STZ-7",
            0,
            None,
            None,
            t0 + 4000,
            t0 + 4000,
            16,
            "Direct 1:1 message from Alice",
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


def test_group_speaker_attribution_cascade(
    synthetic_group_fixture: tuple[Path, Path],
):
    """Verify group message sender resolution cascade: ContactsV2 -> PushName -> Member JID."""
    chat_storage, contacts = synthetic_group_fixture
    engine = IngestionEngine(chat_storage, contacts)
    messages, stats = engine.extract_group_messages()

    assert stats.total_sessions_scanned == 2
    assert stats.group_sessions_ingested == 1
    assert stats.direct_sessions_ingested == 0
    assert len(messages) == 6

    msg_map = {m.pk: m for m in messages}

    # 1. Contacts match: Alice Smith
    assert msg_map[1].sender_name == "Alice Smith"
    assert msg_map[1].sender_jid == "14155550101@s.whatsapp.net"
    assert msg_map[1].is_from_me is False

    # 2. Contacts match takes priority over push name: Bob Jones (Contacts) vs Bob Push (PushName)
    assert msg_map[2].sender_name == "Bob Jones"
    assert msg_map[2].sender_jid == "14155550102@s.whatsapp.net"
    assert msg_map[2].is_from_me is False

    # 3. ZISFROMME = 1 always resolves to 'You'
    assert msg_map[3].sender_name == "You"
    assert msg_map[3].sender_jid == "you"
    assert msg_map[3].is_from_me is True

    # 4. Push name fallback when not in Contacts: Charlie Push
    assert msg_map[4].sender_name == "Charlie Push"
    assert msg_map[4].sender_jid == "14155550103@s.whatsapp.net"
    assert msg_map[4].is_from_me is False

    # 5. Fallback to raw member JID when neither contact nor push name exists
    assert msg_map[5].sender_name == "14155550104@s.whatsapp.net"
    assert msg_map[5].sender_jid == "14155550104@s.whatsapp.net"
    assert msg_map[5].is_from_me is False

    # 6. Alice Smith quoting across intervening chatter
    assert msg_map[6].sender_name == "Alice Smith"
    assert msg_map[6].sender_jid == "14155550101@s.whatsapp.net"
    assert msg_map[6].parent_pk == 2


def test_reply_dag_construction_and_traversal(
    synthetic_group_fixture: tuple[Path, Path],
):
    """Verify Quoted Reply DAG accurately stitches threads across intervening chatter."""
    chat_storage, contacts = synthetic_group_fixture
    engine = IngestionEngine(chat_storage, contacts)
    messages, _ = engine.extract_group_messages()

    dag = ReplyDAG(messages)

    # Roots check: Message 1 and Message 4
    roots = dag.get_roots()
    root_pks = [r.pk for r in roots]
    assert root_pks == [1, 4]

    # Children of Root 1: Bob (Msg 2)
    children_1 = dag.get_children(1)
    assert [c.pk for c in children_1] == [2]

    # Children of Bob (Msg 2): You (Msg 3) and Alice (Msg 6)
    children_2 = dag.get_children(2)
    assert [c.pk for c in children_2] == [3, 6]

    # Children of Root 4: Member 4 (Msg 5)
    children_4 = dag.get_children(4)
    assert [c.pk for c in children_4] == [5]

    # Ancestry traversal from leaf Message 6: [1, 2, 6]
    ancestry_6 = dag.get_linear_ancestry(6)
    assert [m.pk for m in ancestry_6] == [1, 2, 6]

    # Root resolution from intermediate or leaf messages
    assert dag.get_root_for(6).pk == 1
    assert dag.get_root_for(3).pk == 1
    assert dag.get_root_for(2).pk == 1
    assert dag.get_root_for(1).pk == 1
    assert dag.get_root_for(5).pk == 4

    # Full thread collection for root 1 includes [1, 2, 3, 6]
    thread_1_msgs = dag.get_thread_messages(6)
    assert [m.pk for m in thread_1_msgs] == [1, 2, 3, 6]


def test_reply_dag_tree_rendering(
    synthetic_group_fixture: tuple[Path, Path],
):
    """Verify ASCII hierarchical tree representation reflects thread progression."""
    chat_storage, contacts = synthetic_group_fixture
    engine = IngestionEngine(chat_storage, contacts)
    messages, _ = engine.extract_group_messages()

    dag = ReplyDAG(messages)
    tree_text = dag.render_thread_tree(1)

    assert "• [" in tree_text
    assert "Alice Smith: Should we use sqlite-vec" in tree_text
    assert "├── [" in tree_text or "└── [" in tree_text
    assert "Bob Jones: sqlite-vec is zero-cost" in tree_text
    assert "You: Agreed, Simon Willison" in tree_text
    assert "Alice Smith: Going back to Bob's point" in tree_text


def test_cli_thread_inspection_end_to_end(
    synthetic_group_fixture: tuple[Path, Path],
):
    """Verify CLI 'thread' subcommand inspects threads from root question to final response."""
    chat_storage, contacts = synthetic_group_fixture

    # Test tracing from middle message #2
    buf = io.StringIO()
    with redirect_stdout(buf):
        exit_code = cli_main(
            [
                "thread",
                "2",
                "--source",
                str(chat_storage),
                "--contacts",
                str(contacts),
            ]
        )
    assert exit_code == 0
    output = buf.getvalue()
    assert "Thread for message #2 (Root: #1" in output
    assert "Alice Smith: Should we use sqlite-vec" in output
    assert "Bob Jones: sqlite-vec is zero-cost" in output
    assert "Alice Smith: Going back to Bob's point" in output

    # Test tracing from leaf message #6
    buf_leaf = io.StringIO()
    with redirect_stdout(buf_leaf):
        exit_code_leaf = cli_main(
            [
                "thread",
                "6",
                "--source",
                str(chat_storage),
                "--contacts",
                str(contacts),
            ]
        )
    assert exit_code_leaf == 0
    assert "Root: #1" in buf_leaf.getvalue()

    # Test tracing unrelated thread message #5
    buf_lunch = io.StringIO()
    with redirect_stdout(buf_lunch):
        exit_code_lunch = cli_main(
            [
                "thread",
                "5",
                "--source",
                str(chat_storage),
                "--contacts",
                str(contacts),
            ]
        )
    assert exit_code_lunch == 0
    lunch_output = buf_lunch.getvalue()
    assert "Root: #4" in lunch_output
    assert "Charlie Push: Anyone free for lunch today?" in lunch_output
    assert "14155550104@s.whatsapp.net: Tacos sound good to me." in lunch_output


def test_cli_ingest_with_groups_and_thread_inspection(
    synthetic_group_fixture: tuple[Path, Path], tmp_path: Path
):
    """Verify CLI ingestion with --include-groups indexes group messages and supports thread lookup."""
    chat_storage, contacts = synthetic_group_fixture
    target_db = tmp_path / "groups_indexed.db"

    # 1. Ingest with groups
    buf_ingest = io.StringIO()
    with redirect_stdout(buf_ingest):
        code_ingest = cli_main(
            [
                "ingest",
                "--include-groups",
                "--source",
                str(chat_storage),
                "--contacts",
                str(contacts),
                "--db",
                str(target_db),
            ]
        )
    assert code_ingest == 0
    ingest_out = buf_ingest.getvalue()
    assert "Group sessions: 1" in ingest_out
    assert "Direct 1:1 sessions: 1" in ingest_out
    assert "Messages indexed: 7" in ingest_out

    # 2. Inspect thread via target_db directly (without --source)
    buf_thread = io.StringIO()
    with redirect_stdout(buf_thread):
        code_thread = cli_main(
            [
                "thread",
                "6",
                "--db",
                str(target_db),
            ]
        )
    assert code_thread == 0
    thread_out = buf_thread.getvalue()
    assert "Thread for message #6 (Root: #1, 4 messages in thread)" in thread_out
    assert "Alice Smith: Should we use sqlite-vec" in thread_out
    assert "Bob Jones: sqlite-vec is zero-cost" in thread_out


def test_search_index_parent_pk_persistence():
    """Verify SearchIndex stores and retrieves parent_pk for reply graph queries."""
    with SearchIndex(":memory:") as index:
        from total_recall.types import NormalizedMessage

        m1 = NormalizedMessage(
            pk=10,
            chat_session_id=1,
            chat_jid="120363041234567890@g.us",
            stanza_id="S1",
            is_from_me=False,
            sender_name="Alice",
            sender_jid="alice@s.whatsapp.net",
            apple_date=100.0,
            unix_timestamp=APPLE_EPOCH_OFFSET + 100.0,
            iso_date="2001-01-01 00:01:40 UTC",
            text="Initial question",
            parent_pk=None,
        )
        m2 = NormalizedMessage(
            pk=11,
            chat_session_id=1,
            chat_jid="120363041234567890@g.us",
            stanza_id="S2",
            is_from_me=True,
            sender_name="You",
            sender_jid="you",
            apple_date=160.0,
            unix_timestamp=APPLE_EPOCH_OFFSET + 160.0,
            iso_date="2001-01-01 00:02:40 UTC",
            text="Reply answer",
            parent_pk=10,
        )

        index.index_messages([m1, m2])

        fetched_10 = index.get_message(10)
        fetched_11 = index.get_message(11)

        assert fetched_10 is not None
        assert fetched_10.parent_pk is None

        assert fetched_11 is not None
        assert fetched_11.parent_pk == 10

        chat_msgs = index.get_messages_for_chat("120363041234567890@g.us")
        assert len(chat_msgs) == 2
        assert chat_msgs[0].pk == 10
        assert chat_msgs[1].pk == 11
