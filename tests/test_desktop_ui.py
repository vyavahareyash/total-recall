from __future__ import annotations

import json
import socket
import threading
from pathlib import Path

import pytest

from whatsapp_total_recall.chunking import chunk_into_episodes
from whatsapp_total_recall.config import AppConfig
from whatsapp_total_recall.drafter import (
    AutonomousSendingProhibitedError,
    DraftReviewBox,
)
from whatsapp_total_recall.mcp import validate_loopback_host
from whatsapp_total_recall.search import SearchIndex
from whatsapp_total_recall.summarizer import DeterministicMockLLMAdapter
from whatsapp_total_recall.types import NormalizedMessage
from whatsapp_total_recall.ui import (
    UIRequestHandler,
    UIServerContext,
    build_native_menu_bar_app,
)
from whatsapp_total_recall.vault import VaultConfig


@pytest.fixture
def test_env(tmp_path: Path):
    db_path = tmp_path / "test_ui.db"
    vault_path = tmp_path / "vault_config.json"
    review_box_path = db_path

    # Initialize vault
    vault = VaultConfig()
    vault.approve("123456789@s.whatsapp.net")
    vault.create_workspace("Work", ["123456789@s.whatsapp.net"])
    vault.save(vault_path)

    # Populate synthetic messages & episodes in SearchIndex
    messages = [
        NormalizedMessage(
            pk=1,
            chat_session_id=10,
            chat_jid="123456789@s.whatsapp.net",
            stanza_id="msg-1",
            is_from_me=False,
            sender_name="Alice",
            sender_jid="123456789@s.whatsapp.net",
            apple_date=700000000.0,
            unix_timestamp=1678307200.0,
            iso_date="2023-03-08T17:46:40Z",
            text="Can you review the project proposal by tomorrow?",
            parent_pk=None,
        ),
        NormalizedMessage(
            pk=2,
            chat_session_id=10,
            chat_jid="123456789@s.whatsapp.net",
            stanza_id="msg-2",
            is_from_me=True,
            sender_name="Me",
            sender_jid="me@s.whatsapp.net",
            apple_date=700000100.0,
            unix_timestamp=1678307300.0,
            iso_date="2023-03-08T17:48:20Z",
            text="Hey Alice, will review the proposal tonight.",
            parent_pk=1,
        ),
        NormalizedMessage(
            pk=3,
            chat_session_id=10,
            chat_jid="123456789@s.whatsapp.net",
            stanza_id="msg-3",
            is_from_me=False,
            sender_name="Alice",
            sender_jid="123456789@s.whatsapp.net",
            apple_date=700000200.0,
            unix_timestamp=1678307400.0,
            iso_date="2023-03-08T17:50:00Z",
            text="Thanks! Please also check slide 5 metrics.",
            parent_pk=2,
        ),
    ]

    episodes = chunk_into_episodes(
        messages=messages,
        chat_names={"123456789@s.whatsapp.net": "Alice"},
    )

    with SearchIndex(db_path) as index:
        index.index_episodes(episodes)

    llm = DeterministicMockLLMAdapter()
    config = AppConfig(db_path=db_path, vault_path=vault_path)

    context = UIServerContext(
        db_path=db_path,
        vault_path=vault_path,
        review_box_path=review_box_path,
        config=config,
        llm_adapter=llm,
    )

    return {
        "db_path": db_path,
        "vault_path": vault_path,
        "context": context,
    }


def make_ui_request(
    context: UIServerContext, method: str, path: str, body: dict | None = None
) -> tuple[int, dict | str]:
    """Execute HTTP request deterministically via in-memory socket pair."""
    c_sock, s_sock = socket.socketpair()
    c_sock.settimeout(3.0)
    body_bytes = json.dumps(body).encode("utf-8") if body is not None else b""
    header_lines = [
        f"{method} {path} HTTP/1.1",
        "Host: localhost",
        "Connection: close",
        f"Content-Length: {len(body_bytes)}",
    ]
    if body is not None:
        header_lines.append("Content-Type: application/json")
    raw_req = "\r\n".join(header_lines).encode("utf-8") + b"\r\n\r\n" + body_bytes
    c_sock.sendall(raw_req)

    UIRequestHandler.server_context = context

    class DummyServer:
        pass

    t = threading.Thread(
        target=lambda: UIRequestHandler(s_sock, ("127.0.0.1", 12345), DummyServer()),
        daemon=True,
    )
    t.start()

    chunks = []
    while True:
        try:
            chunk = c_sock.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
            raw_data = b"".join(chunks)
            if b"\r\n\r\n" in raw_data:
                header_part, body_part = raw_data.split(b"\r\n\r\n", 1)
                found_cl = False
                for line in header_part.split(b"\r\n"):
                    if line.lower().startswith(b"content-length:"):
                        cl = int(line.split(b":", 1)[1].strip())
                        found_cl = True
                        if len(body_part) >= cl:
                            break
                if found_cl and len(body_part) >= cl:
                    break
        except Exception:
            break

    t.join(timeout=2.0)
    c_sock.close()
    s_sock.close()

    raw_response = b"".join(chunks).decode("utf-8", errors="replace")
    status_line = raw_response.split("\r\n", 1)[0]
    status_code = int(status_line.split(" ")[1])
    parts = raw_response.split("\r\n\r\n", 1)
    body_str = parts[1] if len(parts) > 1 else ""
    try:
        return status_code, json.loads(body_str)
    except Exception:
        return status_code, body_str


def test_loopback_security_enforcement():
    """Verify non-loopback hosts are strictly rejected."""
    with pytest.raises(ValueError, match="must bind exclusively to 127.0.0.1"):
        validate_loopback_host("0.0.0.0")

    with pytest.raises(ValueError, match="must bind exclusively to 127.0.0.1"):
        validate_loopback_host("192.168.1.100")

    # Valid loopback addresses
    validate_loopback_host("127.0.0.1")
    validate_loopback_host("localhost")


def test_ui_index_html_serving(test_env):
    """Verify GET / returns the desktop Single Page Application HTML."""
    status, html = make_ui_request(test_env["context"], "GET", "/")
    assert status == 200
    assert "Total Recall" in html
    assert "Opt-In Chat Vault" in html
    assert "Instant Hybrid Search" in html
    assert "Review Box" in html


def test_api_status(test_env):
    """Verify GET /api/status reports container and permission status."""
    status, data = make_ui_request(test_env["context"], "GET", "/api/status")
    assert status == 200
    assert data["ok"] is True
    assert "has_permission" in data
    assert data["indexed_messages"] == 3
    assert data["approved_chats"] == 1


def test_api_vault_discovery_and_toggles(test_env):
    """Verify GET /api/vault, POST /api/vault/approve, and POST /api/vault/workspace."""
    # 1. Fetch vault state
    status, data = make_ui_request(test_env["context"], "GET", "/api/vault")
    assert status == 200
    assert data["ok"] is True
    assert "123456789@s.whatsapp.net" in data["approved_chat_jids"]
    assert "Work" in data["active_workspaces"]
    assert len(data["discovered_chats"]) >= 1

    # 2. Approve new chat
    status, app_res = make_ui_request(
        test_env["context"],
        "POST",
        "/api/vault/approve",
        {"chat_jid": "987654321@s.whatsapp.net", "approved": True},
    )
    assert status == 200
    assert app_res["is_approved"] is True
    assert "987654321@s.whatsapp.net" in app_res["approved_chat_jids"]

    # 3. Revoke chat
    status, rev_res = make_ui_request(
        test_env["context"],
        "POST",
        "/api/vault/approve",
        {"chat_jid": "987654321@s.whatsapp.net", "approved": False},
    )
    assert status == 200
    assert rev_res["is_approved"] is False

    # 4. Toggle workspace
    status, ws_res = make_ui_request(
        test_env["context"],
        "POST",
        "/api/vault/workspace",
        {"name": "Personal"},
    )
    assert status == 200
    assert "Personal" in ws_res["active_workspaces"]

    # 5. Bulk select and deselect all
    status, bulk_all = make_ui_request(
        test_env["context"],
        "POST",
        "/api/vault/bulk",
        {
            "approved": True,
            "chat_jids": ["123456789@s.whatsapp.net", "987654321@s.whatsapp.net"],
        },
    )
    assert status == 200
    assert bulk_all["ok"] is True
    assert "123456789@s.whatsapp.net" in bulk_all["approved_chat_jids"]

    status, bulk_none = make_ui_request(
        test_env["context"],
        "POST",
        "/api/vault/bulk",
        {"approved": False},
    )
    assert status == 200
    assert bulk_none["ok"] is True
    assert len(bulk_none["approved_chat_jids"]) == 0


def test_api_search_hybrid_results_and_citations(test_env):
    """Verify GET /api/search returns RRF scored episodes with clickable citations."""
    status, data = make_ui_request(
        test_env["context"], "GET", "/api/search?q=proposal&limit=5"
    )
    assert status == 200
    assert data["ok"] is True
    assert data["count"] >= 1

    result = data["results"][0]
    assert result["chat_jid"] == "123456789@s.whatsapp.net"
    assert result["chat_name"] == "Alice"
    assert result["rrf_score"] > 0
    assert len(result["citations"]) > 0

    # Verify exact verbatim citation format [Chat | Date | Sender | msg_id]
    first_citation = result["citations"][0]
    assert "[Alice | " in first_citation["citation_str"]
    assert " | 1]" in first_citation["citation_str"]

    # Verify query-time vault enforcement: revoking chat excludes it from search
    status, _ = make_ui_request(
        test_env["context"],
        "POST",
        "/api/vault/approve",
        {"chat_jid": "123456789@s.whatsapp.net", "approved": False},
    )
    assert status == 200
    status, unappr_data = make_ui_request(
        test_env["context"], "GET", "/api/search?q=proposal&limit=5"
    )
    assert status == 200
    assert unappr_data["count"] == 0

    # Re-approving restores search visibility immediately
    status, _ = make_ui_request(
        test_env["context"],
        "POST",
        "/api/vault/approve",
        {"chat_jid": "123456789@s.whatsapp.net", "approved": True},
    )
    assert status == 200
    status, reappr_data = make_ui_request(
        test_env["context"], "GET", "/api/search?q=proposal&limit=5"
    )
    assert status == 200
    assert reappr_data["count"] >= 1


def test_api_thread_dag_context(test_env):
    """Verify GET /api/thread/<msg_id> traces reply DAG hierarchy."""
    status, data = make_ui_request(test_env["context"], "GET", "/api/thread/2")
    assert status == 200
    assert data["ok"] is True
    assert data["root_id"] == 1
    assert data["thread_length"] == 3
    assert len(data["messages"]) == 3
    assert "rendered_tree" in data


def test_api_draft_review_box_workflow(test_env):
    """Verify draft staging, listing, approval, and rejection workflow."""
    # 1. Stage new draft
    status, stage_res = make_ui_request(
        test_env["context"],
        "POST",
        "/api/drafts/stage",
        {
            "chat_jid": "123456789@s.whatsapp.net",
            "intent_bullets": ["Reviewed proposal", "Everything looks solid"],
        },
    )
    assert status == 200
    draft_id = stage_res["draft_id"]
    assert stage_res["status"] == "STAGED_FOR_REVIEW"
    assert "Reviewed proposal" in stage_res["draft_text"]

    # 2. List drafts
    status, list_res = make_ui_request(test_env["context"], "GET", "/api/drafts")
    assert status == 200
    assert any(d["draft_id"] == draft_id for d in list_res["drafts"])

    # 3. Verify autonomous sending is structurally prohibited
    review_box = DraftReviewBox(test_env["db_path"])
    with pytest.raises(AutonomousSendingProhibitedError):
        review_box.send(draft_id)

    # 4. Delete draft via POST /api/drafts/<id>/delete
    status, del_res = make_ui_request(
        test_env["context"], "POST", f"/api/drafts/{draft_id}/delete", {}
    )
    assert status == 200
    assert del_res["deleted"] is True

    # Verify deleted from DB and subsequent delete returns 404
    assert review_box.get(draft_id) is None
    status, del_res_404 = make_ui_request(
        test_env["context"], "POST", f"/api/drafts/{draft_id}/delete", {}
    )
    assert status == 404

    # 7. Also verify DELETE /api/drafts/<id>
    status, stage_res2 = make_ui_request(
        test_env["context"],
        "POST",
        "/api/drafts/stage",
        {
            "chat_jid": "123456789@s.whatsapp.net",
            "intent_bullets": ["Second test draft"],
        },
    )
    assert status == 200
    draft_id2 = stage_res2["draft_id"]
    status, del2_res = make_ui_request(
        test_env["context"], "DELETE", f"/api/drafts/{draft_id2}"
    )
    assert status == 200
    assert del2_res["deleted"] is True
    assert review_box.get(draft_id2) is None


def test_api_models_runtime_selection(test_env):
    """Verify runtime model discovery and switching via /api/models."""
    # 1. Query models
    status, data = make_ui_request(test_env["context"], "GET", "/api/models")
    assert status == 200
    assert data["ok"] is True
    assert "current_model" in data
    assert len(data["models"]) >= 1

    # 2. Switch model
    status, switch_res = make_ui_request(
        test_env["context"], "POST", "/api/models", {"model": "llama3.2:3b"}
    )
    assert status == 200
    assert switch_res["ok"] is True
    assert switch_res["current_model"] == "llama3.2:3b"

    # 3. Verify switched model persisted in context
    status, data_after = make_ui_request(test_env["context"], "GET", "/api/models")
    assert status == 200
    assert data_after["current_model"] == "llama3.2:3b"

    # 4. Reject empty model name
    status, err_res = make_ui_request(
        test_env["context"], "POST", "/api/models", {"model": "  "}
    )
    assert status == 400


def test_native_swift_menu_bar_binary_compilation():
    """Verify native Swift menu bar application builds and is < 15 MB."""
    binary_path = build_native_menu_bar_app()
    if binary_path is None:
        pytest.skip("swiftc not available in this environment")

    assert binary_path.exists()
    size_mb = binary_path.stat().st_size / (1024 * 1024)
    # Acceptance criterion: < 15 MB
    assert size_mb < 15.0
