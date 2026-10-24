from __future__ import annotations

import io
import sqlite3
from contextlib import redirect_stdout
from pathlib import Path

from total_recall.cli import main as cli_main
from total_recall.config import AppConfig
from total_recall.ingestion import IngestionEngine
from total_recall.vault import VaultConfig


def test_app_config_roundtrip(tmp_path: Path):
    """Verify AppConfig serialization and deserialization."""
    cfg_file = tmp_path / "custom_config.json"
    cfg = AppConfig(
        db_path=Path("custom.db"),
        vault_path=Path("custom_vault.json"),
        ollama_host="http://192.168.1.100:11434",
        default_embedding_model="bge-large",
        default_llm_model="llama-3.2-3b",
        group_burst_minutes=20.0,
        direct_burst_minutes=60.0,
    )
    cfg.save(cfg_file)

    loaded = AppConfig.load(cfg_file)
    assert loaded.db_path == Path("custom.db")
    assert loaded.vault_path == Path("custom_vault.json")
    assert loaded.ollama_host == "http://192.168.1.100:11434"
    assert loaded.default_embedding_model == "bge-large"
    assert loaded.default_llm_model == "llama-3.2-3b"
    assert loaded.group_burst_minutes == 20.0
    assert loaded.direct_burst_minutes == 60.0


def test_vault_config_burst_and_host_persistence(tmp_path: Path):
    """Verify VaultConfig preserves burst thresholds and ollama host."""
    vault_file = tmp_path / "vault.json"
    vault = VaultConfig(
        group_burst_minutes=25.0,
        direct_burst_minutes=50.0,
        ollama_host="http://localhost:11435",
    )
    vault.approve("user@s.whatsapp.net")
    vault.save(vault_file)

    loaded = VaultConfig.load(vault_file)
    assert loaded.group_burst_minutes == 25.0
    assert loaded.direct_burst_minutes == 50.0
    assert loaded.ollama_host == "http://localhost:11435"
    assert loaded.is_chat_approved("user@s.whatsapp.net") is True


def test_cli_config_commands(tmp_path: Path):
    """Verify CLI 'config init', 'config set', and 'config show'."""
    cfg_file = tmp_path / "test_config.json"

    # 1. config init
    ret = cli_main(["config", "init", "--config", str(cfg_file)])
    assert ret == 0
    assert cfg_file.exists()

    # 2. config set
    ret = cli_main(
        [
            "config",
            "set",
            "--config",
            str(cfg_file),
            "--host",
            "http://127.0.0.1:9999",
            "--group-gap",
            "12.5",
            "--direct-gap",
            "35.0",
            "--embedding-model",
            "custom-embed",
        ]
    )
    assert ret == 0

    loaded = AppConfig.load(cfg_file)
    assert loaded.ollama_host == "http://127.0.0.1:9999"
    assert loaded.group_burst_minutes == 12.5
    assert loaded.direct_burst_minutes == 35.0
    assert loaded.default_embedding_model == "custom-embed"

    # 3. config show
    buf = io.StringIO()
    with redirect_stdout(buf):
        ret = cli_main(["config", "show", "--config", str(cfg_file)])
    assert ret == 0
    output = buf.getvalue()
    assert "TOTAL RECALL: CONFIGURATION" in output
    assert "http://127.0.0.1:9999" in output
    assert "12.5 minutes" in output


def test_ingestion_sync_respects_custom_burst_thresholds(tmp_path: Path):
    """Verify IngestionEngine.sync uses custom burst thresholds from VaultConfig."""
    chat_storage = tmp_path / "ChatStorage.sqlite"

    con = sqlite3.connect(chat_storage)
    con.executescript(
        """
        CREATE TABLE ZWACHATSESSION (
            Z_PK INTEGER PRIMARY KEY,
            ZCONTACTJID TEXT UNIQUE,
            ZPARTNERNAME TEXT,
            ZMESSAGECOUNTER INTEGER,
            ZUNREADCOUNT INTEGER
        );
        CREATE TABLE ZWAMESSAGE (
            Z_PK INTEGER PRIMARY KEY,
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
        INSERT INTO ZWACHATSESSION VALUES (1, '120363041234567890@g.us', 'Tech Group', 10, 0);
        """
    )

    # 6 messages spaced 10 minutes (600s) apart in Apple Epoch:
    # Under default 15m (900s) gap: 1 single episode.
    # Under custom 5m (300s) gap: split into multiple bursts.
    base_t = 700000000.0
    for i in range(6):
        con.execute(
            "INSERT INTO ZWAMESSAGE (Z_PK, ZCHATSESSION, ZSTANZAID, ZMESSAGEDATE, ZTEXT, ZMESSAGETYPE) "
            "VALUES (?, 1, ?, ?, ?, 0)",
            (i + 1, f"MSG_{i}", base_t + (i * 600.0), f"Group message {i}"),
        )
    con.commit()
    con.close()

    engine = IngestionEngine(chat_storage)
    vault_default = VaultConfig(approved_chat_jids={"120363041234567890@g.us"})
    episodes_default, _ = engine.sync(vault_default)
    assert len(episodes_default) == 1

    # Custom 5-minute threshold via vault_config
    vault_custom = VaultConfig(
        approved_chat_jids={"120363041234567890@g.us"},
        group_burst_minutes=5.0,
    )
    episodes_custom, _ = engine.sync(vault_custom)
    # Because inactivity gap of 10m > 5m, messages are split into multiple bursts
    assert len(episodes_custom) > 1
