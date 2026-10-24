import pytest

from whatsapp_total_recall import cli


@pytest.fixture(autouse=True)
def isolate_default_vault(tmp_path_factory, monkeypatch):
    """Isolate CLI tests from any vault_config.json present in cwd."""
    tmp_dir = tmp_path_factory.mktemp("isolated_vault")
    fake_vault = tmp_dir / "non_existent_vault.json"
    monkeypatch.setattr(cli, "DEFAULT_VAULT_PATH", fake_vault)
