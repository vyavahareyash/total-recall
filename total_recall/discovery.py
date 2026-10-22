from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WhatsAppContainers:
    group_dir: Path | None
    chat_storage: Path | None
    contacts: Path | None
    has_permission: bool


def discover_whatsapp_containers() -> WhatsAppContainers:
    """Inspect standard macOS paths for WhatsApp Catalyst App Group containers."""
    home = Path.home()
    group_dir = home / "Library/Group Containers/group.net.whatsapp.WhatsApp.shared"
    chat_storage = group_dir / "ChatStorage.sqlite"
    contacts = group_dir / "ContactsV2.sqlite"

    try:
        if not group_dir.exists():
            return WhatsAppContainers(
                group_dir=None,
                chat_storage=None,
                contacts=None,
                has_permission=True,
            )

        cs_path = chat_storage if chat_storage.exists() else None
        contacts_path = contacts if contacts.exists() else None

        return WhatsAppContainers(
            group_dir=group_dir,
            chat_storage=cs_path,
            contacts=contacts_path,
            has_permission=True,
        )
    except PermissionError:
        return WhatsAppContainers(
            group_dir=group_dir,
            chat_storage=None,
            contacts=None,
            has_permission=False,
        )
    except OSError:
        return WhatsAppContainers(
            group_dir=None,
            chat_storage=None,
            contacts=None,
            has_permission=False,
        )
