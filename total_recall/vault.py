from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DiscoveredChat:
    chat_session_id: int
    chat_jid: str
    chat_name: str
    is_group: bool
    message_count: int
    is_approved: bool = False
    workspaces: tuple[str, ...] = ()


@dataclass
class VaultConfig:
    """Opt-In Chat Vault configuration.

    Enforces a strict default-zero indexing policy:
    No chats are indexed unless explicitly approved or included in an active workspace.
    """

    approved_chat_jids: set[str] = field(default_factory=set)
    workspaces: dict[str, set[str]] = field(default_factory=dict)
    active_workspaces: set[str] = field(default_factory=set)

    def is_chat_approved(self, chat_jid: str) -> bool:
        """Return True if chat_jid is explicitly approved or part of an active workspace."""
        if chat_jid in self.approved_chat_jids:
            return True
        for ws in self.active_workspaces:
            if chat_jid in self.workspaces.get(ws, set()):
                return True
        return False

    def approve(self, chat_jid: str) -> None:
        """Explicitly approve a chat JID for indexing."""
        self.approved_chat_jids.add(chat_jid)

    def revoke(self, chat_jid: str) -> None:
        """Revoke approval for a chat JID across approved set and workspaces."""
        self.approved_chat_jids.discard(chat_jid)
        for jids in self.workspaces.values():
            jids.discard(chat_jid)

    def create_workspace(
        self, name: str, jids: Iterable[str] | None = None, active: bool = True
    ) -> None:
        """Create or replace a named workspace with a set of chat JIDs."""
        self.workspaces[name] = set(jids) if jids else set()
        if active:
            self.active_workspaces.add(name)

    def add_to_workspace(self, name: str, chat_jid: str) -> None:
        """Add a chat JID to an existing or new workspace."""
        if name not in self.workspaces:
            self.workspaces[name] = set()
        self.workspaces[name].add(chat_jid)

    def remove_from_workspace(self, name: str, chat_jid: str) -> None:
        """Remove a chat JID from a workspace."""
        if name in self.workspaces:
            self.workspaces[name].discard(chat_jid)

    def toggle_workspace(self, name: str, active: bool | None = None) -> bool:
        """Toggle active state for a workspace. If active is None, inverts current state."""
        if name not in self.workspaces:
            self.workspaces[name] = set()
        if active is None:
            active = name not in self.active_workspaces
        if active:
            self.active_workspaces.add(name)
        else:
            self.active_workspaces.discard(name)
        return name in self.active_workspaces

    def get_workspaces_for_chat(self, chat_jid: str) -> list[str]:
        """Return all workspace names that include the specified chat JID."""
        return [ws for ws, jids in sorted(self.workspaces.items()) if chat_jid in jids]

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a JSON-compatible dictionary."""
        return {
            "version": 1,
            "approved_chat_jids": sorted(self.approved_chat_jids),
            "workspaces": {
                name: sorted(jids) for name, jids in sorted(self.workspaces.items())
            },
            "active_workspaces": sorted(self.active_workspaces),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VaultConfig:
        """Deserialize configuration from dictionary."""
        approved = set(data.get("approved_chat_jids", []))
        raw_workspaces = data.get("workspaces", {})
        workspaces = {name: set(jids) for name, jids in raw_workspaces.items()}
        active_workspaces = set(data.get("active_workspaces", []))
        return cls(
            approved_chat_jids=approved,
            workspaces=workspaces,
            active_workspaces=active_workspaces,
        )

    def save(self, path: Path | str) -> None:
        """Save configuration to JSON file."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path | str) -> VaultConfig:
        """Load configuration from JSON file. Returns empty default if file does not exist."""
        file_path = Path(path)
        if not file_path.exists():
            return cls()
        data = json.loads(file_path.read_text())
        return cls.from_dict(data)


@dataclass(frozen=True)
class IngestionReport:
    total_chats_discovered: int
    direct_chats_discovered: int
    group_chats_discovered: int
    total_messages_discovered: int
    approved_chats_count: int
    active_workspaces: tuple[str, ...]
    chats: tuple[DiscoveredChat, ...]
    ingested_chats_count: int = 0
    ingested_messages_count: int = 0
    episodes_generated_count: int = 0
    revocations_purged_count: int = 0
    expired_purged_count: int = 0

    def format_report(self) -> str:
        """Generate human-readable ingestion & discovery report."""
        lines = [
            "============================================================",
            "             TOTAL RECALL: INGESTION REPORT                 ",
            "============================================================",
            f"Discovered Chats: {self.total_chats_discovered} total "
            f"({self.direct_chats_discovered} direct 1:1, {self.group_chats_discovered} groups)",
            f"Total Messages in Storage: {self.total_messages_discovered}",
            f"Approved Chats in Vault: {self.approved_chats_count}",
            f"Active Workspaces: {', '.join(self.active_workspaces) if self.active_workspaces else '(none)'}",
            f"Ingested Messages: {self.ingested_messages_count} across {self.ingested_chats_count} chats",
            f"Conversational Flow Episodes: {self.episodes_generated_count}",
        ]
        if self.revocations_purged_count > 0:
            lines.append(f"Revoked Messages Purged: {self.revocations_purged_count}")
        if self.expired_purged_count > 0:
            lines.append(
                f"Expired Ephemeral Messages Purged: {self.expired_purged_count}"
            )

        lines.extend(
            [
                "------------------------------------------------------------",
                "Discovered Chats Inventory:",
            ]
        )

        if not self.chats:
            lines.append("  (No chat sessions found in storage)")
        else:
            for chat in self.chats:
                mark = "[x]" if chat.is_approved else "[ ]"
                chat_type = "Group" if chat.is_group else "1:1"
                ws_str = (
                    f" [Workspaces: {', '.join(chat.workspaces)}]"
                    if chat.workspaces
                    else ""
                )
                lines.append(
                    f"  {mark} {chat.chat_jid} ({chat.chat_name}) - "
                    f"{chat_type}, {chat.message_count} msgs{ws_str}"
                )

        lines.append("============================================================")
        return "\n".join(lines)
