from __future__ import annotations

import argparse
import sys
from pathlib import Path

from total_recall.chunking import chunk_into_episodes
from total_recall.dag import ReplyDAG
from total_recall.discovery import discover_whatsapp_containers
from total_recall.ingestion import IngestionEngine
from total_recall.search import SearchIndex
from total_recall.vault import VaultConfig

DEFAULT_DB_PATH = Path("total_recall.db")
DEFAULT_VAULT_PATH = Path("vault_config.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="total-recall",
        description="Total Recall: Privacy-First WhatsApp Conversational Intelligence",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. Ingest subcommand
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest messages from WhatsApp Catalyst storage into the local index",
    )
    ingest_parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Path to ChatStorage.sqlite (defaults to discovered macOS path)",
    )
    ingest_parser.add_argument(
        "--contacts",
        type=Path,
        default=None,
        help="Path to ContactsV2.sqlite (defaults to discovered macOS path)",
    )
    ingest_parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Target SQLite database for indexed messages (default: total_recall.db)",
    )
    ingest_parser.add_argument(
        "--include-groups",
        "--groups",
        action="store_true",
        default=False,
        help="Include group chats in ingestion (defaults to 1:1 only)",
    )
    ingest_parser.add_argument(
        "--vault",
        type=Path,
        default=None,
        help="Path to Opt-In Chat Vault config JSON (default: vault_config.json if present)",
    )
    ingest_parser.add_argument(
        "--approve",
        action="append",
        default=[],
        help="Approve specific chat JID(s) on the fly for this ingestion run",
    )
    ingest_parser.add_argument(
        "--workspace",
        action="append",
        default=[],
        help="Include active workspace(s) for this ingestion run",
    )
    ingest_parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Bypass Opt-In Vault perimeter and ingest all matching chats",
    )

    # 2. Search subcommand
    search_parser = subparsers.add_parser(
        "search",
        help="Search indexed messages using SQLite FTS5 BM25 lexical retrieval",
    )
    search_parser.add_argument(
        "query",
        type=str,
        help="Search query terms",
    )
    search_parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Target SQLite database to query (default: total_recall.db)",
    )
    search_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum matching results to return (default: 10)",
    )

    # 3. Thread subcommand
    thread_parser = subparsers.add_parser(
        "thread",
        help="Inspect full quoted reply thread from root question to final response",
    )
    thread_parser.add_argument(
        "message_id",
        type=int,
        nargs="?",
        default=None,
        help="Message PK to trace (resolves thread root and all replies)",
    )
    thread_parser.add_argument(
        "--chat",
        type=str,
        default=None,
        help="Target chat JID to inspect reply threads for",
    )
    thread_parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Path to ChatStorage.sqlite (defaults to discovered macOS path or local DB)",
    )
    thread_parser.add_argument(
        "--contacts",
        type=Path,
        default=None,
        help="Path to ContactsV2.sqlite",
    )
    thread_parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Local SQLite index database to query (default: total_recall.db)",
    )

    # 4. Vault subcommand
    vault_parser = subparsers.add_parser(
        "vault",
        help="Manage Opt-In Chat Vault perimeters, workspaces, and chat discovery",
    )
    vault_parser.add_argument(
        "--vault",
        type=Path,
        default=DEFAULT_VAULT_PATH,
        help="Path to Vault config file (default: vault_config.json)",
    )
    vault_parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Path to ChatStorage.sqlite",
    )
    vault_parser.add_argument(
        "--contacts",
        type=Path,
        default=None,
        help="Path to ContactsV2.sqlite",
    )
    vault_subparsers = vault_parser.add_subparsers(dest="vault_command", required=True)

    # vault list
    vault_subparsers.add_parser(
        "list",
        help="List all discovered chats and their vault approval status",
    )

    # vault report
    vault_subparsers.add_parser(
        "report",
        help="Surface comprehensive ingestion report of discovered chats and workspaces",
    )

    # vault approve <jids...>
    approve_sub = vault_subparsers.add_parser(
        "approve",
        help="Approve one or more chat JIDs in the vault",
    )
    approve_sub.add_argument("jids", nargs="+", help="Chat JIDs to approve")

    # vault revoke <jids...>
    revoke_sub = vault_subparsers.add_parser(
        "revoke",
        help="Revoke approval for one or more chat JIDs in the vault",
    )
    revoke_sub.add_argument("jids", nargs="+", help="Chat JIDs to revoke")

    # vault workspace [list|create|toggle]
    ws_sub = vault_subparsers.add_parser(
        "workspace",
        help="Manage workspace groupings",
    )
    ws_sub.add_argument(
        "action",
        choices=["list", "create", "toggle"],
        help="Workspace action",
    )
    ws_sub.add_argument("name", nargs="?", default=None, help="Workspace name")
    ws_sub.add_argument(
        "jids", nargs="*", default=[], help="Chat JIDs for create action"
    )

    # 5. Episodes subcommand
    episodes_parser = subparsers.add_parser(
        "episodes",
        help="Inspect or search indexed Conversational Flow Episodes",
    )
    episodes_parser.add_argument(
        "query",
        type=str,
        nargs="?",
        default=None,
        help="Optional search query across episodes",
    )
    episodes_parser.add_argument(
        "--chat",
        type=str,
        default=None,
        help="Filter episodes by chat JID",
    )
    episodes_parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Target SQLite database to query (default: total_recall.db)",
    )
    episodes_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum episodes to return (default: 5)",
    )

    return parser


def _resolve_source_and_contacts(
    source_path: Path | None, contacts_path: Path | None
) -> tuple[Path | None, Path | None]:
    if source_path and source_path.exists():
        return source_path, contacts_path

    containers = discover_whatsapp_containers()
    if (
        containers.has_permission
        and containers.chat_storage
        and containers.chat_storage.exists()
    ):
        return containers.chat_storage, contacts_path or containers.contacts
    return None, None


def run_ingest(args: argparse.Namespace) -> int:
    source_path = args.source
    contacts_path = args.contacts

    if not source_path:
        containers = discover_whatsapp_containers()
        if not containers.has_permission:
            print(
                "Error: Permission denied accessing WhatsApp container. Full Disk Access is required.",
                file=sys.stderr,
            )
            return 1
        if not containers.chat_storage or not containers.chat_storage.exists():
            print(
                "Error: Could not locate ChatStorage.sqlite. Specify --source explicitly.",
                file=sys.stderr,
            )
            return 1
        source_path = containers.chat_storage
        contacts_path = containers.contacts

    if not source_path.exists():
        print(f"Error: ChatStorage file not found: {source_path}", file=sys.stderr)
        return 1

    # Resolve VaultConfig
    vault_config: VaultConfig | None = None
    if not args.all:
        if args.vault:
            vault_config = VaultConfig.load(args.vault)
        elif args.approve or args.workspace:
            vault_config = (
                VaultConfig.load(DEFAULT_VAULT_PATH)
                if DEFAULT_VAULT_PATH.exists()
                else VaultConfig()
            )
        elif DEFAULT_VAULT_PATH.exists():
            vault_config = VaultConfig.load(DEFAULT_VAULT_PATH)

    if vault_config is not None:
        for jid in args.approve:
            vault_config.approve(jid)
        for ws in args.workspace:
            vault_config.toggle_workspace(ws, active=True)

    engine = IngestionEngine(source_path, contacts_path)
    try:
        report = engine.discover_chats(vault_config=vault_config)
        include_groups = bool(args.include_groups)
        messages, stats = engine.extract_messages(
            include_direct=True,
            include_groups=include_groups,
            vault_config=vault_config,
        )
    except Exception as exc:
        print(f"Error extracting messages: {exc}", file=sys.stderr)
        return 1

    # Chunk into Conversational Flow Episodes
    chat_names = {c.chat_jid: c.chat_name for c in report.chats}
    episodes = chunk_into_episodes(messages, chat_names=chat_names)

    with SearchIndex(args.db) as index:
        index.index_messages(messages)
        indexed_episodes = index.index_episodes(
            episodes, index_constituent_messages=False
        )

    print(f"Ingestion complete into '{args.db}':")
    print(f"  Sessions scanned: {stats.total_sessions_scanned}")
    print(f"  Direct 1:1 sessions: {stats.direct_sessions_ingested}")
    if stats.group_sessions_ingested > 0:
        print(f"  Group sessions: {stats.group_sessions_ingested}")
    print(f"  Messages indexed: {stats.messages_ingested}")
    if indexed_episodes > 0:
        print(f"  Episodes generated: {indexed_episodes}")

    if vault_config is not None:
        print("\n" + report.format_report())

    return 0


def run_search(args: argparse.Namespace) -> int:
    db_path: Path = args.db
    if not db_path.exists():
        print(
            f"Error: Index database '{db_path}' does not exist. Run 'total-recall ingest' first.",
            file=sys.stderr,
        )
        return 1

    with SearchIndex(db_path) as index:
        results = index.search(args.query, limit=args.limit)

    if not results:
        print(f"No matching messages found for: {args.query}")
        return 0

    print(f"Found {len(results)} matching messages for '{args.query}':\n")
    for r in results:
        print(f"[{r.iso_date}] {r.sender_name}:")
        print(f"{r.text}\n")
    return 0


def run_thread(args: argparse.Namespace) -> int:
    messages = []

    if args.source and args.source.exists():
        engine = IngestionEngine(args.source, args.contacts)
        messages, _ = engine.extract_messages(include_direct=True, include_groups=True)
    elif args.db and args.db.exists():
        with SearchIndex(args.db) as index:
            if args.chat:
                messages = index.get_messages_for_chat(args.chat)
            else:
                messages = index.get_all_messages()
    else:
        containers = discover_whatsapp_containers()
        if containers.chat_storage and containers.chat_storage.exists():
            engine = IngestionEngine(containers.chat_storage, containers.contacts)
            messages, _ = engine.extract_messages(
                include_direct=True, include_groups=True
            )
        else:
            print(
                "Error: No index database or WhatsApp containers found. "
                "Run 'total-recall ingest' or specify --source.",
                file=sys.stderr,
            )
            return 1

    if args.chat:
        messages = [m for m in messages if m.chat_jid == args.chat]

    if not messages:
        print(
            f"No messages found{' for chat ' + args.chat if args.chat else ''}.",
            file=sys.stderr,
        )
        return 1

    dag = ReplyDAG(messages)

    if args.message_id is not None:
        if args.message_id not in dag.messages:
            print(
                f"Error: Message #{args.message_id} not found in available messages.",
                file=sys.stderr,
            )
            return 1

        root = dag.get_root_for(args.message_id)
        thread_messages = dag.get_thread_messages(root.pk)
        tree_view = dag.render_thread_tree(root.pk)

        print(
            f"Thread for message #{args.message_id} "
            f"(Root: #{root.pk}, {len(thread_messages)} messages in thread):\n"
        )
        print(tree_view)
        return 0

    if args.chat:
        roots = dag.get_roots()
        print(f"Reply threads for chat {args.chat} ({len(roots)} threads):\n")
        print(dag.render_ascii_tree())
        return 0

    print(
        "Error: Please specify a message_id or --chat <chat_jid> to inspect reply threads.",
        file=sys.stderr,
    )
    return 1


def run_vault(args: argparse.Namespace) -> int:
    vault_path: Path = args.vault
    config = VaultConfig.load(vault_path) if vault_path.exists() else VaultConfig()

    cmd = args.vault_command

    if cmd == "approve":
        for jid in args.jids:
            config.approve(jid)
        config.save(vault_path)
        print(f"Approved {len(args.jids)} chat(s) in vault '{vault_path}':")
        for jid in args.jids:
            print(f"  + {jid}")
        return 0

    if cmd == "revoke":
        for jid in args.jids:
            config.revoke(jid)
        config.save(vault_path)
        print(f"Revoked {len(args.jids)} chat(s) from vault '{vault_path}':")
        for jid in args.jids:
            print(f"  - {jid}")
        return 0

    if cmd == "workspace":
        action = args.action
        if action == "list":
            if not config.workspaces:
                print(f"No workspaces configured in '{vault_path}'.")
                return 0
            print(f"Workspaces in '{vault_path}':")
            for name, jids in sorted(config.workspaces.items()):
                status = "ACTIVE" if name in config.active_workspaces else "INACTIVE"
                print(f"  • {name} [{status}] ({len(jids)} chats):")
                for jid in sorted(jids):
                    print(f"      - {jid}")
            return 0

        if action == "create":
            if not args.name:
                print(
                    "Error: Workspace name is required to create a workspace.",
                    file=sys.stderr,
                )
                return 1
            config.create_workspace(args.name, args.jids, active=True)
            config.save(vault_path)
            print(
                f"Workspace '{args.name}' created with {len(args.jids)} chats and set to ACTIVE."
            )
            return 0

        if action == "toggle":
            if not args.name:
                print("Error: Workspace name is required to toggle.", file=sys.stderr)
                return 1
            new_state = config.toggle_workspace(args.name)
            config.save(vault_path)
            state_str = "ACTIVE" if new_state else "INACTIVE"
            print(f"Workspace '{args.name}' toggled to {state_str}.")
            return 0

    if cmd in ("list", "report"):
        source, contacts = _resolve_source_and_contacts(args.source, args.contacts)
        if not source:
            print(
                "Error: ChatStorage.sqlite not found. Specify --source explicitly.",
                file=sys.stderr,
            )
            return 1

        engine = IngestionEngine(source, contacts)
        report = engine.discover_chats(vault_config=config)

        if cmd == "report":
            print(report.format_report())
        else:
            print(f"Discovered chats in '{source}' ({len(report.chats)} total):\n")
            for c in report.chats:
                mark = "[x]" if c.is_approved else "[ ]"
                ctype = "Group" if c.is_group else "1:1"
                ws = f" (Workspaces: {', '.join(c.workspaces)})" if c.workspaces else ""
                print(
                    f"  {mark} {c.chat_jid} | {c.chat_name} | {ctype} | {c.message_count} msgs{ws}"
                )

        return 0

    return 0


def run_episodes(args: argparse.Namespace) -> int:
    db_path: Path = args.db
    if not db_path.exists():
        print(
            f"Error: Database '{db_path}' does not exist. Run 'total-recall ingest' first.",
            file=sys.stderr,
        )
        return 1

    with SearchIndex(db_path) as index:
        if args.query:
            scored = index.search_episodes(args.query, limit=args.limit)
            if not scored:
                print(f"No matching episodes found for query: '{args.query}'")
                return 0
            print(f"Found {len(scored)} matching episodes for '{args.query}':\n")
            for s in scored:
                print(f"{s.episode.header}\nScore: {s.bm25_score:.3f}")
                print("-" * 50)
                print(s.episode.content)
                print("\n" + "=" * 60 + "\n")
            return 0

        if args.chat:
            episodes = index.get_episodes_for_chat(args.chat)[: args.limit]
        else:
            episodes = index.get_all_episodes()[: args.limit]

    if not episodes:
        print("No episodes found.")
        return 0

    print(f"Displaying {len(episodes)} Conversational Flow Episodes:\n")
    for ep in episodes:
        print(ep.header)
        print("-" * 50)
        print(ep.content)
        print("\n" + "=" * 60 + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest":
        return run_ingest(args)
    if args.command == "search":
        return run_search(args)
    if args.command == "thread":
        return run_thread(args)
    if args.command == "vault":
        return run_vault(args)
    if args.command == "episodes":
        return run_episodes(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
