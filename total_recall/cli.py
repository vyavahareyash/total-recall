from __future__ import annotations

import argparse
import sys
from pathlib import Path

from total_recall.dag import ReplyDAG
from total_recall.discovery import discover_whatsapp_containers
from total_recall.ingestion import IngestionEngine
from total_recall.search import SearchIndex

DEFAULT_DB_PATH = Path("total_recall.db")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="total-recall",
        description="Total Recall: Privacy-First WhatsApp Conversational Intelligence",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ingest subcommand
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

    # Search subcommand
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

    # Thread subcommand
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

    return parser


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

    engine = IngestionEngine(source_path, contacts_path)
    try:
        if args.include_groups:
            messages, stats = engine.extract_messages(
                include_direct=True, include_groups=True
            )
        else:
            messages, stats = engine.extract_1to1_messages()
    except Exception as exc:
        print(f"Error extracting messages: {exc}", file=sys.stderr)
        return 1

    with SearchIndex(args.db) as index:
        indexed_count = index.index_messages(messages)

    print(f"Ingestion complete into '{args.db}':")
    print(f"  Sessions scanned: {stats.total_sessions_scanned}")
    print(f"  Direct 1:1 sessions: {stats.direct_sessions_ingested}")
    if stats.group_sessions_ingested > 0:
        print(f"  Group sessions: {stats.group_sessions_ingested}")
    print(f"  Messages indexed: {indexed_count}")
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

    # 1. If explicit --source provided, ingest directly from source
    if args.source and args.source.exists():
        engine = IngestionEngine(args.source, args.contacts)
        messages, _ = engine.extract_messages(include_direct=True, include_groups=True)
    # 2. Else check if local indexed db exists
    elif args.db and args.db.exists():
        with SearchIndex(args.db) as index:
            if args.chat:
                messages = index.get_messages_for_chat(args.chat)
            else:
                messages = index.get_all_messages()
    # 3. Else fallback to discovering local WhatsApp containers
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest":
        return run_ingest(args)
    if args.command == "search":
        return run_search(args)
    if args.command == "thread":
        return run_thread(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
