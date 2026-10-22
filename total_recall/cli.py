from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
        help="Ingest 1:1 messages from WhatsApp Catalyst storage into the local index",
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
        messages, stats = engine.extract_1to1_messages()
    except Exception as exc:
        print(f"Error extracting messages: {exc}", file=sys.stderr)
        return 1

    with SearchIndex(args.db) as index:
        indexed_count = index.index_messages(messages)

    print(f"Ingestion complete into '{args.db}':")
    print(f"  Sessions scanned: {stats.total_sessions_scanned}")
    print(f"  Direct 1:1 sessions: {stats.direct_sessions_ingested}")
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest":
        return run_ingest(args)
    if args.command == "search":
        return run_search(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
