from __future__ import annotations

from collections import defaultdict

from total_recall.types import NormalizedMessage


class ReplyDAG:
    """Assembles quoted reply relationships into a Directed Acyclic Graph."""

    def __init__(self, messages: list[NormalizedMessage]):
        # Index messages by PK, sorted by timestamp and PK
        sorted_messages = sorted(messages, key=lambda m: (m.unix_timestamp, m.pk))
        self.messages: dict[int, NormalizedMessage] = {m.pk: m for m in sorted_messages}
        self.adjacency_list: dict[int, list[int]] = defaultdict(list)
        self.parents: dict[int, int] = {}
        self.roots: list[int] = []

        # Detect and avoid cycles while establishing parent-child edges
        for m in sorted_messages:
            if m.parent_pk and m.parent_pk in self.messages and m.parent_pk != m.pk:
                # Verify that adding this edge does not create a cycle
                curr: int | None = m.parent_pk
                is_cycle = False
                visited: set[int] = {m.pk}
                while curr is not None:
                    if curr in visited:
                        is_cycle = True
                        break
                    visited.add(curr)
                    curr = self.parents.get(curr)

                if not is_cycle:
                    self.adjacency_list[m.parent_pk].append(m.pk)
                    self.parents[m.pk] = m.parent_pk
                else:
                    self.roots.append(m.pk)
            else:
                self.roots.append(m.pk)

    def get_roots(self) -> list[NormalizedMessage]:
        """Return all root messages in the DAG."""
        return [self.messages[pk] for pk in self.roots]

    def get_children(self, pk: int) -> list[NormalizedMessage]:
        """Return direct child replies for the specified message."""
        child_pks = self.adjacency_list.get(pk, [])
        return [self.messages[cpk] for cpk in child_pks]

    def get_parent(self, pk: int) -> NormalizedMessage | None:
        """Return the parent message of the specified message, if any."""
        parent_pk = self.parents.get(pk)
        return self.messages.get(parent_pk) if parent_pk is not None else None

    def get_root_for(self, pk: int) -> NormalizedMessage:
        """Walk up the ancestry chain to find the thread's root question or starter."""
        if pk not in self.messages:
            raise KeyError(f"Message pk {pk} not found in ReplyDAG")

        curr = pk
        visited = {curr}
        while curr in self.parents:
            parent = self.parents[curr]
            if parent in visited:
                break
            visited.add(parent)
            curr = parent
        return self.messages[curr]

    def get_linear_ancestry(self, pk: int) -> list[NormalizedMessage]:
        """Return the linear chain of messages from the root message down to pk."""
        if pk not in self.messages:
            raise KeyError(f"Message pk {pk} not found in ReplyDAG")

        chain: list[int] = []
        curr: int | None = pk
        visited: set[int] = set()

        while curr is not None and curr not in visited:
            visited.add(curr)
            chain.append(curr)
            curr = self.parents.get(curr)

        chain.reverse()
        return [self.messages[p] for p in chain]

    def get_thread_messages(self, pk: int) -> list[NormalizedMessage]:
        """Return all messages belonging to the thread containing pk, starting at its root."""
        root = self.get_root_for(pk)
        collected_pks: list[int] = []
        visited: set[int] = set()

        # Depth-first traversal from root
        def _collect(current_pk: int) -> None:
            if current_pk in visited:
                return
            visited.add(current_pk)
            collected_pks.append(current_pk)
            for child_pk in self.adjacency_list.get(current_pk, []):
                _collect(child_pk)

        _collect(root.pk)
        return [self.messages[p] for p in collected_pks]

    def render_thread_tree(self, root_pk: int) -> str:
        """Render a formatted ASCII hierarchy tree for the thread starting at root_pk."""
        if root_pk not in self.messages:
            raise KeyError(f"Root message pk {root_pk} not found in ReplyDAG")

        lines: list[str] = []
        root_msg = self.messages[root_pk]
        lines.append(
            f"• [{root_msg.iso_date}] {root_msg.sender_name}: {root_msg.text} (msg_id: {root_msg.pk})"
        )

        def _render_children(parent_pk: int, prefix: str) -> None:
            children = self.adjacency_list.get(parent_pk, [])
            total = len(children)
            for idx, child_pk in enumerate(children):
                is_last = idx == total - 1
                connector = "└── " if is_last else "├── "
                child_msg = self.messages[child_pk]
                lines.append(
                    f"{prefix}{connector}[{child_msg.iso_date}] {child_msg.sender_name}: {child_msg.text} (msg_id: {child_msg.pk})"
                )
                child_prefix = prefix + ("    " if is_last else "│   ")
                _render_children(child_pk, child_prefix)

        _render_children(root_pk, "  ")
        return "\n".join(lines)

    def render_ascii_tree(self) -> str:
        """Render all threads in the DAG as an indented hierarchical tree."""
        trees: list[str] = []
        for root_pk in self.roots:
            trees.append(self.render_thread_tree(root_pk))
        return "\n\n".join(trees)

    def find_threads(self) -> dict[int, list[NormalizedMessage]]:
        """Map each root message PK to the full list of messages in its thread."""
        return {root_pk: self.get_thread_messages(root_pk) for root_pk in self.roots}
