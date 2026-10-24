---
name: total-recall
description: Query local WhatsApp history, reconstruct reply DAG thread contexts, and synthesize intent-constrained reply drafts via Total Recall's local MCP server.
---

# Total Recall MCP Skill

Guide for AI agents (Claude, Cursor, Antigravity) interacting with the local-first Total Recall Model Context Protocol (MCP) server.

## Core Capabilities

Total Recall exposes three local read-only and staging tools:
1. `search_messages(query, limit)`: Fast FTS5 BM25 lexical search over indexed WhatsApp chats.
2. `get_thread_context(message_id)`: Reconstructs the full parent-child reply DAG, resolving thread root, branch participants, and conversation hierarchy.
3. `stage_reply_draft(chat_id, intent_points)`: Synthesizes a response matching the user's historical tone (`ZISFROMME = 1`) strictly bounded to user intent bullet points, and stages it in the local review box.

---

## Architectural Invariants & Safety Boundaries

- **No Autonomous Sending**: The server **cannot** send WhatsApp messages directly. All generated drafts are staged with status `STAGED_FOR_REVIEW` in `DraftReviewBox`. The user must manually inspect and clipboard-copy the text into WhatsApp.
- **No Presence Tracking**: Online status and presence snooping tools are omitted by design to prevent account bans and stalking.
- **Strict Intent Bounding**: Drafts must only state facts and commitments explicitly provided in `intent_points`. Never invent dates, prices, or promises.

---

## Recommended Execution Flow

### 1. Retrieve Historical Context
When asked about past conversations, decisions, or commitments:
```json
{
  "name": "search_messages",
  "arguments": {
    "query": "budget Q3 deployment",
    "limit": 10
  }
}
```

### 2. Inspect Reply Thread DAG
When context around a specific message or question is ambiguous or part of a multi-turn group debate:
```json
{
  "name": "get_thread_context",
  "arguments": {
    "message_id": 142
  }
}
```
Examine `root_id`, `rendered_tree`, and chronological `messages` to identify unanswered questions and the exact discussion thread.

### 3. Stage Reply Draft
When drafting a reply for the user:
1. Extract or prompt for concrete intent points from the user.
2. Invoke `stage_reply_draft`:
```json
{
  "name": "stage_reply_draft",
  "arguments": {
    "chat_id": "1234567890@s.whatsapp.net",
    "intent_points": [
      "Confirm receipt of the report",
      "Mention review will be finished by Thursday morning"
    ]
  }
}
```
3. Report the generated `draft_text`, `status`, and `tone_profile` back to the user with a clear prompt to copy it to their WhatsApp clipboard.
