# WhatsApp Conversational Intelligence (macOS)

A privacy-first, zero-cost conversational intelligence engine running natively on macOS to index, search, and synthesize local WhatsApp desktop communication history without network interception or ban risk.

## Language

### Messaging & Addressing

**JID (Jabber ID)**:
The protocol-level address identifier in WhatsApp, formatted as `<phone_number>@s.whatsapp.net` for individuals and `<id>@g.us` for group chats.
_Avoid_: Phone number, user ID, WhatsApp handle

**LID (Linked Identity)**:
An opaque, privacy-preserving identifier (`<id>@lid`) assigned to devices and community members to decouple message routing from visible phone numbers.
_Avoid_: Device token, alias ID, internal JID

**Canonical Contact**:
The unified internal identity representing a single human contact, consolidating all mapped JIDs and LIDs under one persistent UUID.
_Avoid_: User profile, address book entry

### Ingestion & Storage

**ChatStorage**:
The unencrypted SQLite database maintained by the macOS WhatsApp Catalyst desktop client inside its App Group container.
_Avoid_: Message store, local cache, WhatsApp database

**Snapshot**:
An immutable, isolated copy or read-only connection (`mode=ro&immutable=1`) of `ChatStorage.sqlite` and its WAL/SHM sidecars used to query messages without locking the live client.
_Avoid_: Backup, export, dump

**Apple Epoch**:
The timestamp convention starting on January 1, 2001 00:00:00 UTC used in macOS/iOS CoreData timestamps, requiring an offset of 978,307,200 seconds to align with Unix Epoch time.
_Avoid_: Mac time, CoreData timestamp

**Sentinel Cluster**:
Artificially large integer values (near `INT32_MAX`) written by WhatsApp into internal sort columns (`ZSORT`, `ZMESSAGECOUNTER`) that must be excluded during chronological sorting.
_Avoid_: Sort bug, overflow value, corrupt record

### Graph & Retrieval

**Conversational Flow Episode**:
A coherent conversational turn consisting of 5 to 25 temporally linked messages, stitched across burst gaps via parent reply pointers and enriched with structured participant metadata.
_Avoid_: Chunk, document, window, conversation snippet

**Reply DAG**:
A directed acyclic graph capturing explicit quote-reply relationships between messages via parent pointers (`ZPARENTMESSAGE`), preserving multi-threaded debates within high-traffic groups.
_Avoid_: Thread tree, quote hierarchy, reply chain

**Hybrid Retrieval**:
A search pipeline combining lexical keyword ranking (SQLite FTS5 BM25) and dense semantic vector similarity (sqlite-vec SIMD KNN) merged via Reciprocal Rank Fusion (RRF).
_Avoid_: Mixed search, ensemble retrieval, two-stage search

### Privacy & Governance

**Opt-In Chat Vault**:
A strictly selective indexing perimeter that requires the user to manually designate which chats, groups, and communities are ingested into the searchable index.
_Avoid_: Filter list, whitelist, index scope

**Cryptographic Purge**:
The immediate and irreversible deletion of retracted messages (`ProtocolMessage_REVOKE` / cleared `ZTEXT`) from raw tables, FTS5 tokens, and vector embeddings, finalized via `PRAGMA secure_delete`.
_Avoid_: Soft delete, tombstone, message hiding

**Epistemic Neutrality**:
The architectural constraint requiring summaries and reply drafts to remain strictly grounded in verbatim quotes without emotional projection, unstated intent inference, or artificial consensus.
_Avoid_: Unbiased mode, objective tone, moderation
