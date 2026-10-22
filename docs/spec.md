# Specification: macOS Native WhatsApp Conversational Intelligence Engine

## Problem Statement

As a WhatsApp power user, team lead, or active community participant on macOS, I communicate across dozens of personal chats, multi-tiered community channels, and high-velocity group threads. Because WhatsApp enforces End-to-End Encryption (E2EE) and offers no free personal REST or WebSocket API, finding critical past agreements, tracking commitments, and searching voice notes is nearly impossible without hours of manual scrollback. Furthermore, existing unofficial multi-device automation bridges require complex developer environments (Go, Python, compilers), require repeated QR re-pairing, and risk permanent account bans under Meta's automated anti-scraping heuristics. Cloud-hosted bot relays violate privacy by exposing decrypted correspondence to third-party infrastructure.

## Solution

A native, zero-cost, privacy-first desktop application for macOS that passively ingests local decrypted chat records directly from WhatsApp Desktop's Catalyst SQLite container (`ChatStorage.sqlite`) without network interception, QR code pairing, or account ban risk. The system reconstructs fragmented group chatter into coherent Conversational Flow Episodes via reply DAGs, indexes content into a local hybrid lexical-vector SQLite store (FTS5 BM25 + `sqlite-vec` SIMD KNN), and serves factual, citation-anchored thread summaries and staged reply drafts through a native UI and local authenticated MCP interface. Voice note transcription is planned for Phase 2.

## User Stories

1. As a macOS WhatsApp user, I want the application to automatically locate my local `ChatStorage.sqlite` container, so that I do not have to manually browse hidden library folders.
2. As a privacy-conscious user, I want onboarding to operate via standard macOS Full Disk Access permissions without scanning a QR code, so that my WhatsApp account is never exposed to companion session expiration or anti-bot account bans.
3. As a desktop user, I want the ingestion engine to access the database without write locks (`mode=ro&immutable=1`), so that the running WhatsApp Desktop application never crashes or triggers OS `0xDEAD10CC` termination kills.
4. As a user with an active WhatsApp account, I want the system to continuously detect new incoming messages via lock-free WAL polling, so that my search index remains up to date in near real-time.
5. As a user with multi-year chat archives, I want the importer to normalize Apple Epoch timestamps (+978,307,200 seconds) and filter 32-bit sentinel sort values, so that my chronological message history displays accurately.
6. As a group chat participant, I want multi-turn quotes and reply chains (`ZPARENTMESSAGE`) to be assembled into a Directed Acyclic Graph (DAG), so that replies separated by hours of intervening chatter remain logically linked.
7. As a community member, I want group sender IDs to be accurately attributed to human display names and address book contacts via `ZWAGROUPMEMBER` and `ContactsV2.sqlite`, so that I know exactly who made each statement.
8. As a member of structured WhatsApp Communities, I want queries targeting a community to automatically search across both the Announcement channel and linked discussion sub-groups, so that I do not have to search each group individually.
9. As a user receiving rapid message bursts, I want fragmented one-line messages within a 15-minute window to be grouped into unified Conversational Flow Episodes, so that semantic searches retrieve coherent thoughts rather than disconnected sentences.
10. As a user in 1:1 chats, I want conversations to be segmented using a 45-minute temporal burst threshold, so that distinct daily discussions are properly partitioned.
11. [Phase 2] As a user frequently receiving audio messages, I want `.ogg` Opus voice notes to be automatically transcribed on my local CPU using `faster-whisper`, so that spoken agreements are searchable alongside text without incurring cloud API costs.
12. [Phase 2] As a user listening to long voice notes, I want Voice Activity Detection (Silero VAD) to strip leading and trailing silence, so that transcription processing finishes rapidly without wasting CPU cycles.
13. As a user searching for specific alphanumeric codes, contact numbers, or jargon, I want SQLite FTS5 BM25 lexical search, so that exact terms are matched immediately.
14. As a user asking conceptual questions in natural language, I want `sqlite-vec` SIMD vector similarity search, so that relevant discussions are retrieved even when exact words differ.
15. As a user querying diverse chat histories, I want Reciprocal Rank Fusion (RRF) to merge keyword and vector search results into a single ranked list, so that I get the highest relevance without tuning score thresholds.
16. As a privacy-focused user, I want an Opt-In Chat Vault selector on initial startup with zero chats indexed by default, so that sensitive medical, financial, or intimate 1:1 chats are never indexed without my explicit consent.
17. As an ethical communicator, I want the system to execute a hard cryptographic purge (`PRAGMA secure_delete = ON`) whenever a message is deleted for everyone, so that retracted mistakes can never be searched or recovered.
18. As a user respecting disappearing message policies, I want ephemeral chats to enforce TTL expiration on indexed chunks, so that temporary messages are purged when their timer expires.
19. As a user querying contentious group debates, I want summaries generated with near-zero temperature and extractive citation ledgers (`[Chat | Date | Sender | msg_id]`), so that the AI reports verifiable facts rather than hallucinated interpretations.
20. As a user reviewing contentious discussions, I want sentiment normalization applied to summaries, so that emotionally loaded arguments are reported neutrally without taking sides.
21. As a user seeking reliable summaries, I want a local prefix-level NLI verifier (`MiniTruePrefixes`) to discard generated sentences that lack source entailment, so that factual drift is prevented.
22. As a busy professional, I want the system to draft contextual replies matching my historical tone (`ZISFROMME = 1`) based on explicit bullet points I provide, so that I can respond quickly while maintaining appropriate formality.
23. As a user who values communication control, I want all drafted replies staged in a UI review box requiring manual approval before sending, so that an autonomous AI agent never speaks on my behalf.
24. As an AI power user, I want a local MCP server bound to `127.0.0.1` with bearer-token security, so that I can safely connect Claude Desktop or Cursor to my local WhatsApp index.
25. As a laptop user, I want the entire application (daemon, index, transcription, and UI) to run self-contained on my Mac without Docker or cloud subscriptions, so that the solution remains truly zero-cost and lightweight.

## Implementation Decisions

### Architectural Seams & Modules
The architecture separates core functionality into Phase 1 (text ingestion, hybrid retrieval, and summarization) and Phase 2 (voice note transcription):

#### Phase 1: Core Modules
1. **IngestionEngine (Deep Module):**
   * *Role:* Introspects the macOS WhatsApp Catalyst container (`ChatStorage.sqlite`, `ContactsV2.sqlite`), enforces lock-free read-only snapshot isolation (`mode=ro&immutable=1`), converts Apple Epoch timestamps to Unix Epoch (+978,307,200s), strips 32-bit sentinel clusters, resolves sender identities, and constructs Conversational Flow Episodes via reply DAGs and temporal burst segmentation.
   * *Interface:* Small, high-leverage interface: `sync(vaultConfig: VaultConfig): Promise<SyncResult>` and `watch(vaultConfig: VaultConfig): AsyncIterable<EpisodeEvent>`.
   * *Seam:* Local-substitutable store. Production reads macOS App Group containers; test suite uses SQLite fixture databases.

2. **SearchIndex (Deep Module):**
   * *Role:* Manages a single local SQLite database (`total_recall.db`) combining an FTS5 BM25 virtual table, `sqlite-vec` SIMD KNN extension, and a unified Reciprocal Rank Fusion (RRF) query. Implements immediate cascading deletion and `PRAGMA secure_delete = ON` upon message revocation.
   * *Interface:* `index(episodes: ConversationalFlowEpisode[]): Promise<void>`, `search(query: string, limit: number): Promise<ScoredEpisode[]>`, and `purge(messageId: string): Promise<PurgeReport>`.
   * *Seam:* Local-substitutable SQLite database.

3. **Summarizer (Deep Module):**
   * *Role:* Generates extractive, court-reporter style summaries anchored by verbatim citations (`[Chat | Date | Sender | msg_id]`), executes local LLM inference (`Ollama` / `llama.cpp` bound to `127.0.0.1`), validates claims with prefix-level NLI verification (`MiniTruePrefixes`), and stages intent-constrained reply drafts.
   * *Interface:* `summarize(episodes: ConversationalFlowEpisode[], intentNotes?: string): Promise<VerifiedSummary>`.
   * *Seam:* Real seam with two adapters: production `LocalOllamaAdapter` and test `DeterministicMockLLMAdapter`.

#### Phase 2: Audio Module
4. **AudioTranscriber (Deep Module at a Seam - Phase 2):**
   * *Role:* Converts `.ogg` Opus streams to 16 kHz WAV, runs Silero VAD to strip silence, and executes `faster-whisper` (`small-int8` / `medium-int8`) locally on CPU/Apple Silicon.
   * *Interface:* `transcribe(audioFilePath: string): Promise<AudioTranscript>`.
   * *Seam:* Real seam with two adapters: production `FasterWhisperAdapter` (heavy inference) and test `FakeAudioTranscriber` (instant deterministic transcript).

### Domain Alignment & Vocabulary
* Adheres strictly to the canonical definitions in `GLOSSARY.md` (JID, LID, Canonical Contact, ChatStorage, Snapshot, Apple Epoch, Sentinel Cluster, Conversational Flow Episode, Reply DAG, Hybrid Retrieval, Opt-In Chat Vault, Cryptographic Purge, Epistemic Neutrality).
* Follows `docs/adr/0001-macos-app-group-sqlite-ingestion.md` designating passive macOS CoreData SQLite snapshot ingestion as the sole ingestion mechanism.

### Security and Concurrency Constraints
* Pure edge operation: All daemon listeners and MCP transports bind strictly to `127.0.0.1` requiring bearer-token authentication. Zero network packets are transmitted to Meta or cloud servers.
* Concurrency protection: The live `ChatStorage.sqlite` container file is never opened in read-write mode.

## Testing Decisions

### What Makes a Good Test
* Tests must exercise observable behavior across module interfaces, never asserting on private internal state, intermediate table joins, or raw SQL queries.
* Tests must survive internal schema refactorings as long as observable outcomes through the interface remain consistent.
* Fast, deterministic execution: No network dependencies, no external Docker containers, and no prolonged CPU ML inference during regular unit test runs.

### Module Test Plan

#### Phase 1 Tests
1. **IngestionEngine Tests:** Driven through `sync()` using synthetic SQLite fixtures containing sample `ZWAMESSAGE`, `ZWAGROUPMEMBER`, and `ContactsV2` tables. Verifies:
   * Unix timestamp normalization (+978,307,200s offset).
   * Filtering of sentinel values in `ZSORT` and `ZMESSAGECOUNTER`.
   * Assembly of multi-turn reply DAGs into coherent episodes.
   * Correct filtering according to `VaultConfig`.
2. **SearchIndex Tests:** Tested directly through `index()`, `search()`, and `purge()` on temporary SQLite instances. Verifies:
   * Correct RRF merging of exact keyword hits and vector semantic matches.
   * Irreversible removal of revoked messages after `purge()` verification.
3. **Summarizer Tests:** Tested through `summarize()` using `DeterministicMockLLMAdapter`. Verifies:
   * Citation ledger formatting.
   * Stripping of unentailed claim sentences via the prefix NLI verification step.

#### Phase 2 Tests
4. **AudioTranscriber Tests (Phase 2):** Tested across the `transcribe()` seam using `FakeAudioTranscriber` in the test suite, with isolated integration tests verifying Opus conversion and quantized inference on sample audio files.

### Prior Art
* Patterns adapted from `openclaw/wacrawl` test fixtures for CoreData Apple Epoch decoding.
* Reciprocal Rank Fusion tests adapted from Simon Willison's and sqlite-vec hybrid search specifications.

## Out of Scope

* Support for Windows DPAPI-NG / UEFI key decryption or Android ADB / root extraction.
* Unofficial active companion Web protocol connections (`whatsmeow` / `Baileys`) for automated background sending or QR-based session pairing.
* Multi-tenant or cloud-hosted database deployments (Qdrant, OpenSearch, Pinecone).
* Autonomous background message sending without explicit human-in-the-loop review.
* Contact presence tracking, online status logging, or psychological profiling.

## Further Notes

* Target desktop framework: Lightweight native wrapper built with Go + Wails or Rust + Tauri, packaged as a `.dmg` with automated Homebrew installation via GoReleaser.
* Local Model Requirements: Embeddings via `nomic-embed-text` (or local ONNX runtime); LLM inference via local Ollama (`Qwen-2.5-7B-Instruct` or `Llama-3.2-3B`).
