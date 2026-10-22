# System Context & Problem Domain

## 1. Executive Context

WhatsApp is the primary communications channel for billions of individuals, distributed teams, and community organizations. However, WhatsApp's desktop and mobile applications lack conversational synthesis capabilities:
* **Fragmented Communication:** Messages in multi-participant groups are split across short asynchronous bursts, interleaved across unrelated topics, and augmented by voice notes that lack native global searchability.
* **Walled Garden & E2EE:** WhatsApp protects communication using Signal Protocol End-to-End Encryption (E2EE). There is no official free user API to query personal chat archives. The official Cloud API requires enterprise onboarding, charges per conversation, and cannot read personal group discussions.
* **Account Ban Risk:** Unofficial companion Web protocol bridges (`whatsmeow`, `Baileys`) emulate linked browser sessions over WebSockets, leaving users vulnerable to anti-bot detection and account bans when handling automated actions or aggressive backfills.

## 2. Target Platform & Scope: macOS Desktop

This repository targets **macOS** as its primary host environment:
* **Catalyst Architecture Discovery:** WhatsApp Desktop on macOS uses Apple's Catalyst framework, storing fully decrypted messages, contacts, and media records locally in an unencrypted CoreData SQLite database within the user's App Group container.
* **Passive Ingestion Opportunity:** By reading the local `ChatStorage.sqlite` database in read-only snapshot mode, the system achieves instant historical and live access across millions of messages with **zero network traffic, zero QR code pairing, and zero account ban risk**.

### In Scope
1. **Passive CoreData Ingestion:** Ingesting `ChatStorage.sqlite`, `ContactsV2.sqlite`, and media references from `~/Library/Group Containers/group.net.whatsapp.WhatsApp.shared`.
2. **Snapshot & Lock Management:** Eliminating SQLite write contention and avoiding `0xDEAD10CC` termination crashes via immutable URI reads (`mode=ro&immutable=1`) and snapshot isolation.
3. **Conversational Flow Reconstruction:** Rebuilding Reply DAGs (`ZPARENTMESSAGE`), group sender attribution, community hierarchies, and burst segmentation ($\Delta t > 15\text{m} / 45\text{m}$).
4. **On-Device Multimodal Processing (Phase 2):** Local CPU transcription of `.ogg` Opus audio notes using `faster-whisper` + Silero VAD.
5. **Zero-Cost Hybrid Retrieval:** Local single-file hybrid search combining SQLite `FTS5` (BM25) and `sqlite-vec` (SIMD KNN) merged via Reciprocal Rank Fusion (RRF).
6. **Strict Privacy Architecture:** Localhost binding (`127.0.0.1`), hard cryptographic purge of revoked messages (`PRAGMA secure_delete`), and Opt-In Chat Vault perimeters.
7. **Epistemic Neutrality:** Extractive quotation ledgers, sentiment normalization, and Prefix-Level NLI validation for summaries and reply drafts.

### Out of Scope
1. **Windows / Android Native Filesystem Ingestion:** Windows DPAPI-NG / UEFI key extraction and Android root/ADB extraction are excluded due to fragile reverse-engineering dependencies and unacceptable onboarding friction.
2. **Cloud-Hosted Middleware & SaaS Relays:** No unencrypted chat payloads or embeddings are permitted to leave the local device seam.
3. **Autonomous Outbound Sending:** Autonomous background bot messaging is strictly prohibited to avoid anti-spam triggers and interpersonal misrepresentation.

## 3. User Personas & Use Cases

* **Power Communicator / Community Member:** Needs to retrieve decisions, action items, recommendations, and shared documents across hundreds of chat threads and voice notes without scrolling endlessly.
* **Privacy-Conscious Professional:** Demands complete data sovereignty where personal and professional correspondence remains on the local SSD, protected from third-party analytics and cloud leakage.
* **Local AI Practitioner:** Utilizes desktop AI agents (e.g., Claude Desktop, Cursor, local Ollama) to query personal WhatsApp context through a local, token-authenticated interface.
