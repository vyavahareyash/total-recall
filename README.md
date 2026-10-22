# Total Recall

> **A Zero-Cost, Privacy-First Conversational Intelligence Engine for WhatsApp (macOS)**

Total Recall is an edge-native conversational intelligence platform designed to synthesize, search, and retrieve information across personal WhatsApp chats, group discussions, and multi-tiered communities on macOS—with **zero cloud costs, zero network packet egress, and zero account ban risk**.

---

## Key Highlights

* **Zero Ban Risk & Zero Pairing Friction:** Passively reads local unencrypted CoreData databases (`ChatStorage.sqlite`) from the macOS WhatsApp Catalyst App Group container. No companion QR pairing, no Puppeteer/Selenium sessions, and no risk of automated anti-bot bans.
* **Lock-Free Concurrency:** Uses immutable read-only connections (`mode=ro&immutable=1`) and snapshot isolation to prevent SQLite write-lock contention with WhatsApp Desktop's active WAL writer, avoiding OS `0xDEAD10CC` crashes.
* **Conversational Flow Graph (CFG):** Reconstructs multi-turn reply DAGs (`ZPARENTMESSAGE`), attributes group senders (`ZWAGROUPMEMBER` + `ContactsV2.sqlite`), and chunks fragmented burst chatter into coherent 5–25 message conversational episodes.
* **Zero-Cost Local Hybrid Search:** Single-file local SQLite storage (`total_recall.db`) combining keyword search (`FTS5` BM25) and dense semantic vector search (`sqlite-vec` SIMD KNN), fused via Reciprocal Rank Fusion (RRF).
* **Cryptographic Privacy & Epistemic Neutrality:** Mandatory hard purging on message revocation (`PRAGMA secure_delete = ON`), Opt-In Chat Vault perimeters, court-reporter citation ledgers (`[Chat | Date | Sender | msg_id]`), and prefix-level NLI verification (`MiniTruePrefixes`).
* **Dual Delivery (Single Binary):** Runs as a standalone macOS menu-bar / tray application (Wails / Tauri) while concurrently serving as a local Model Context Protocol (MCP) server for Claude Desktop, Cursor, and local AI agents.
* **Phase 2 Multimodal Roadmap:** On-device CPU voice note transcription via quantized `faster-whisper` + Silero VAD.

---

## Deep Module Architecture

The system is engineered around four deep modules with minimal interfaces and high leverage:

| Module | Interface Surface | Seam & Implementation |
| :--- | :--- | :--- |
| **IngestionEngine** | `sync(vaultConfig)`, `watch(vaultConfig)` | Local-substitutable; reads CoreData App Group containers; normalizes Apple Epoch (+978,307,200s); filters 32-bit sentinels; builds reply DAGs. |
| **SearchIndex** | `index(episodes)`, `search(query, k)`, `purge(msgId)` | Local-substitutable; manages FTS5 virtual tables, `sqlite-vec` SIMD KNN, SQL RRF scoring, and `secure_delete` purges. |
| **Summarizer** | `summarize(episodes, intentNotes?)` | Real seam (Ollama adapter vs test mock); enforces citation ledgers, sentiment normalization, and prefix-level NLI verification. |
| **AudioTranscriber** *(Phase 2)* | `transcribe(audioFilePath)` | Real seam (faster-whisper adapter vs test fake); handles Opus decoding, Silero VAD, and quantized speech-to-text. |

---

## Documentation Navigation

* **[CONTEXT.md](CONTEXT.md):** Problem space, macOS Catalyst constraints, user personas, and project boundaries.
* **[GLOSSARY.md](GLOSSARY.md):** Canonical domain language, strict definitions, and `_Avoid_` rules.
* **[docs/concepts.md](docs/concepts.md):** Deep explanation of core concepts: episodes, reply DAGs, Apple Epoch, sentinels, RRF, and epistemic neutrality.
* **[docs/architecture.md](docs/architecture.md):** Detailed macOS architecture, deep modules, seams, SQL RRF formulas, and concurrency mechanics.
* **[docs/spec.md](docs/spec.md):** Comprehensive product specification, 25 user stories, implementation decisions, and test plan.
* **[docs/adr/0001-macos-app-group-sqlite-ingestion.md](docs/adr/0001-macos-app-group-sqlite-ingestion.md):** Architecture Decision Record establishing macOS Catalyst SQLite snapshot ingestion.
* **[docs/research/research.md](docs/research/research.md):** Upstream forensic, architectural, benchmark, and protocol analysis from the research paper.


---

## Prerequisites

* **Operating System:** macOS (Apple Silicon or Intel).
* **WhatsApp Desktop:** Official macOS desktop application (Catalyst build) installed and active.
* **Permissions:** macOS Full Disk Access granted to Total Recall to read the shared App Group container.
* **Local Inference:** [Ollama](https://ollama.com/) or `llama.cpp` running locally for dense vector embeddings (`nomic-embed-text`) and summarization models (`Qwen-2.5-7B` / `Llama-3.2-3B`).

---

## License

MIT License. See LICENSE for details.
