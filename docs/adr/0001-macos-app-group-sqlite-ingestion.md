# 0001: Use macOS App Group SQLite Snapshot Ingestion

We will implement the initial conversational intelligence engine exclusively for macOS by passively reading the local unencrypted WhatsApp Catalyst `ChatStorage.sqlite` database in the macOS App Group container, rather than using multi-device WebSocket bridges or reverse-engineering Windows/mobile storage.

## Context

A zero-cost personal intelligence tool for WhatsApp must access historical and real-time messages across private chats, active groups, and multi-tiered communities. WhatsApp protects messages in transit with Signal Protocol E2EE and provides no free user-level REST/WebSocket interfaces. Building cross-platform active bridges (such as `whatsmeow` or `Baileys`) requires QR pairing, triggers frequent breaking protocol changes, and introduces account ban risks under Meta's automated anti-scraping heuristics. On Windows, local SQLite stores are protected by hardware-bound DPAPI-NG encryption; on Android, databases are inaccessible without root or intrusive ADB setups.

Conversely, WhatsApp Desktop on macOS (built on Apple's Catalyst framework) stores decrypted chat history, contact records, and media metadata in an unencrypted CoreData SQLite database located in `~/Library/Group Containers/group.net.whatsapp.WhatsApp.shared/ChatStorage.sqlite`.

## Decision

We will build the ingestion pipeline around passive, read-only snapshot ingestion of macOS `ChatStorage.sqlite`:

1. **Passive Filesystem Ingestion:** The engine will request macOS Full Disk Access or container permissions to read `ChatStorage.sqlite` and `ContactsV2.sqlite` directly.
2. **Lock-Free Concurrency:** To avoid write contention and prevent OS `0xDEAD10CC` termination of the active WhatsApp process, the engine will query via SQLite immutable URI flags (`mode=ro&immutable=1`) or isolated temporary snapshots.
3. **No Active Network Bridge:** The engine will execute zero network requests to WhatsApp servers, completely eliminating account suspension and ban risks.

## Considered Options

* **Companion Multi-Device Protocol (`whatsmeow` / `Baileys`):** Rejected for default ingestion due to pairing friction (QR scan expiration every ~20 days), `@lid` split-brain mapping instability, protocol breakage, and account ban risks when performing large history syncs.
* **Windows Desktop Decryption (`ZAPiXDESK` / DOM injection):** Rejected due to the high fragility of DPAPI-NG / UEFI key recovery, breakage across app updates, and DOM injection limitations that require manual viewport scrollback.
* **Android ADB / Notification Listeners:** Rejected because ADB requires developer setup and USB debugging, while notification listeners miss outgoing replies, media files, and historical threads.
* **Official Meta WhatsApp Cloud API:** Rejected because it cannot access personal or group chats and charges commercial per-conversation fees.

## Consequences

* **Zero Ban Risk & Zero Pairing Friction:** Users on macOS do not need to scan QR codes or maintain active companion connections; data is indexed directly from disk.
* **macOS Desktop Prerequisite:** The user must have WhatsApp Desktop installed and running on macOS to receive live updates. When WhatsApp Desktop is closed, the local database does not receive new messages until reopened.
* **Platform Limitation:** Windows and Linux users cannot run this architecture without a separate companion bridge fallback.
* **Schema Coupling:** The ingestion layer is coupled to WhatsApp's internal Apple CoreData schema (`ZWAMESSAGE`, `ZWACHATSESSION`, `ZPARENTMESSAGE`, `ZWAGROUPMEMBER`), requiring timestamp normalization (Apple Epoch offset of `978,307,200` seconds) and filtering of 32-bit sentinel sort values.
