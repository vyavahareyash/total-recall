from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from collections.abc import Sequence
from pathlib import Path

from total_recall.ingestion import format_iso_timestamp
from total_recall.summarizer import (
    DeterministicMockLLMAdapter,
    LLMAdapter,
)
from total_recall.types import (
    DraftStatus,
    IntentVerificationResult,
    NormalizedMessage,
    StagedDraft,
    ToneFormality,
    ToneProfile,
    UnansweredQuestion,
)


class AutonomousSendingProhibitedError(RuntimeError):
    """Raised when any autonomous transmission or send action is attempted."""


# Interrogative keywords and patterns
QUESTION_START_REGEX = re.compile(
    r"\b(can you|could you|would you|will you|should we|shall we|are you|is it|is there|"
    r"do you|did you|have you|has anyone|who|what|when|where|why|how)\b",
    re.IGNORECASE,
)

# Common greeting patterns
GREETING_REGEX = re.compile(
    r"^(hey|hi|hello|dear|good morning|good afternoon|good evening|morning|yo)\b[,\s]*([a-zA-Z]+)?",
    re.IGNORECASE,
)

# Common sign-off patterns
SIGNOFF_REGEX = re.compile(
    r"(?:^|\n+)(thanks|thank you|best|best regards|cheers|regards|sincerely|warmly)\b[,\s]*([a-zA-Z]+)?$",
    re.IGNORECASE,
)

# Casual markers
CASUAL_CONTRACTIONS_REGEX = re.compile(
    r"\b(i'm|i'll|i've|i'd|can't|won't|don't|doesn't|didn't|it's|that's|we're|they're|you're|gonna|wanna)\b",
    re.IGNORECASE,
)
CASUAL_SLANG_REGEX = re.compile(
    r"\b(hey|yo|sup|cool|yeah|yep|nope|lol|haha|ok|okay)\b",
    re.IGNORECASE,
)

# Commitment / promise indicators
COMMITMENT_REGEX = re.compile(
    r"\b(promise|guarantee|definitely will|will make sure|can assure|"
    r"i will|i'll|i can|i shall|let me|we will|we'll|"
    r"let's meet|let's call|let's sync|i'll call|i will call|see you at|schedule a)\b",
    re.IGNORECASE,
)

TEMPORAL_DEADLINE_REGEX = re.compile(
    r"\b(by tomorrow|by friday|by monday|by tuesday|by wednesday|by thursday|by saturday|by sunday|"
    r"by next week|at \d+(?::\d+)?\s*(?:am|pm)?|tonight|later today)\b",
    re.IGNORECASE,
)


def detect_unanswered_questions(
    messages: Sequence[NormalizedMessage],
) -> list[UnansweredQuestion]:
    """Identify questions from external contacts in the thread that have not received a user response.

    A message is considered an unanswered question if:
    1. It is incoming from another sender (is_from_me = False).
    2. It contains a question mark '?' or interrogative phrasing.
    3. No subsequent message from the user (is_from_me = True) exists in the thread.
    """
    if not messages:
        return []

    unanswered: list[UnansweredQuestion] = []

    # Map message parent_pk replies to check if user explicitly replied to a parent
    user_replied_pks: set[int] = {
        m.parent_pk for m in messages if m.is_from_me and m.parent_pk is not None
    }

    # Find the latest timestamp of a user message
    user_timestamps = [m.unix_timestamp for m in messages if m.is_from_me]
    latest_user_ts = max(user_timestamps) if user_timestamps else -1.0

    for msg in messages:
        if msg.is_from_me:
            continue

        text = msg.text.strip()
        if not text:
            continue

        # Check for question marks or interrogative clauses
        is_question = "?" in text or bool(QUESTION_START_REGEX.search(text))
        if not is_question:
            continue

        # If user explicitly replied to this message via DAG parent_pk, it's answered
        if msg.pk in user_replied_pks:
            continue

        # If a user message appeared chronologically after this question, consider it addressed
        if msg.unix_timestamp < latest_user_ts:
            continue

        # Extract individual question sentences if multiple sentences exist
        sentences = re.split(r"(?<=[.?!])\s+", text)
        q_sentences = [
            s.strip() for s in sentences if "?" in s or QUESTION_START_REGEX.search(s)
        ]
        question_text = " ".join(q_sentences) if q_sentences else text

        unanswered.append(
            UnansweredQuestion(
                message_pk=msg.pk,
                sender_name=msg.sender_name,
                iso_date=msg.iso_date,
                text=text,
                question_text=question_text,
            )
        )

    return unanswered


class ToneAnalyzer:
    """Extracts surface tone conventions, greetings, and sign-offs from recent outgoing messages."""

    def analyze(self, outgoing_messages: Sequence[NormalizedMessage]) -> ToneProfile:
        """Analyze 5–10 recent outgoing user messages (is_from_me = 1) from the chat."""
        if not outgoing_messages:
            return ToneProfile(
                formality=ToneFormality.NEUTRAL,
                typical_greeting=None,
                typical_signoff=None,
                use_contractions=True,
                sample_count=0,
                exemplars=(),
            )

        casual_score = 0
        formal_score = 0
        greetings_found: list[str] = []
        signoffs_found: list[str] = []
        contractions_detected = False
        exemplars: list[str] = []

        for msg in outgoing_messages:
            text = msg.text.strip()
            if not text:
                continue
            exemplars.append(text)

            # Check greetings
            greet_match = GREETING_REGEX.match(text)
            if greet_match:
                greetings_found.append(greet_match.group(1).capitalize())

            # Check sign-offs
            signoff_match = SIGNOFF_REGEX.search(text)
            if signoff_match:
                signoffs_found.append(signoff_match.group(1).capitalize())

            # Check contractions
            if CASUAL_CONTRACTIONS_REGEX.search(text):
                contractions_detected = True
                casual_score += 2

            # Check slang & casual tokens
            if CASUAL_SLANG_REGEX.search(text):
                casual_score += 2

            # Check sentence casing & punctuation
            if text and text[0].islower():
                casual_score += 1
            if "!" in text:
                casual_score += 1
            if "..." in text:
                casual_score += 1

            # Formal markers
            if any(
                f in text.lower()
                for f in ("dear", "sincerely", "regards", "best regards", "thank you")
            ):
                formal_score += 2

            # Ending in period with standard capitalization
            if (
                text.endswith(".")
                and text[0].isupper()
                and not CASUAL_CONTRACTIONS_REGEX.search(text)
            ):
                formal_score += 1

        # Determine typical greeting
        typical_greeting: str | None = None
        if greetings_found and (len(greetings_found) / len(exemplars)) >= 0.25:
            # Most common greeting
            typical_greeting = max(set(greetings_found), key=greetings_found.count)

        # Determine typical sign-off
        typical_signoff: str | None = None
        if signoffs_found and (len(signoffs_found) / len(exemplars)) >= 0.25:
            # Most common sign-off
            typical_signoff = max(set(signoffs_found), key=signoffs_found.count)

        # Formality classification
        if casual_score > formal_score:
            formality = ToneFormality.CASUAL
        elif formal_score > casual_score:
            formality = ToneFormality.FORMAL
        else:
            formality = ToneFormality.NEUTRAL

        return ToneProfile(
            formality=formality,
            typical_greeting=typical_greeting,
            typical_signoff=typical_signoff,
            use_contractions=contractions_detected
            or (formality == ToneFormality.CASUAL),
            sample_count=len(exemplars),
            exemplars=tuple(exemplars[-10:]),
        )


class IntentVerifier:
    """Verifies that generated drafts do not exceed supplied user intent bullet points."""

    def verify(
        self,
        draft_text: str,
        intent_bullets: Sequence[str],
        question_context: Sequence[UnansweredQuestion] | None = None,
    ) -> IntentVerificationResult:
        """Verify draft sentences against intent bullets. Detects unrequested commitments."""
        violations: list[str] = []
        verified_clauses: list[str] = []

        if not intent_bullets:
            return IntentVerificationResult(
                is_valid=False,
                violations=("NO_INTENT_POINTS_PROVIDED",),
                verified_clauses=(),
                grounding_score=0.0,
            )

        # Clean bullets for comparison
        clean_bullets = [
            re.sub(r"^[-*•\d.]+\s*", "", b).strip().lower() for b in intent_bullets
        ]
        bullet_text_combined = " ".join(clean_bullets)
        bullet_words = set(re.findall(r"\b[a-zA-Z0-9]+\b", bullet_text_combined))

        # Add context words from questions
        q_text_combined = ""
        if question_context:
            q_text_combined = " ".join(
                q.question_text.lower() for q in question_context
            )
            bullet_words |= set(re.findall(r"\b[a-zA-Z0-9]+\b", q_text_combined))

        # Strip greetings and sign-offs for substantive content verification
        substantive_text = GREETING_REGEX.sub("", draft_text).strip()
        substantive_text = SIGNOFF_REGEX.sub("", substantive_text).strip()

        # Split into individual sentences or clauses
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+|\n+", substantive_text)
            if s.strip()
        ]

        # 1. Check for unrequested commitments or promises
        for s in sentences:
            s_lower = s.lower()
            commit_match = COMMITMENT_REGEX.search(s_lower)
            if commit_match:
                # Find the commitment verb / predicate
                verb_part = s_lower[commit_match.start() :]
                # Check if this commitment predicate exists in intent bullets
                verb_words = [
                    w
                    for w in re.findall(r"\b[a-z]{3,}\b", verb_part)
                    if w
                    not in ("will", "can", "shall", "let", "the", "and", "you", "that")
                ]
                grounded = any(vw in bullet_words for vw in verb_words)
                if not grounded:
                    violations.append(
                        f"UNREQUESTED_COMMITMENT: '{commit_match.group(0)}' in '{s}'"
                    )
                    continue

            # 2. Check for ungrounded temporal deadlines
            deadline_match = TEMPORAL_DEADLINE_REGEX.search(s_lower)
            if deadline_match:
                dl = deadline_match.group(0)
                dl_words = set(re.findall(r"\b[a-z0-9]+\b", dl))
                if not (dl_words & bullet_words):
                    violations.append(f"UNGROUNDED_DEADLINE: '{dl}' in '{s}'")
                    continue

            # 3. Check for ungrounded numbers
            draft_nums = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", s_lower))
            allowed_nums = set(
                re.findall(
                    r"\b\d+(?:\.\d+)?%?\b", bullet_text_combined + " " + q_text_combined
                )
            )
            ungrounded_nums = draft_nums - allowed_nums
            if ungrounded_nums:
                violations.append(
                    f"UNGROUNDED_NUMBERS: {', '.join(sorted(ungrounded_nums))} in '{s}'"
                )
                continue

            # Check general semantic grounding of sentence
            s_words = {
                w
                for w in re.findall(r"\b[a-zA-Z]{3,}\b", s_lower)
                if w
                not in ("the", "this", "that", "with", "from", "for", "have", "been")
            }
            if s_words:
                overlap = s_words & bullet_words
                overlap_ratio = len(overlap) / len(s_words)
                if overlap_ratio < 0.15 and len(s_words) >= 4:
                    violations.append(f"UNGROUNDED_CONTENT: '{s}'")
                    continue

            verified_clauses.append(s)

        total_clauses = len(sentences)
        score = len(verified_clauses) / total_clauses if total_clauses > 0 else 1.0
        is_valid = len(violations) == 0

        return IntentVerificationResult(
            is_valid=is_valid,
            violations=tuple(violations),
            verified_clauses=tuple(verified_clauses),
            grounding_score=score if is_valid else 0.0,
        )


class DraftReviewBox:
    """Airgapped staging box for human review. Structural prohibition against sending."""

    def __init__(self, db_path: Path | str = ":memory:"):
        self.db_path = str(db_path) if isinstance(db_path, Path) else db_path
        self.con = sqlite3.connect(self.db_path)
        self.con.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        """Create staged_drafts table with strict is_sent = 0 check constraint."""
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS staged_drafts (
                draft_id TEXT PRIMARY KEY,
                chat_jid TEXT NOT NULL,
                created_at REAL NOT NULL,
                iso_date TEXT NOT NULL,
                thread_root_pk INTEGER,
                unanswered_questions_json TEXT NOT NULL,
                intent_bullets_json TEXT NOT NULL,
                tone_summary TEXT NOT NULL,
                draft_text TEXT NOT NULL,
                is_sent INTEGER NOT NULL DEFAULT 0 CHECK(is_sent = 0)
            );
            """
        )
        # Migration: drop legacy status column if present and purge rejected drafts
        cols = [
            r[1]
            for r in self.con.execute("PRAGMA table_info(staged_drafts)").fetchall()
        ]
        if "status" in cols:
            self.con.execute(
                "DELETE FROM staged_drafts WHERE status = 'REJECTED_BY_USER'"
            )
            self.con.execute("ALTER TABLE staged_drafts DROP COLUMN status")
        self.con.commit()

    def stage(self, draft: StagedDraft) -> StagedDraft:
        """Stage a generated draft for manual human review."""
        uq_data = [
            {
                "pk": u.message_pk,
                "sender": u.sender_name,
                "date": u.iso_date,
                "text": u.text,
                "question": u.question_text,
            }
            for u in draft.unanswered_questions
        ]
        tone_data = {
            "formality": draft.tone_profile.formality.value,
            "greeting": draft.tone_profile.typical_greeting,
            "signoff": draft.tone_profile.typical_signoff,
            "use_contractions": draft.tone_profile.use_contractions,
            "sample_count": draft.tone_profile.sample_count,
        }

        cur = self.con.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO staged_drafts (
                draft_id, chat_jid, created_at, iso_date, thread_root_pk,
                unanswered_questions_json, intent_bullets_json, tone_summary,
                draft_text, is_sent
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                draft.draft_id,
                draft.chat_jid,
                draft.created_at,
                draft.iso_date,
                draft.thread_root_pk,
                json.dumps(uq_data),
                json.dumps(list(draft.intent_bullets)),
                json.dumps(tone_data),
                draft.draft_text,
            ),
        )
        self.con.commit()
        return draft

    def get(self, draft_id: str) -> StagedDraft | None:
        """Retrieve a staged draft by ID."""
        cur = self.con.cursor()
        cur.execute("SELECT * FROM staged_drafts WHERE draft_id = ?", (draft_id,))
        row = cur.fetchone()
        if not row:
            return None
        return self._row_to_staged_draft(row)

    def list_staged(self, chat_jid: str | None = None) -> list[StagedDraft]:
        """List drafts currently staged for review."""
        cur = self.con.cursor()
        if chat_jid:
            cur.execute(
                "SELECT * FROM staged_drafts WHERE chat_jid = ? ORDER BY created_at DESC",
                (chat_jid,),
            )
        else:
            cur.execute("SELECT * FROM staged_drafts ORDER BY created_at DESC")
        return [self._row_to_staged_draft(r) for r in cur.fetchall()]

    def approve(self, draft_id: str) -> StagedDraft:
        """Approve draft (no-op since status tracking is removed)."""
        draft = self.get(draft_id)
        if not draft:
            raise KeyError(f"Draft '{draft_id}' not found.")
        return draft

    def reject(self, draft_id: str) -> StagedDraft:
        """Reject draft by deleting it from staged drafts."""
        draft = self.get(draft_id)
        if not draft:
            raise KeyError(f"Draft '{draft_id}' not found.")
        self.delete(draft_id)
        return draft

    def delete(self, draft_id: str) -> bool:
        """Permanently delete a staged draft."""
        cur = self.con.cursor()
        cur.execute("DELETE FROM staged_drafts WHERE draft_id = ?", (draft_id,))
        self.con.commit()
        return cur.rowcount > 0

    def send(self, draft_id: str) -> None:
        """Structural prohibition against autonomous sending."""
        raise AutonomousSendingProhibitedError(
            "Autonomous sending is structurally prohibited by system architecture. "
            "Drafts are staged strictly for manual human approval and clipboard action."
        )

    def _row_to_staged_draft(self, row: sqlite3.Row) -> StagedDraft:
        uq_list = json.loads(row["unanswered_questions_json"])
        unanswered = tuple(
            UnansweredQuestion(
                message_pk=u["pk"],
                sender_name=u["sender"],
                iso_date=u["date"],
                text=u["text"],
                question_text=u["question"],
            )
            for u in uq_list
        )
        tone_data = json.loads(row["tone_summary"])
        tone_profile = ToneProfile(
            formality=ToneFormality(tone_data.get("formality", "neutral")),
            typical_greeting=tone_data.get("greeting"),
            typical_signoff=tone_data.get("signoff"),
            use_contractions=tone_data.get("use_contractions", True),
            sample_count=tone_data.get("sample_count", 0),
            exemplars=(),
        )
        row_keys = row.keys()
        raw_status = row["status"] if "status" in row_keys and row["status"] else None
        status = (
            DraftStatus(raw_status) if raw_status else DraftStatus.STAGED_FOR_REVIEW
        )
        return StagedDraft(
            draft_id=row["draft_id"],
            chat_jid=row["chat_jid"],
            created_at=row["created_at"],
            iso_date=row["iso_date"],
            thread_root_pk=row["thread_root_pk"],
            unanswered_questions=unanswered,
            intent_bullets=tuple(json.loads(row["intent_bullets_json"])),
            tone_profile=tone_profile,
            draft_text=row["draft_text"],
            status=status,
        )

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> DraftReviewBox:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class ReplyDrafter:
    """Context-aware reply drafting engine bounded by user intent points and surface tone matching."""

    def __init__(
        self,
        llm: LLMAdapter | None = None,
        tone_analyzer: ToneAnalyzer | None = None,
        intent_verifier: IntentVerifier | None = None,
        review_box: DraftReviewBox | None = None,
    ):
        self.llm = llm or DeterministicMockLLMAdapter()
        self.tone_analyzer = tone_analyzer or ToneAnalyzer()
        self.verifier = intent_verifier or IntentVerifier()
        self.review_box = review_box

    def draft_reply(
        self,
        chat_jid: str,
        thread_messages: Sequence[NormalizedMessage],
        outgoing_history: Sequence[NormalizedMessage],
        intent_bullets: Sequence[str],
        thread_root_pk: int | None = None,
        temperature: float = 0.0,
    ) -> tuple[StagedDraft, IntentVerificationResult]:
        """Execute drafting pipeline: detect questions -> analyze tone -> synthesize -> verify -> stage."""
        if not (0.0 <= temperature <= 0.2):
            raise ValueError(
                f"Drafting temperature must be in range [0.0, 0.2], got {temperature}"
            )

        if not intent_bullets:
            raise ValueError(
                "At least one intent bullet point is required to draft a reply."
            )

        # 1. Identify unanswered questions
        unanswered = detect_unanswered_questions(thread_messages)

        # 2. Extract tone from recent outgoing messages
        tone_profile = self.tone_analyzer.analyze(outgoing_history)

        # Recipient name resolution
        recipient_name = "there"
        if unanswered:
            recipient_name = unanswered[0].sender_name.split()[0]
        elif thread_messages:
            for m in reversed(thread_messages):
                if not m.is_from_me:
                    recipient_name = m.sender_name.split()[0]
                    break

        # 3. Build synthesis prompt
        prompt = self._build_prompt(
            recipient_name=recipient_name,
            unanswered=unanswered,
            intent_bullets=intent_bullets,
            tone_profile=tone_profile,
        )

        system_prompt = (
            "You are an intent-constrained reply drafter. "
            "Draft a WhatsApp reply that strictly addresses the supplied intent points. "
            "Do NOT add any unrequested commitments, promises, deadlines, or claims. "
            "Mirror the user's greeting style, sign-off style, and formality."
        )

        # 4. Generate candidate draft via LLMAdapter
        # If deterministic mock LLM adapter is used, synthesize a structured mock response
        raw_draft = ""
        if isinstance(self.llm, DeterministicMockLLMAdapter):
            raw_draft = self._generate_mock_draft(
                recipient_name=recipient_name,
                intent_bullets=intent_bullets,
                tone_profile=tone_profile,
            )
        else:
            raw_draft = self.llm.generate(
                prompt=prompt,
                system=system_prompt,
                temperature=temperature,
            ).strip()

        # 5. Verify draft content against intent bullet boundaries
        verification = self.verifier.verify(
            draft_text=raw_draft,
            intent_bullets=intent_bullets,
            question_context=unanswered,
        )

        # 6. Create StagedDraft
        now_ts = time.time()
        draft_id = f"drf_{uuid.uuid4().hex[:10]}"
        staged = StagedDraft(
            draft_id=draft_id,
            chat_jid=chat_jid,
            created_at=now_ts,
            iso_date=format_iso_timestamp(now_ts),
            thread_root_pk=thread_root_pk,
            unanswered_questions=tuple(unanswered),
            intent_bullets=tuple(intent_bullets),
            tone_profile=tone_profile,
            draft_text=raw_draft,
            status=DraftStatus.STAGED_FOR_REVIEW,
        )

        # 7. Stage in dedicated review box if configured
        if self.review_box:
            self.review_box.stage(staged)

        return staged, verification

    def _build_prompt(
        self,
        recipient_name: str,
        unanswered: Sequence[UnansweredQuestion],
        intent_bullets: Sequence[str],
        tone_profile: ToneProfile,
    ) -> str:
        lines = [
            f"Recipient: {recipient_name}",
            "Draft a reply adhering strictly to the following parameters.",
            "\n--- UNANSWERED QUESTIONS ---",
        ]
        if unanswered:
            for u in unanswered:
                lines.append(f'• From {u.sender_name}: "{u.question_text}"')
        else:
            lines.append("• No explicit unanswered questions found in thread.")

        lines.append("\n--- USER INTENT BULLET POINTS (STRICT BOUNDARY) ---")
        for b in intent_bullets:
            lines.append(f"• {b}")

        lines.append("\n--- TONE & STYLE PARAMETERS ---")
        lines.append(f"• Formality: {tone_profile.formality.value}")
        lines.append(f"• Greeting: {tone_profile.typical_greeting or 'None'}")
        lines.append(f"• Sign-off: {tone_profile.typical_signoff or 'None'}")
        lines.append(
            f"• Contractions: {'Allowed' if tone_profile.use_contractions else 'Disallowed'}"
        )

        if tone_profile.exemplars:
            lines.append("\n--- EXEMPLAR OUTGOING MESSAGES FROM USER ---")
            for ex in tone_profile.exemplars[-5:]:
                lines.append(f'- "{ex}"')

        lines.append("\n--- OUTPUT DRAFT ---")
        return "\n".join(lines)

    def _generate_mock_draft(
        self,
        recipient_name: str,
        intent_bullets: Sequence[str],
        tone_profile: ToneProfile,
    ) -> str:
        """Deterministic draft generation for offline testing."""
        parts: list[str] = []

        # 1. Greeting
        if tone_profile.typical_greeting:
            parts.append(f"{tone_profile.typical_greeting} {recipient_name},")
        elif tone_profile.formality == ToneFormality.CASUAL:
            parts.append(f"Hey {recipient_name},")
        elif tone_profile.formality == ToneFormality.FORMAL:
            parts.append(f"Dear {recipient_name},")

        # 2. Body from intent bullets
        body_sentences: list[str] = []
        for b in intent_bullets:
            clean = re.sub(r"^[-*•\d.]+\s*", "", b).strip().rstrip(".")
            if clean:
                body_sentences.append(f"{clean}.")

        if body_sentences:
            parts.append(" ".join(body_sentences))
        else:
            parts.append("Acknowledged.")

        # 3. Sign-off
        if tone_profile.typical_signoff:
            parts.append(f"\n{tone_profile.typical_signoff}")
        elif tone_profile.formality == ToneFormality.FORMAL:
            parts.append("\nBest regards,")

        return "\n\n".join(parts) if len(parts) > 1 else parts[0]
