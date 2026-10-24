from __future__ import annotations

import io
import sqlite3
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from total_recall.cli import main as cli_main
from total_recall.drafter import (
    AutonomousSendingProhibitedError,
    DraftReviewBox,
    IntentVerifier,
    ReplyDrafter,
    ToneAnalyzer,
    detect_unanswered_questions,
)
from total_recall.ingestion import APPLE_EPOCH_OFFSET, format_iso_timestamp
from total_recall.search import SearchIndex
from total_recall.summarizer import DeterministicMockLLMAdapter
from total_recall.types import (
    DraftStatus,
    NormalizedMessage,
    StagedDraft,
    ToneFormality,
    ToneProfile,
)


def make_msg(
    pk: int,
    chat_jid: str,
    sender_name: str,
    text: str,
    timestamp: float,
    parent_pk: int | None = None,
    is_from_me: bool = False,
    chat_session_id: int = 1,
) -> NormalizedMessage:
    return NormalizedMessage(
        pk=pk,
        chat_session_id=chat_session_id,
        chat_jid=chat_jid,
        stanza_id=f"STANZA-{pk}",
        is_from_me=is_from_me,
        sender_name=sender_name,
        sender_jid="you" if is_from_me else f"{sender_name.lower()}@s.whatsapp.net",
        apple_date=timestamp - APPLE_EPOCH_OFFSET,
        unix_timestamp=timestamp,
        iso_date=format_iso_timestamp(timestamp),
        text=text,
        parent_pk=parent_pk,
    )


# ==============================================================================
# 1. Unanswered Question Detection
# ==============================================================================


def test_detect_unanswered_questions_linear_thread():
    t0 = 1761200000.0
    msgs = [
        make_msg(1, "chat1@s.whatsapp.net", "Alice", "Hey, how are you?", t0),
        make_msg(
            2, "chat1@s.whatsapp.net", "You", "Good, thanks!", t0 + 60, is_from_me=True
        ),
        make_msg(
            3,
            "chat1@s.whatsapp.net",
            "Alice",
            "Can you review the PR by Friday?",
            t0 + 120,
        ),
    ]

    unanswered = detect_unanswered_questions(msgs)
    assert len(unanswered) == 1
    assert unanswered[0].message_pk == 3
    assert unanswered[0].sender_name == "Alice"
    assert "Can you review the PR by Friday?" in unanswered[0].question_text


def test_detect_unanswered_questions_already_answered():
    t0 = 1761200000.0
    msgs = [
        make_msg(1, "chat1@s.whatsapp.net", "Alice", "Can you review the PR?", t0),
        make_msg(
            2,
            "chat1@s.whatsapp.net",
            "You",
            "Yes, on it now.",
            t0 + 60,
            is_from_me=True,
        ),
    ]

    unanswered = detect_unanswered_questions(msgs)
    assert len(unanswered) == 0


def test_detect_unanswered_questions_reply_dag_parent():
    t0 = 1761200000.0
    msgs = [
        make_msg(1, "chat1@s.whatsapp.net", "Alice", "Are you available tomorrow?", t0),
        make_msg(2, "chat1@s.whatsapp.net", "Bob", "What about Friday?", t0 + 30),
        # You reply explicitly to message 1 as parent
        make_msg(
            3,
            "chat1@s.whatsapp.net",
            "You",
            "Yes I am available tomorrow.",
            t0 + 60,
            parent_pk=1,
            is_from_me=True,
        ),
    ]

    # Alice's question (1) is explicitly replied to; Bob's question (2) was asked before You replied (ts +30 vs +60).
    # Since You replied at +60 without addressing Bob, Bob's question occurred before the latest user timestamp.
    # Now let's test a question that occurs AFTER the latest user message:
    msgs.append(
        make_msg(
            4,
            "chat1@s.whatsapp.net",
            "Charlie",
            "Could you send the spreadsheet?",
            t0 + 120,
        )
    )

    unanswered = detect_unanswered_questions(msgs)
    assert len(unanswered) == 1
    assert unanswered[0].message_pk == 4
    assert unanswered[0].sender_name == "Charlie"


def test_detect_unanswered_questions_ignores_user_questions():
    t0 = 1761200000.0
    msgs = [
        make_msg(
            1,
            "chat1@s.whatsapp.net",
            "You",
            "Where is the documentation?",
            t0,
            is_from_me=True,
        ),
    ]
    unanswered = detect_unanswered_questions(msgs)
    assert len(unanswered) == 0


# ==============================================================================
# 2. Tone Analyzer
# ==============================================================================


def test_tone_analyzer_casual_style():
    analyzer = ToneAnalyzer()
    t0 = 1761200000.0
    msgs = [
        make_msg(
            1,
            "c1",
            "You",
            "Hey Alice, yeah I'll check it out today!",
            t0,
            is_from_me=True,
        ),
        make_msg(
            2,
            "c1",
            "You",
            "hey, can't make it to the call lol",
            t0 + 10,
            is_from_me=True,
        ),
        make_msg(
            3,
            "c1",
            "You",
            "Hey! that's cool, don't worry about it",
            t0 + 20,
            is_from_me=True,
        ),
        make_msg(4, "c1", "You", "yep sounds good!", t0 + 30, is_from_me=True),
        make_msg(5, "c1", "You", "Hey, I'm heading out now", t0 + 40, is_from_me=True),
    ]

    profile = analyzer.analyze(msgs)
    assert profile.formality == ToneFormality.CASUAL
    assert profile.typical_greeting == "Hey"
    assert profile.use_contractions is True
    assert profile.sample_count == 5


def test_tone_analyzer_formal_style():
    analyzer = ToneAnalyzer()
    t0 = 1761200000.0
    msgs = [
        make_msg(
            1,
            "c1",
            "You",
            "Dear Mr. Smith, thank you for the update.\n\nBest regards,\nYash",
            t0,
            is_from_me=True,
        ),
        make_msg(
            2,
            "c1",
            "You",
            "Dear Mr. Smith, I have reviewed the contractual clauses.\n\nBest regards,\nYash",
            t0 + 10,
            is_from_me=True,
        ),
        make_msg(
            3,
            "c1",
            "You",
            "Dear Mr. Smith, please find the attached documents.\n\nBest regards,\nYash",
            t0 + 20,
            is_from_me=True,
        ),
        make_msg(
            4,
            "c1",
            "You",
            "Dear Mr. Smith, we will proceed according to the schedule.\n\nSincerely,\nYash",
            t0 + 30,
            is_from_me=True,
        ),
        make_msg(
            5,
            "c1",
            "You",
            "Dear Mr. Smith, the proposal is finalized.\n\nBest regards,\nYash",
            t0 + 40,
            is_from_me=True,
        ),
    ]

    profile = analyzer.analyze(msgs)
    assert profile.formality == ToneFormality.FORMAL
    assert profile.typical_greeting == "Dear"
    assert "Regards" in (profile.typical_signoff or "") or "Best" in (
        profile.typical_signoff or ""
    )
    assert profile.sample_count == 5


def test_tone_analyzer_empty_fallback():
    analyzer = ToneAnalyzer()
    profile = analyzer.analyze([])
    assert profile.formality == ToneFormality.NEUTRAL
    assert profile.typical_greeting is None
    assert profile.typical_signoff is None
    assert profile.sample_count == 0


# ==============================================================================
# 3. Intent Verifier & Guardrail
# ==============================================================================


def test_intent_verifier_strictly_bounded_draft():
    verifier = IntentVerifier()
    intent_bullets = [
        "Will review the PR by Friday",
        "Cannot attend morning sync",
    ]
    draft_text = "Hey Alice,\n\nI will review the PR by Friday, but cannot attend the morning sync."
    res = verifier.verify(draft_text, intent_bullets)
    assert res.is_valid is True
    assert len(res.violations) == 0
    assert res.grounding_score > 0.0


def test_intent_verifier_rejects_unrequested_commitments():
    verifier = IntentVerifier()
    intent_bullets = [
        "Cannot attend today's meeting",
    ]
    # Draft injects unrequested commitment: promising to call tonight
    draft_text = "Hey Alice,\n\nI cannot attend today's meeting. I promise I will call you tonight to discuss."
    res = verifier.verify(draft_text, intent_bullets)
    assert res.is_valid is False
    assert any("UNREQUESTED_COMMITMENT" in v for v in res.violations)


def test_intent_verifier_rejects_ungrounded_deadlines():
    verifier = IntentVerifier()
    intent_bullets = [
        "Working on the documentation",
    ]
    # Draft adds an unrequested temporal deadline "by Friday"
    draft_text = "Hey Alice,\n\nI am working on the documentation by Friday."
    res = verifier.verify(draft_text, intent_bullets)
    assert res.is_valid is False
    assert any("UNGROUNDED_DEADLINE" in v for v in res.violations)


def test_intent_verifier_rejects_ungrounded_numbers():
    verifier = IntentVerifier()
    intent_bullets = [
        "Deploying the patch tomorrow",
    ]
    # Draft introduces an ungrounded number "99"
    draft_text = "Hey Alice,\n\nDeploying 99 patches tomorrow."
    res = verifier.verify(draft_text, intent_bullets)
    assert res.is_valid is False
    assert any("UNGROUNDED_NUMBERS" in v for v in res.violations)


def test_intent_verifier_empty_intent():
    verifier = IntentVerifier()
    res = verifier.verify("Some reply", [])
    assert res.is_valid is False
    assert "NO_INTENT_POINTS_PROVIDED" in res.violations[0]


# ==============================================================================
# 4. Airgapped Review Box & Structural Prohibition
# ==============================================================================


def test_review_box_structural_prohibition():
    box = DraftReviewBox(":memory:")
    with pytest.raises(
        AutonomousSendingProhibitedError,
        match="Autonomous sending is structurally prohibited",
    ):
        box.send("drf_12345")


def test_review_box_sqlite_check_constraint():
    box = DraftReviewBox(":memory:")
    # Direct raw SQL execution attempting to set is_sent = 1 must be rejected by SQLite CHECK constraint
    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
        box.con.execute(
            """
            INSERT INTO staged_drafts (
                draft_id, chat_jid, created_at, iso_date, thread_root_pk,
                unanswered_questions_json, intent_bullets_json, tone_summary,
                draft_text, status, is_sent
            ) VALUES ('test', 'c1', 1000.0, '2026-10-24', NULL, '[]', '[]', '{}', 'draft', 'STAGED_FOR_REVIEW', 1)
            """
        )


def test_review_box_stage_approve_reject_workflow():
    box = DraftReviewBox(":memory:")
    draft = StagedDraft(
        draft_id="drf_test01",
        chat_jid="team@g.us",
        created_at=1761200000.0,
        iso_date="2026-10-24T10:00:00Z",
        thread_root_pk=None,
        unanswered_questions=(),
        intent_bullets=("Point 1", "Point 2"),
        tone_profile=ToneProfile(
            formality=ToneFormality.CASUAL,
            typical_greeting="Hey",
            typical_signoff=None,
            use_contractions=True,
            sample_count=5,
            exemplars=(),
        ),
        draft_text="Hey team,\n\nPoint 1. Point 2.",
        status=DraftStatus.STAGED_FOR_REVIEW,
    )

    box.stage(draft)
    retrieved = box.get("drf_test01")
    assert retrieved is not None
    assert retrieved.draft_id == "drf_test01"
    assert retrieved.status == DraftStatus.STAGED_FOR_REVIEW

    # Approve
    approved = box.approve("drf_test01")
    assert approved.status == DraftStatus.APPROVED_BY_USER

    # Reject
    rejected = box.reject("drf_test01")
    assert rejected.status == DraftStatus.REJECTED_BY_USER


# ==============================================================================
# 5. Reply Drafter Engine End-to-End
# ==============================================================================


def test_reply_drafter_temperature_boundary():
    drafter = ReplyDrafter(llm=DeterministicMockLLMAdapter())
    with pytest.raises(ValueError, match="Drafting temperature must be in range"):
        drafter.draft_reply("c1", [], [], ["intent"], temperature=0.5)

    with pytest.raises(ValueError, match="Drafting temperature must be in range"):
        drafter.draft_reply("c1", [], [], ["intent"], temperature=-0.1)


def test_reply_drafter_empty_intent_error():
    drafter = ReplyDrafter(llm=DeterministicMockLLMAdapter())
    with pytest.raises(
        ValueError, match="At least one intent bullet point is required"
    ):
        drafter.draft_reply("c1", [], [], [])


def test_reply_drafter_end_to_end_mock():
    review_box = DraftReviewBox(":memory:")
    drafter = ReplyDrafter(llm=DeterministicMockLLMAdapter(), review_box=review_box)

    t0 = 1761200000.0
    thread_msgs = [
        make_msg(1, "team@g.us", "Alice", "Can you review the PR by Friday?", t0),
    ]
    outgoing_history = [
        make_msg(
            2,
            "team@g.us",
            "You",
            "Hey Alice, yeah sounds good!",
            t0 - 100,
            is_from_me=True,
        ),
        make_msg(
            3,
            "team@g.us",
            "You",
            "Hey, I'll take a look later",
            t0 - 50,
            is_from_me=True,
        ),
    ]
    intent_bullets = ["Will review the PR by Friday"]

    staged, verification = drafter.draft_reply(
        chat_jid="team@g.us",
        thread_messages=thread_msgs,
        outgoing_history=outgoing_history,
        intent_bullets=intent_bullets,
        temperature=0.0,
    )

    assert staged.status == DraftStatus.STAGED_FOR_REVIEW
    assert "Alice" in staged.draft_text
    assert "review the PR by Friday" in staged.draft_text
    assert verification.is_valid is True

    # Check persistence in review box
    saved = review_box.get(staged.draft_id)
    assert saved is not None
    assert saved.draft_id == staged.draft_id


# ==============================================================================
# 6. CLI Integration Tests
# ==============================================================================


def test_cli_draft_missing_arguments():
    stderr_buf = io.StringIO()
    with redirect_stderr(stderr_buf):
        ret = cli_main(["draft"])
    assert ret == 1
    assert (
        "Specify either --chat <chat_jid> or --thread <message_pk>"
        in stderr_buf.getvalue()
    )


def test_cli_draft_temperature_validation():
    stderr_buf = io.StringIO()
    with redirect_stderr(stderr_buf):
        ret = cli_main(["draft", "--chat", "c1", "--temperature", "0.9"])
    assert ret == 1
    assert "Drafting temperature must be in range" in stderr_buf.getvalue()


def test_cli_draft_end_to_end_with_chat(tmp_path: Path):
    db_path = tmp_path / "test.db"
    t0 = 1761200000.0
    msgs = [
        make_msg(1, "team@g.us", "Alice", "Are you available for the demo?", t0),
        make_msg(
            2,
            "team@g.us",
            "You",
            "Hey Alice, yeah I'm around!",
            t0 - 100,
            is_from_me=True,
        ),
        make_msg(
            3, "team@g.us", "You", "Hey, sounds good lol", t0 - 50, is_from_me=True
        ),
    ]

    with SearchIndex(db_path) as idx:
        idx.index_messages(msgs)

    stdout_buf = io.StringIO()
    with redirect_stdout(stdout_buf):
        ret = cli_main(
            [
                "draft",
                "--chat",
                "team@g.us",
                "--db",
                str(db_path),
                "-i",
                "Available for the demo tomorrow afternoon",
                "--llm",
                "mock",
            ]
        )

    assert ret == 0
    out = stdout_buf.getvalue()
    assert "STAGED DRAFT REVIEW BOX" in out
    assert "team@g.us" in out
    assert "VERIFIED & GROUNDED" in out
    assert "Available for the demo tomorrow afternoon" in out
    assert "Autonomous Sending     : STRUCTURALLY PROHIBITED" in out


def test_cli_draft_list_and_approve(tmp_path: Path):
    db_path = tmp_path / "test.db"
    t0 = 1761200000.0
    msgs = [
        make_msg(1, "team@g.us", "Alice", "Can you review?", t0),
        make_msg(2, "team@g.us", "You", "Hey Alice, sure!", t0 - 50, is_from_me=True),
    ]

    with SearchIndex(db_path) as idx:
        idx.index_messages(msgs)

    # 1. Draft a reply to stage it
    cli_main(
        [
            "draft",
            "--chat",
            "team@g.us",
            "--db",
            str(db_path),
            "-i",
            "Will review shortly",
            "--llm",
            "mock",
        ]
    )

    # 2. List staged drafts
    list_buf = io.StringIO()
    with redirect_stdout(list_buf):
        ret = cli_main(["draft", "--db", str(db_path), "--list-staged"])
    assert ret == 0
    list_out = list_buf.getvalue()
    assert "STAGED DRAFTS REVIEW BOX" in list_out
    assert "team@g.us" in list_out

    # Find the draft id
    import re

    draft_id_match = re.search(r"Draft ID\s*:\s*(drf_[a-zA-Z0-9]+)", list_out)
    assert draft_id_match is not None
    draft_id = draft_id_match.group(1)

    # 3. Approve draft
    appr_buf = io.StringIO()
    with redirect_stdout(appr_buf):
        ret_appr = cli_main(["draft", "--db", str(db_path), "--approve", draft_id])
    assert ret_appr == 0
    assert "APPROVED_BY_USER" in appr_buf.getvalue()

    # 4. Reject draft
    rej_buf = io.StringIO()
    with redirect_stdout(rej_buf):
        ret_rej = cli_main(["draft", "--db", str(db_path), "--reject", draft_id])
    assert ret_rej == 0
    assert "REJECTED_BY_USER" in rej_buf.getvalue()
