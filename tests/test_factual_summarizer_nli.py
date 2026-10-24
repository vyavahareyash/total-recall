from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from whatsapp_total_recall.chunking import chunk_into_episodes
from whatsapp_total_recall.cli import main as cli_main
from whatsapp_total_recall.ingestion import APPLE_EPOCH_OFFSET, format_iso_timestamp
from whatsapp_total_recall.search import SearchIndex
from whatsapp_total_recall.summarizer import (
    DeterministicMockLLMAdapter,
    MiniTruePrefixesVerifier,
    OllamaLLMAdapter,
    SentimentNormalizer,
    Summarizer,
    format_citation,
    parse_citation,
)
from whatsapp_total_recall.types import NormalizedMessage


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
# 1. Temperature & LLM Adapter Constraints
# ==============================================================================


def test_temperature_boundary_validation():
    adapter = DeterministicMockLLMAdapter()

    # Temperature in [0.0, 0.1] must be accepted
    res0 = adapter.generate("Hello", temperature=0.0)
    assert res0

    res_tenth = adapter.generate("Hello", temperature=0.1)
    assert res_tenth

    # Temperatures outside [0.0, 0.1] must raise ValueError
    with pytest.raises(ValueError, match="temperature must be in range"):
        adapter.generate("Hello", temperature=0.2)

    with pytest.raises(ValueError, match="temperature must be in range"):
        adapter.generate("Hello", temperature=-0.01)

    with pytest.raises(ValueError, match="temperature must be in range"):
        adapter.generate("Hello", temperature=0.7)


def test_ollama_llm_adapter_payload():
    adapter = OllamaLLMAdapter(host="http://127.0.0.1:11434", model="qwen2.5:7b")

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.read.return_value = (
            b'{"response": "Alice confirmed release [Team | 2026-10-22 | Alice | 1]."}'
        )
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        out = adapter.generate("Summarize transcript", temperature=0.0)
        assert "Alice confirmed release" in out

        # Verify call payload
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "http://127.0.0.1:11434/api/generate"
        assert req.method == "POST"


def test_ollama_llm_adapter_connection_error():
    adapter = OllamaLLMAdapter(host="http://127.0.0.1:9999", timeout=0.1)
    with pytest.raises(
        ConnectionError, match="Failed to communicate with local Ollama"
    ):
        adapter.generate("Test prompt", temperature=0.0)


# ==============================================================================
# 2. Citation Formatter & Parser
# ==============================================================================


def test_citation_formatting_and_parsing():
    cit_str = format_citation("Dev Group", "2026-10-22 10:00:00 UTC", "Alice", 42)
    assert cit_str == "[Dev Group | 2026-10-22 10:00:00 UTC | Alice | 42]"

    parsed = parse_citation(f"Statement here {cit_str} and more text.")
    assert parsed is not None
    assert parsed.chat_name == "Dev Group"
    assert parsed.iso_date == "2026-10-22 10:00:00 UTC"
    assert parsed.sender_name == "Alice"
    assert parsed.message_pk == 42
    assert str(parsed) == cit_str


def test_parse_citation_none_on_invalid():
    assert parse_citation("No citation here") is None
    assert parse_citation("[Invalid | Citation]") is None


# ==============================================================================
# 3. Sentiment Normalization & Evaluative Adjectives
# ==============================================================================


def test_sentiment_normalizer_detects_evaluative_bias():
    normalizer = SentimentNormalizer()

    biased_texts = [
        "Alice made a ridiculous proposal yesterday.",
        "Bob gave an incompetent and terrible presentation.",
        "The server crashed in a disastrous manner.",
        "She acted furiously and aggressively during the debate.",
        "His code was brilliant and genius.",
    ]
    for text in biased_texts:
        assert normalizer.contains_evaluative_bias(text) is True
        words = normalizer.find_evaluative_words(text)
        assert len(words) >= 1

    neutral_texts = [
        "Alice proposed deploying the database update on Friday.",
        "Bob reported that latency increased by 15ms after the patch.",
        "The migration script completed in 42 seconds.",
    ]
    for text in neutral_texts:
        assert normalizer.contains_evaluative_bias(text) is False
        assert normalizer.find_evaluative_words(text) == []


def test_sentiment_normalizer_strips_evaluative_adjectives():
    normalizer = SentimentNormalizer()
    text = "Alice made a ridiculous proposal regarding the terrible database."
    cleaned, stripped = normalizer.normalize(text)
    assert "ridiculous" in stripped
    assert "terrible" in stripped
    assert "ridiculous" not in cleaned
    assert "terrible" not in cleaned


# ==============================================================================
# 4. MiniTruePrefixes NLI Entailment Verification
# ==============================================================================


def test_nli_verifier_entailed_claim():
    verifier = MiniTruePrefixesVerifier()
    msg = make_msg(
        pk=1,
        chat_jid="work@g.us",
        sender_name="Alice",
        text="We successfully deployed version 2.4 to production.",
        timestamp=1700000000.0,
    )
    source = {1: msg}

    claim = "Alice stated that version 2.4 was deployed to production [Work | 2026-10-22 | Alice | 1]."
    result = verifier.verify_claim(claim, source)
    assert result.entailed is True
    assert result.discard_reason is None
    assert result.citation is not None
    assert result.citation.message_pk == 1


def test_nli_verifier_missing_citation():
    verifier = MiniTruePrefixesVerifier()
    source = {1: make_msg(1, "work@g.us", "Alice", "Hello", 1700000000.0)}

    claim = "Alice said hello to the team."
    result = verifier.verify_claim(claim, source)
    assert result.entailed is False
    assert result.discard_reason == "MISSING_CITATION"


def test_nli_verifier_invalid_source_msg():
    verifier = MiniTruePrefixesVerifier()
    source = {1: make_msg(1, "work@g.us", "Alice", "Hello", 1700000000.0)}

    claim = "Alice stated something [Work | 2026-10-22 | Alice | 999]."
    result = verifier.verify_claim(claim, source)
    assert result.entailed is False
    assert result.discard_reason == "INVALID_SOURCE_MSG"


def test_nli_verifier_sender_mismatch():
    verifier = MiniTruePrefixesVerifier()
    source = {
        1: make_msg(1, "work@g.us", "Alice", "I will do the review", 1700000000.0)
    }

    claim = "Bob stated that he will do the review [Work | 2026-10-22 | Bob | 1]."
    result = verifier.verify_claim(claim, source)
    assert result.entailed is False
    assert result.discard_reason == "SENDER_MISMATCH"


def test_nli_verifier_rejects_evaluative_bias():
    verifier = MiniTruePrefixesVerifier()
    source = {
        1: make_msg(
            1,
            "work@g.us",
            "Alice",
            "The migration failed due to a timeout",
            1700000000.0,
        )
    }

    claim = "Alice gave an incompetent and ridiculous explanation [Work | 2026-10-22 | Alice | 1]."
    result = verifier.verify_claim(claim, source)
    assert result.entailed is False
    assert "EVALUATIVE_BIAS" in (result.discard_reason or "")


def test_nli_verifier_rejects_ungrounded_numbers():
    verifier = MiniTruePrefixesVerifier()
    source = {
        1: make_msg(
            1, "work@g.us", "Alice", "We onboarded 15 new beta users", 1700000000.0
        )
    }

    # Claim asserts 500 users instead of 15
    claim = "Alice stated that 500 new users were onboarded [Work | 2026-10-22 | Alice | 1]."
    result = verifier.verify_claim(claim, source)
    assert result.entailed is False
    assert "UNGROUNDED_NUMBERS" in (result.discard_reason or "")


def test_nli_verifier_rejects_ungrounded_entities():
    verifier = MiniTruePrefixesVerifier()
    source = {
        1: make_msg(
            1,
            "work@g.us",
            "Alice",
            "We will meet at the office cafeteria",
            1700000000.0,
        )
    }

    # Claim introduces unmentioned proper noun "Starbucks"
    claim = "Alice stated that the team will meet at Starbucks [Work | 2026-10-22 | Alice | 1]."
    result = verifier.verify_claim(claim, source)
    assert result.entailed is False
    assert "UNGROUNDED_ENTITY" in (result.discard_reason or "")


def test_nli_verifier_rejects_contradiction():
    verifier = MiniTruePrefixesVerifier()
    source = {
        1: make_msg(
            1,
            "work@g.us",
            "Alice",
            "We cannot deploy the database migration on Friday",
            1700000000.0,
        )
    }

    # Premise has 'cannot' (negation), claim asserts positive deployment
    claim = "Alice confirmed we will deploy the database migration on Friday [Work | 2026-10-22 | Alice | 1]."
    result = verifier.verify_claim(claim, source)
    assert result.entailed is False
    assert result.discard_reason == "CONTRADICTION"


# ==============================================================================
# 5. Group Debate Perspective Splits (No Synthetic Compromises)
# ==============================================================================


def test_group_debate_perspective_splits():
    msgs = [
        make_msg(
            1,
            "team@g.us",
            "Alice",
            "I propose moving our backend services to Rust.",
            1700000000.0,
        ),
        make_msg(
            2,
            "team@g.us",
            "Bob",
            "I strongly oppose Rust and prefer continuing with Go.",
            1700000060.0,
            parent_pk=1,
        ),
    ]
    episodes = chunk_into_episodes(msgs)

    adapter = DeterministicMockLLMAdapter(inject_split_debate=True)
    summarizer = Summarizer(llm=adapter)

    summary = summarizer.summarize(episodes, temperature=0.0)

    # Must contain explicit perspective splits
    assert len(summary.splits) > 0
    split = summary.splits[0]
    assert len(split.perspectives) >= 2
    assert "Perspective A" in split.perspectives[0][0]
    assert "Perspective B" in split.perspectives[1][0]

    # Verify summary text formats distinct perspectives
    assert "Conflicting Perspectives" in summary.summary_text
    assert "Perspective A" in summary.summary_text
    assert "Perspective B" in summary.summary_text
    # No false compromise claims
    assert "agreed on a compromise" not in summary.summary_text.lower()


# ==============================================================================
# 6. Deep Module Summarizer End-to-End
# ==============================================================================


def test_summarizer_end_to_end_with_ungrounded_discard():
    msgs = [
        make_msg(
            1, "sync@g.us", "Alice", "Project Total Recall is active", 1700000000.0
        ),
        make_msg(
            2, "sync@g.us", "Bob", "Hybrid retrieval search is passing", 1700000050.0
        ),
    ]
    episodes = chunk_into_episodes(msgs)

    # Injects both valid claims, one ungrounded claim (invalid msg_id), and one biased claim
    adapter = DeterministicMockLLMAdapter(
        inject_ungrounded_sentence=True,
        inject_evaluative_adjective=True,
    )
    summarizer = Summarizer(llm=adapter)

    summary = summarizer.summarize(episodes, temperature=0.0)

    assert len(summary.claims) == 2  # The two legitimate grounded claims
    assert len(summary.discarded_claims) == 2  # 1 ungrounded + 1 evaluative bias

    # Verify discarded claims have explicit reasons
    reasons = [c.discard_reason for c in summary.discarded_claims]
    assert any("INVALID_SOURCE_MSG" in r for r in reasons if r)
    assert any("EVALUATIVE_BIAS" in r for r in reasons if r)

    # Legitimate claims appear in summary text
    assert "Alice" in summary.summary_text
    assert "Bob" in summary.summary_text


def test_summarizer_thread_dag():
    msgs = [
        make_msg(10, "chat@g.us", "Alice", "Should we enable WAL mode?", 1700000000.0),
        make_msg(
            11,
            "chat@g.us",
            "Bob",
            "Yes, WAL mode prevents 0xDEAD10CC crashes.",
            1700000050.0,
            parent_pk=10,
        ),
    ]
    adapter = DeterministicMockLLMAdapter()
    summarizer = Summarizer(llm=adapter)

    summary = summarizer.summarize_thread(messages=msgs, root_pk=10, temperature=0.0)
    assert len(summary.claims) == 2
    assert summary.episodes_analyzed >= 1


def test_summarizer_empty_input():
    summarizer = Summarizer()
    summary = summarizer.summarize([])
    assert summary.episodes_analyzed == 0
    assert summary.messages_analyzed == 0
    assert "No episodes provided" in summary.summary_text


# ==============================================================================
# 7. CLI Subcommand: total-recall summarize
# ==============================================================================


def test_cli_summarize_chat(tmp_path: Path):
    db_path = tmp_path / "whatsapp_total_recall.db"
    with SearchIndex(db_path) as index:
        msgs = [
            make_msg(
                1, "chat1@g.us", "Alice", "Release v1.0 is scheduled", 1700000000.0
            ),
            make_msg(
                2, "chat1@g.us", "Bob", "QA tests are 100% complete", 1700000050.0
            ),
        ]
        episodes = chunk_into_episodes(msgs)
        index.index_messages(msgs)
        index.index_episodes(episodes)

    stdout_buf = io.StringIO()
    with redirect_stdout(stdout_buf):
        exit_code = cli_main(
            [
                "summarize",
                "--chat",
                "chat1@g.us",
                "--db",
                str(db_path),
                "--llm",
                "mock",
                "--temperature",
                "0.0",
            ]
        )

    assert exit_code == 0
    out = stdout_buf.getvalue()
    assert "VERIFIED FACTUAL SUMMARY" in out
    assert "chat1@g.us" in out
    assert "VERIFICATION AUDIT REPORT" in out
    assert "Claims Entailed & Verified : 2" in out


def test_cli_summarize_thread(tmp_path: Path):
    db_path = tmp_path / "whatsapp_total_recall.db"
    with SearchIndex(db_path) as index:
        msgs = [
            make_msg(10, "chat2@g.us", "Alice", "Is the index ready?", 1700000000.0),
            make_msg(
                11,
                "chat2@g.us",
                "Bob",
                "Yes, the index is ready now.",
                1700000050.0,
                parent_pk=10,
            ),
        ]
        index.index_messages(msgs)

    stdout_buf = io.StringIO()
    with redirect_stdout(stdout_buf):
        exit_code = cli_main(
            [
                "summarize",
                "--thread",
                "10",
                "--db",
                str(db_path),
                "--llm",
                "mock",
            ]
        )

    assert exit_code == 0
    out = stdout_buf.getvalue()
    assert "VERIFIED FACTUAL SUMMARY" in out
    assert "Claims Entailed & Verified : 2" in out


def test_cli_summarize_missing_arguments():
    stderr_buf = io.StringIO()
    with redirect_stderr(stderr_buf):
        exit_code = cli_main(["summarize"])
    assert exit_code == 1
    assert (
        "Specify either --chat <chat_jid> or --thread <message_pk>"
        in stderr_buf.getvalue()
    )


def test_cli_summarize_invalid_temperature():
    stderr_buf = io.StringIO()
    with redirect_stderr(stderr_buf):
        exit_code = cli_main(
            [
                "summarize",
                "--chat",
                "test@g.us",
                "--temperature",
                "0.8",
            ]
        )
    assert exit_code == 1
    assert "temperature must be in range [0.0, 0.1]" in stderr_buf.getvalue()
