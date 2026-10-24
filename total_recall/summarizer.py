from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Protocol, runtime_checkable

from total_recall.chunking import chunk_into_episodes
from total_recall.dag import ReplyDAG
from total_recall.types import (
    Citation,
    ConversationalFlowEpisode,
    NormalizedMessage,
    PerspectiveSplit,
    VerifiedClaim,
    VerifiedSummary,
)

# Regex to match exact citation pattern: [Chat | Date | Sender | msg_id]
CITATION_REGEX = re.compile(
    r"\[\s*([^|\]]+?)\s*\|\s*([^|\]]+?)\s*\|\s*([^|\]]+?)\s*\|\s*([0-9]+)\s*\]"
)

# Common evaluative adjectives and affective bias markers to reject or normalize
EVALUATIVE_ADJECTIVES = frozenset(
    {
        "absurd",
        "absurdly",
        "aggressive",
        "aggressively",
        "amazing",
        "amazingly",
        "arrogant",
        "arrogantly",
        "awful",
        "brilliant",
        "brilliantly",
        "catastrophic",
        "clueless",
        "disaster",
        "disastrous",
        "flawless",
        "foolish",
        "foolishly",
        "furious",
        "furiously",
        "genius",
        "heroic",
        "heroically",
        "hopeless",
        "horrible",
        "horribly",
        "hostile",
        "idiotic",
        "incompetent",
        "incompetently",
        "incredibly",
        "insane",
        "malicious",
        "nasty",
        "nonsense",
        "outrageous",
        "outrageously",
        "pathetic",
        "pathetically",
        "ridiculous",
        "ridiculously",
        "stupid",
        "terrible",
        "terribly",
    }
)


def format_citation(
    chat_name: str, iso_date: str, sender_name: str, message_pk: int
) -> str:
    """Format exact citation string [Chat | Date | Sender | msg_id]."""
    return f"[{chat_name} | {iso_date} | {sender_name} | {message_pk}]"


def parse_citation(text: str) -> Citation | None:
    """Extract first valid citation [Chat | Date | Sender | msg_id] from text."""
    match = CITATION_REGEX.search(text)
    if not match:
        return None
    chat_name = match.group(1).strip()
    iso_date = match.group(2).strip()
    sender_name = match.group(3).strip()
    message_pk = int(match.group(4).strip())
    raw = match.group(0)
    return Citation(
        chat_name=chat_name,
        iso_date=iso_date,
        sender_name=sender_name,
        message_pk=message_pk,
        raw=raw,
    )


@runtime_checkable
class LLMAdapter(Protocol):
    """Protocol for local LLM text generation adapters."""

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Generate text completion from prompt with strict temperature controls."""
        ...


class OllamaLLMAdapter:
    """Invokes local Ollama daemon over 127.0.0.1 HTTP API with zero third-party dependencies."""

    def __init__(
        self,
        host: str = "http://127.0.0.1:11434",
        model: str = "qwen2.5:7b",
        timeout: float = 30.0,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        if not (0.0 <= temperature <= 0.1):
            raise ValueError(
                f"Summarization temperature must be in range [0.0, 0.1], got {temperature}"
            )

        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system or "",
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response", "").strip()
        except urllib.error.HTTPError as err:
            body_msg = ""
            try:
                body_json = json.loads(err.read().decode("utf-8"))
                body_msg = body_json.get("error", "")
            except Exception:
                pass
            detail = f": {body_msg}" if body_msg else f" ({err})"
            raise ConnectionError(
                f"Failed to communicate with local Ollama at {self.host}: HTTP {err.code} for model '{self.model}'{detail}. "
                f"Make sure model '{self.model}' is installed (try: ollama pull {self.model})."
            ) from err
        except urllib.error.URLError as err:
            raise ConnectionError(
                f"Failed to communicate with local Ollama at {self.host}: {err}"
            ) from err


class DeterministicMockLLMAdapter:
    """Zero-dependency deterministic LLM adapter for offline test execution."""

    def __init__(
        self,
        model: str = "mock-court-reporter",
        custom_responses: list[str] | None = None,
        inject_ungrounded_sentence: bool = False,
        inject_evaluative_adjective: bool = False,
        inject_split_debate: bool = False,
    ):
        self.model = model
        self._custom_responses = list(custom_responses) if custom_responses else []
        self._inject_ungrounded_sentence = inject_ungrounded_sentence
        self._inject_evaluative_adjective = inject_evaluative_adjective
        self._inject_split_debate = inject_split_debate

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        if not (0.0 <= temperature <= 0.1):
            raise ValueError(
                f"Summarization temperature must be in range [0.0, 0.1], got {temperature}"
            )

        if self._custom_responses:
            return self._custom_responses.pop(0)

        # Parse messages from prompt if possible
        statements: list[str] = []

        # Find any message lines in the prompt like:
        # [2026-10-22 09:00:00 UTC] Alice: ... (msg_id: 1)
        msg_pattern = re.compile(
            r"\[([^\]]+)\]\s+([^:]+?)(?:\s+\([^\)]+\))?:\s+(.+?)\s+\(msg_id:\s*(\d+)\)"
        )
        matches = msg_pattern.findall(prompt)

        # Also find chat name if present
        chat_name_match = re.search(r"\[Chat:\s*([^\]\(\)]+)", prompt)
        chat_name = chat_name_match.group(1).strip() if chat_name_match else "Chat"

        if matches:
            if self._inject_split_debate and len(matches) >= 2:
                # Format an explicit split rather than compromise
                m1_date, m1_sender, m1_text, m1_id = matches[0]
                m2_date, m2_sender, m2_text, m2_id = matches[1]
                statements.append("Dispute regarding proposal:")
                statements.append(
                    f"Perspective A: {m1_sender} proposed: '{m1_text}' [{chat_name} | {m1_date} | {m1_sender} | {m1_id}]."
                )
                statements.append(
                    f"Perspective B: {m2_sender} opposed: '{m2_text}' [{chat_name} | {m2_date} | {m2_sender} | {m2_id}]."
                )
            else:
                for date, sender, text, msg_id in matches:
                    clean_text = text.strip().rstrip(".")
                    statements.append(
                        f"{sender} stated that {clean_text} [{chat_name} | {date} | {sender} | {msg_id}]."
                    )

            if self._inject_ungrounded_sentence:
                statements.append(
                    f"The project deadline was moved to next month [{chat_name} | 2026-10-22 | Alice | 999999]."
                )

            if self._inject_evaluative_adjective:
                target_match = matches[1] if len(matches) > 1 else matches[0]
                statements.append(
                    f"{target_match[1]} gave a ridiculous and terrible explanation [{chat_name} | 2026-10-22 | {target_match[1]} | {target_match[3]}]."
                )

            return "\n".join(statements)

        return f"Discussion occurred without recorded statements [{chat_name} | 2026-01-01 | Unknown | 0]."


class SentimentNormalizer:
    """Enforces Epistemic Neutrality by detecting and stripping affective bias and evaluative adjectives."""

    def __init__(self, evaluative_words: set[str] | frozenset[str] | None = None):
        self.evaluative_words = (
            frozenset(w.lower() for w in evaluative_words)
            if evaluative_words is not None
            else EVALUATIVE_ADJECTIVES
        )

    def find_evaluative_words(self, text: str) -> list[str]:
        """Find any evaluative adjectives or affective bias tokens present in text."""
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        found = [w for w in words if w in self.evaluative_words]
        return found

    def contains_evaluative_bias(self, text: str) -> bool:
        """Check if statement contains evaluative adjectives or affective bias."""
        return len(self.find_evaluative_words(text)) > 0

    def normalize(self, text: str) -> tuple[str, list[str]]:
        """Strip evaluative adjectives and emotional speculation from claim."""
        found = self.find_evaluative_words(text)
        if not found:
            return text, []

        pattern = re.compile(
            r"\b(" + "|".join(re.escape(w) for w in found) + r")\b\s*",
            re.IGNORECASE,
        )
        cleaned = pattern.sub("", text)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        return cleaned, found


class MiniTruePrefixesVerifier:
    """Prefix-Level Natural Language Inference (NLI) Verifier.

    Validates candidate claims against source episode messages and discards unentailed,
    uncited, contradicted, or biased statements.
    """

    def __init__(
        self,
        normalizer: SentimentNormalizer | None = None,
        llm: LLMAdapter | None = None,
    ):
        self.normalizer = normalizer or SentimentNormalizer()
        self.llm = llm

    def verify_claim(
        self,
        claim_text: str,
        source_messages: dict[int, NormalizedMessage],
    ) -> VerifiedClaim:
        """Verify an individual candidate claim sentence against source messages."""
        # 1. Parse citation
        citation = parse_citation(claim_text)
        if citation is None:
            return VerifiedClaim(
                text=claim_text.strip(),
                citation=None,
                entailed=False,
                entailment_score=0.0,
                discard_reason="MISSING_CITATION",
            )

        # 2. Check source message existence
        msg = source_messages.get(citation.message_pk)
        if msg is None:
            return VerifiedClaim(
                text=claim_text.strip(),
                citation=citation,
                entailed=False,
                entailment_score=0.0,
                discard_reason="INVALID_SOURCE_MSG",
            )

        # 3. Check sender identity match
        if citation.sender_name.lower() not in msg.sender_name.lower():
            return VerifiedClaim(
                text=claim_text.strip(),
                citation=citation,
                entailed=False,
                entailment_score=0.0,
                discard_reason="SENDER_MISMATCH",
            )

        # 4. Check for evaluative adjectives & affective bias
        bias_words = self.normalizer.find_evaluative_words(claim_text)
        if bias_words:
            return VerifiedClaim(
                text=claim_text.strip(),
                citation=citation,
                entailed=False,
                entailment_score=0.0,
                discard_reason=f"EVALUATIVE_BIAS: {', '.join(bias_words)}",
            )

        # 5. Extract claim premise (source text) and hypothesis (claim without citation)
        hypothesis = CITATION_REGEX.sub("", claim_text).strip()
        premise = msg.text.strip()

        # Build allowed entities for speaker attributions and framing
        allowed_entities = {
            "the",
            "this",
            "that",
            "there",
            "when",
            "where",
            "what",
            "who",
            "how",
            "why",
            "stated",
            "said",
            "proposed",
            "perspective",
            "opposed",
            "reported",
            "confirmed",
            "noted",
            "mentioned",
            "replied",
            "asked",
            "yes",
            "no",
            "and",
            "but",
            "stance",
            "chat",
            "date",
            "team",
            "sync",
            "project",
            "work",
            "release",
            "dispute",
            "proposal",
        }
        if citation:
            allowed_entities.add(citation.sender_name.lower())
            allowed_entities.add(citation.chat_name.lower())
        if msg:
            allowed_entities.add(msg.sender_name.lower())
        for sm in source_messages.values():
            allowed_entities.add(sm.sender_name.lower())

        # 6. Prefix-level NLI entailment evaluation
        entailed, reason, score = self._evaluate_prefix_entailment(
            premise, hypothesis, allowed_entities=allowed_entities
        )
        if not entailed:
            return VerifiedClaim(
                text=claim_text.strip(),
                citation=citation,
                entailed=False,
                entailment_score=score,
                discard_reason=reason,
            )

        return VerifiedClaim(
            text=claim_text.strip(),
            citation=citation,
            entailed=True,
            entailment_score=score,
            discard_reason=None,
        )

    def _evaluate_prefix_entailment(
        self,
        premise: str,
        hypothesis: str,
        allowed_entities: set[str] | None = None,
    ) -> tuple[bool, str | None, float]:
        """Evaluate prefix entailment and factual grounding between premise and hypothesis."""
        allowed_entities = allowed_entities or set()
        premise_lower = premise.lower()
        hypothesis_lower = hypothesis.lower()

        # Check for direct numerical grounding (numbers in hypothesis must be present in premise)
        hyp_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", hypothesis_lower))
        prem_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", premise_lower))
        ungrounded_nums = hyp_numbers - prem_numbers
        if ungrounded_nums:
            return (
                False,
                f"UNGROUNDED_NUMBERS: {', '.join(sorted(ungrounded_nums))}",
                0.0,
            )

        # Check for negation contradiction
        prem_has_neg = any(
            re.search(rf"\b{neg}\b", premise_lower)
            for neg in ("not", "never", "cannot", "can't", "won't", "no")
        )
        hyp_has_neg = any(
            re.search(rf"\b{neg}\b", hypothesis_lower)
            for neg in ("not", "never", "cannot", "can't", "won't", "no")
        )

        # If polarities strongly conflict on identical root predicates
        prem_words = set(re.findall(r"\w+", premise_lower))
        hyp_words = set(re.findall(r"\w+", hypothesis_lower))
        shared_words = hyp_words & prem_words

        # If hypothesis asserts something directly inverted from premise
        if prem_has_neg != hyp_has_neg and len(shared_words) >= 3:
            return False, "CONTRADICTION", 0.0

        # Prefix-level check: split hypothesis into cumulative prefix clauses
        hyp_sentences = [s.strip() for s in re.split(r"[.!?]", hypothesis) if s.strip()]
        for s in hyp_sentences:
            s_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", s.lower()))
            if not s_words:
                continue

            # Check entity grounding: look for capitalized proper nouns
            proper_nouns = [
                m.group(0)
                for m in re.finditer(r"\b[A-Z][a-z]+\b", s)
                if m.group(0).lower() not in allowed_entities
            ]
            for noun in proper_nouns:
                if (
                    noun.lower() not in premise_lower
                    and noun.lower() not in allowed_entities
                ):
                    return False, f"UNGROUNDED_ENTITY: {noun}", 0.0

            # Lexical semantic overlap between clause and premise
            semantic_words = {
                w for w in s_words if w not in allowed_entities and len(w) >= 3
            }
            if semantic_words:
                overlap = semantic_words & prem_words
                overlap_ratio = len(overlap) / len(semantic_words)
                if overlap_ratio < 0.20 and len(semantic_words) >= 3:
                    return (
                        False,
                        f"NOT_ENTAILED: insufficient grounding ({overlap_ratio:.2f})",
                        overlap_ratio,
                    )

        return True, None, 1.0


class Summarizer:
    """Deep module for factual, court-reporter style conversational summarization.

    Enforces temperature T = 0.0 - 0.1, verbatim citations [Chat | Date | Sender | msg_id],
    sentiment normalization, and prefix-level NLI verification.
    """

    def __init__(
        self,
        llm: LLMAdapter | None = None,
        verifier: MiniTruePrefixesVerifier | None = None,
        normalizer: SentimentNormalizer | None = None,
    ):
        self.normalizer = normalizer or SentimentNormalizer()
        self.llm = llm or DeterministicMockLLMAdapter()
        self.verifier = verifier or MiniTruePrefixesVerifier(
            normalizer=self.normalizer, llm=self.llm
        )

    def summarize(
        self,
        episodes: list[ConversationalFlowEpisode],
        intent_notes: str | None = None,
        title: str | None = None,
        temperature: float = 0.0,
    ) -> VerifiedSummary:
        """Generate factual court-reporter summary across one or more episodes."""
        if not episodes:
            return VerifiedSummary(
                chat_jid="",
                title=title or "Empty Summary",
                claims=(),
                discarded_claims=(),
                splits=(),
                summary_text="No episodes provided for summarization.",
                episodes_analyzed=0,
                messages_analyzed=0,
            )

        # Collect all messages across episodes
        all_messages: dict[int, NormalizedMessage] = {}
        for ep in episodes:
            for msg in ep.messages:
                all_messages[msg.pk] = msg

        chat_jid = episodes[0].chat_jid
        chat_name = episodes[0].chat_name or chat_jid

        # 1. Build prompt
        prompt = self._build_prompt(episodes, intent_notes, chat_name)
        system_prompt = (
            "You are a court-reporter factual summarizer. "
            "Report only directly stated facts with verbatim citations formatted strictly as "
            "[Chat | Date | Sender | msg_id]. Do not synthesize artificial consensus. "
            "If participants disagree, state conflicting perspectives separately. "
            "Strip all emotional language, affective speculation, and evaluative adjectives."
        )

        # 2. Invoke local LLM adapter with strict temperature
        raw_output = self.llm.generate(
            prompt=prompt,
            system=system_prompt,
            temperature=temperature,
        )

        # 3. Parse and verify statements
        return self._verify_and_assemble_summary(
            raw_output=raw_output,
            chat_jid=chat_jid,
            chat_name=chat_name,
            source_messages=all_messages,
            episodes_analyzed=len(episodes),
            title=title,
        )

    def summarize_thread(
        self,
        messages: list[NormalizedMessage],
        root_pk: int | None = None,
        title: str | None = None,
        temperature: float = 0.0,
    ) -> VerifiedSummary:
        """Generate factual court-reporter summary for a reply DAG thread."""
        if not messages:
            return VerifiedSummary(
                chat_jid="",
                title=title or "Empty Thread Summary",
                claims=(),
                discarded_claims=(),
                splits=(),
                summary_text="No thread messages provided for summarization.",
                episodes_analyzed=0,
                messages_analyzed=0,
            )

        dag = ReplyDAG(messages)
        if root_pk is not None and root_pk in dag.messages:
            thread_msgs = dag.get_thread_messages(root_pk)
        else:
            thread_msgs = messages

        episodes = chunk_into_episodes(thread_msgs)
        return self.summarize(
            episodes=episodes,
            title=title or f"Thread Summary #{root_pk or thread_msgs[0].pk}",
            temperature=temperature,
        )

    def _build_prompt(
        self,
        episodes: list[ConversationalFlowEpisode],
        intent_notes: str | None,
        chat_name: str,
    ) -> str:
        """Construct prompt containing grounded message transcripts with citation anchors."""
        lines = [
            f"Transcript for chat: {chat_name}",
            "Summarize the following conversation in neutral court-reporter style.",
            "Rules:",
            "1. Every single claim must end with an exact citation: [Chat | Date | Sender | msg_id].",
            "2. Do NOT use evaluative adjectives (e.g. brilliant, terrible, ridiculous).",
            "3. If participants debate or disagree, present each perspective as an explicit split.",
        ]

        if intent_notes:
            lines.append(f"Focus areas requested: {intent_notes}")

        lines.append("\n--- CONVERSATION TRANSCRIPT ---")
        for ep in episodes:
            lines.append(f"{ep.header}")
            lines.append(f"{ep.content}\n")

        lines.append("--- SUMMARY ---")
        return "\n".join(lines)

    def _verify_and_assemble_summary(
        self,
        raw_output: str,
        chat_jid: str,
        chat_name: str,
        source_messages: dict[int, NormalizedMessage],
        episodes_analyzed: int,
        title: str | None,
    ) -> VerifiedSummary:
        """Parse raw output lines, verify through MiniTruePrefixes, and separate splits."""
        raw_lines = [line.strip() for line in raw_output.splitlines() if line.strip()]

        verified_claims: list[VerifiedClaim] = []
        discarded_claims: list[VerifiedClaim] = []
        split_perspectives: list[tuple[str, tuple[VerifiedClaim, ...]]] = []

        current_perspective_name: str | None = None
        current_perspective_claims: list[VerifiedClaim] = []
        topic_title = "Debate / Disagreement"

        for line in raw_lines:
            clean_lower = line.lower().strip()
            # Check for topic section header like "Dispute regarding proposal:"
            if (
                clean_lower.endswith(":")
                and not any(p in clean_lower for p in ("perspective", "stance"))
            ) or clean_lower.startswith(("topic:", "debate:", "dispute:")):
                parts = line.split(":", 1)
                if parts[0].strip():
                    topic_title = parts[0].strip()
                continue

            # Check for perspective split header
            if line.lower().startswith("perspective") or "stance:" in line.lower():
                # If we had a previous perspective, commit it
                if current_perspective_name and current_perspective_claims:
                    split_perspectives.append(
                        (current_perspective_name, tuple(current_perspective_claims))
                    )
                    current_perspective_claims = []

                parts = line.split(":", 1)
                current_perspective_name = parts[0].strip()
                if len(parts) > 1 and parts[1].strip():
                    claim_result = self.verifier.verify_claim(
                        parts[1].strip(), source_messages
                    )
                    if claim_result.entailed:
                        current_perspective_claims.append(claim_result)
                        verified_claims.append(claim_result)
                    else:
                        discarded_claims.append(claim_result)
                continue

            # Standard claim line
            claim_result = self.verifier.verify_claim(line, source_messages)
            if claim_result.entailed:
                if current_perspective_name:
                    current_perspective_claims.append(claim_result)
                verified_claims.append(claim_result)
            else:
                discarded_claims.append(claim_result)

        if current_perspective_name and current_perspective_claims:
            split_perspectives.append(
                (current_perspective_name, tuple(current_perspective_claims))
            )

        splits: tuple[PerspectiveSplit, ...] = ()
        if split_perspectives:
            splits = (
                PerspectiveSplit(
                    topic=topic_title,
                    perspectives=tuple(split_perspectives),
                ),
            )

        # Assemble clean summary text
        summary_paragraphs: list[str] = []
        if splits:
            summary_paragraphs.append("### Conflicting Perspectives")
            for split in splits:
                for stance, claims in split.perspectives:
                    summary_paragraphs.append(f"**{stance}**:")
                    for c in claims:
                        summary_paragraphs.append(f"- {c.text}")
        else:
            for c in verified_claims:
                summary_paragraphs.append(f"- {c.text}")

        if not summary_paragraphs:
            summary_paragraphs.append(
                "No claims met factual prefix NLI verification standards."
            )

        summary_text = "\n".join(summary_paragraphs)

        return VerifiedSummary(
            chat_jid=chat_jid,
            title=title or f"Verified Summary for {chat_name}",
            claims=tuple(verified_claims),
            discarded_claims=tuple(discarded_claims),
            splits=splits,
            summary_text=summary_text,
            episodes_analyzed=episodes_analyzed,
            messages_analyzed=len(source_messages),
        )
