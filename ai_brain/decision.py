"""Feedback-conditioned internal studio decision engine."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from decimal import Decimal
from hashlib import sha256
from typing import TYPE_CHECKING

from .artist_config import ArtistConfigManager
from .decision_schemas import (
    ArtistOption,
    ArtistSuggestion,
    DecisionAction,
    DecisionHistoryExample,
    InternalPriceEstimate,
    NextAction,
    StudioDecisionContext,
    StudioDecisionFeedback,
    StudioDecisionOutput,
    StudioLearningRecord,
)
from .pricing import InternalPricingEstimator, parse_size_cm
from .schemas import AIExtractionOutput, ConfidenceLevel, TattooInquiryInput

if TYPE_CHECKING:
    from .vector_store import VectorStoreManager

LOGGER = logging.getLogger(__name__)

_PRICE_TERMS = ("price", "pricing", "cost", "quote", "how much")
_VECTOR_SEARCH_TOP_K = 12


class StudioDecisionEngine:
    """Use verified historical decisions to guide studio recommendations."""

    def __init__(
        self,
        pricing_estimator: InternalPricingEstimator | None = None,
        vector_store: VectorStoreManager | None = None,
        artist_config_manager: ArtistConfigManager | None = None,
    ) -> None:
        """Initialize deterministic decision and candidate-retrieval services.

        Args:
            pricing_estimator: Optional staff-only pricing implementation.
            vector_store: Optional FAISS manager containing verified examples.
                When omitted, request-scoped decision history is used.
            artist_config_manager: Optional studio artist catalog used to
                validate active status and detected-style compatibility.
        """
        self._pricing_estimator = (
            pricing_estimator
            if pricing_estimator is not None
            else InternalPricingEstimator()
        )
        self._vector_store = vector_store
        self._artist_config = (
            artist_config_manager
            if artist_config_manager is not None
            else ArtistConfigManager()
        )

    def decide(
        self,
        analysis: AIExtractionOutput,
        current_message: str,
        context: StudioDecisionContext,
    ) -> StudioDecisionOutput:
        """Create staff-only artist, action, and price recommendations."""
        ranked = self._rank_examples(analysis, context)
        artist, artist_ids = self._suggest_artist(analysis, context, ranked)
        estimate = self._pricing_estimator.estimate(
            analysis=analysis,
            context=context,
            artist_key=artist.artist_key,
            history_examples=[example for _, example in ranked],
        )
        safe_analysis = self._protect_internal_price(
            analysis=analysis,
            estimate=estimate,
            disclosure_approved=(
                context.human_approved_price_disclosure
            ),
        )
        action, action_ids = self._suggest_action(
            analysis=analysis,
            current_message=current_message,
            ranked_examples=ranked,
            price_estimate=estimate,
        )
        applied_ids = sorted(set(artist_ids).union(action_ids))
        return StudioDecisionOutput(
            analysis=safe_analysis,
            artist_suggestion=artist,
            suggested_next_action=action,
            internal_price_estimate=estimate,
            applied_history_example_ids=applied_ids,
        )

    def _protect_internal_price(
        self,
        analysis: AIExtractionOutput,
        estimate: InternalPriceEstimate | None,
        disclosure_approved: bool,
    ) -> AIExtractionOutput:
        """Replace drafts that expose an unapproved internal estimate."""
        if estimate is None or disclosure_approved:
            return analysis
        forbidden_amounts = {
            self._format_price_amount(estimate.price_range.minimum),
            self._format_price_amount(estimate.price_range.maximum),
        }
        if not any(
            self._draft_contains_amount(analysis.draft_reply, amount)
            for amount in forbidden_amounts
        ):
            return analysis

        payload = analysis.model_dump(mode="python")
        payload["draft_reply"] = (
            "A staff member will review your request and reply shortly."
        )
        return AIExtractionOutput.model_validate(payload)

    def _format_price_amount(self, amount: Decimal) -> str:
        """Format a decimal for deterministic draft-disclosure checks."""
        return format(amount.normalize(), "f")

    def _draft_contains_amount(self, draft: str, amount: str) -> bool:
        """Detect a complete numeric amount without substring collisions."""
        pattern = rf"(?<![\d.]){re.escape(amount)}(?![\d.])"
        return bool(re.search(pattern, draft))

    def build_learning_record(
        self,
        inquiry: TattooInquiryInput,
        context: StudioDecisionContext,
        decision: StudioDecisionOutput,
        feedback: StudioDecisionFeedback,
    ) -> StudioLearningRecord:
        """Build a serializable record for caller-managed persistence."""
        if feedback.final_artist_key is not None:
            available = {item.key for item in context.artist_options}
            if feedback.final_artist_key not in available:
                raise ValueError(
                    "feedback final artist must exist in artist_options."
                )
        return StudioLearningRecord(
            channel=context.channel,
            original_client_message=inquiry.current_message,
            recent_chat_history=inquiry.recent_chat_history[-7:],
            reference_image_urls=inquiry.new_image_urls,
            decision=decision,
            human_feedback=feedback,
        )

    def build_history_example(
        self,
        record: StudioLearningRecord,
        example_id: str,
    ) -> DecisionHistoryExample:
        """Convert a completed feedback record into verified evidence."""
        analysis = record.decision.analysis
        feedback = record.human_feedback
        return DecisionHistoryExample(
            example_id=example_id,
            channel=record.channel,
            style_tags=analysis.style_tags,
            placement=analysis.placement,
            size_estimate_cm=analysis.size_estimate_cm,
            color_preference=analysis.color_preference,
            original_ai_artist_key=(
                record.decision.artist_suggestion.artist_key
            ),
            final_artist_key=feedback.final_artist_key,
            original_ai_action=(
                record.decision.suggested_next_action.action
            ),
            final_action=feedback.final_action,
            ai_suggestion_outcome=feedback.ai_suggestion_outcome,
            correction_reason=feedback.correction_reason,
            approved_price_range=feedback.approved_price_range,
        )

    def ingest_new_record(self, record: StudioLearningRecord) -> None:
        """Convert a completed learning record and add it to vector search.

        Args:
            record: Validated studio decision and its human feedback.

        Raises:
            RuntimeError: If this engine has no injected vector store.
        """
        if self._vector_store is None:
            raise RuntimeError(
                "A VectorStoreManager is required to ingest learning records."
            )

        example = self.build_history_example(
            record=record,
            example_id=self._build_learning_example_id(record),
        )
        self._vector_store.add_records([example])
        # Backend developer persists this record to PostgreSQL.

    def _build_learning_example_id(
        self,
        record: StudioLearningRecord,
    ) -> str:
        """Create a stable identifier for idempotent backend persistence."""
        serialized_record = record.model_dump_json().encode("utf-8")
        record_digest = sha256(serialized_record).hexdigest()[:24]
        return f"learning-{record_digest}"

    def _rank_examples(
        self,
        analysis: AIExtractionOutput,
        context: StudioDecisionContext,
    ) -> list[tuple[int, DecisionHistoryExample]]:
        """Rank vector-retrieved candidates with deterministic scoring."""
        candidates = self._retrieve_candidate_examples(analysis, context)
        ranked_candidates: list[
            tuple[int, int, DecisionHistoryExample]
        ] = []
        for candidate_order, example in enumerate(candidates):
            score = self._example_score(example, analysis, context.channel)
            if score >= 3:
                ranked_candidates.append((score, candidate_order, example))
        ranked_candidates.sort(
            key=lambda item: (-item[0], item[1], item[2].example_id)
        )
        return [
            (score, example)
            for score, _, example in ranked_candidates
        ]

    def _retrieve_candidate_examples(
        self,
        analysis: AIExtractionOutput,
        context: StudioDecisionContext,
    ) -> list[DecisionHistoryExample]:
        """Retrieve semantic candidates or use request-scoped history."""
        if self._vector_store is None:
            return list(context.decision_history)

        query_text = self._build_vector_query(analysis)
        try:
            matches = self._vector_store.search_similar_cases(
                query_text,
                top_k=_VECTOR_SEARCH_TOP_K,
            )
        except Exception as exc:  # pragma: no cover - defensive branch
            LOGGER.warning("Decision vector search failed: %s", exc)
            return list(context.decision_history)
        return self._deduplicate_examples(
            [
                *context.decision_history,
                *(example for _, example in matches),
            ]
        )

    def _build_vector_query(self, analysis: AIExtractionOutput) -> str:
        """Render current tattoo features in the indexed record format."""
        styles = ", ".join(analysis.style_tags) or "unknown"
        placement = analysis.placement or "unknown"
        size = analysis.size_estimate_cm or "unknown"
        color = analysis.color_preference or "unknown"
        return (
            f"Tattoo styles: {styles}. Placement: {placement}. "
            f"Size in centimeters: {size}. Color preference: {color}."
        )

    def _deduplicate_examples(
        self,
        examples: list[DecisionHistoryExample],
    ) -> list[DecisionHistoryExample]:
        """Keep the strongest vector match for each persistent example ID."""
        unique: list[DecisionHistoryExample] = []
        seen_ids: set[str] = set()
        for example in examples:
            if example.example_id in seen_ids:
                continue
            seen_ids.add(example.example_id)
            unique.append(example)
        return unique

    def _example_score(
        self,
        example: DecisionHistoryExample,
        analysis: AIExtractionOutput,
        channel: str,
    ) -> int:
        """Score an example using style, placement, size, and channel."""
        overlap = set(example.style_tags).intersection(analysis.style_tags)
        same_placement = bool(
            example.placement
            and analysis.placement
            and example.placement.lower() == analysis.placement.lower()
        )
        if not overlap and not same_placement:
            return 0

        score = len(overlap) * 4
        score += int(same_placement) * 2
        score += int(example.channel == channel)
        score += int(
            bool(example.color_preference)
            and example.color_preference.lower()
            == analysis.color_preference.lower()
        )
        current_size = parse_size_cm(analysis.size_estimate_cm)
        previous_size = parse_size_cm(example.size_estimate_cm)
        if current_size is not None and previous_size is not None:
            score += self._size_score(current_size, previous_size)
        return score

    def _size_score(self, current: Decimal, previous: Decimal) -> int:
        """Return a small similarity bonus for comparable sizes."""
        difference = abs(current - previous)
        if difference <= Decimal("2"):
            return 2
        if difference <= Decimal("5"):
            return 1
        return 0

    def _suggest_artist(
        self,
        analysis: AIExtractionOutput,
        context: StudioDecisionContext,
        ranked: list[tuple[int, DecisionHistoryExample]],
    ) -> tuple[ArtistSuggestion, list[str]]:
        """Prefer verified final assignments when evidence is unambiguous."""
        artist_map = {
            item.key: item
            for item in context.artist_options
            if self._artist_config.validate_artist_assignment(
                item.key,
                analysis.style_tags,
            )
        }
        votes: dict[str, int] = defaultdict(int)
        support: dict[str, list[str]] = defaultdict(list)
        for score, example in ranked:
            key = example.final_artist_key
            if key is None or key not in artist_map:
                continue
            votes[key] += score
            support[key].append(example.example_id)

        if votes:
            best = max(votes.values())
            winners = [key for key, score in votes.items() if score == best]
            if len(winners) == 1:
                key = winners[0]
                artist = artist_map[key]
                confidence: ConfidenceLevel = "medium"
                if best >= 8 or len(support[key]) > 1:
                    confidence = "high"
                return (
                    ArtistSuggestion(
                        artist_key=key,
                        artist_name=artist.display_name,
                        confidence_level=confidence,
                        reasoning=(
                            "Verified similar decisions support this artist."
                        ),
                        source="verified_history",
                    ),
                    support[key],
                )
        return self._default_artist(analysis, context.artist_options), []

    def _default_artist(
        self,
        analysis: AIExtractionOutput,
        options: list[ArtistOption],
    ) -> ArtistSuggestion:
        """Map legacy routing output into the supplied artist catalog."""
        if analysis.suggested_artist != "Unclear":
            for artist in options:
                name_matches = artist.display_name.casefold() == (
                    analysis.suggested_artist.casefold()
                )
                is_eligible = self._artist_config.validate_artist_assignment(
                    artist.key,
                    analysis.style_tags,
                )
                if name_matches and is_eligible:
                    return ArtistSuggestion(
                        artist_key=artist.key,
                        artist_name=artist.display_name,
                        confidence_level=analysis.confidence_level,
                        reasoning=analysis.ai_reasoning,
                        source="default_rules",
                    )
        return ArtistSuggestion(
            artist_key=None,
            artist_name=None,
            confidence_level="low",
            reasoning="No unambiguous artist assignment evidence was found.",
            source="unresolved",
        )

    def _suggest_action(
        self,
        analysis: AIExtractionOutput,
        current_message: str,
        ranked_examples: list[tuple[int, DecisionHistoryExample]],
        price_estimate: InternalPriceEstimate | None,
    ) -> tuple[NextAction, list[str]]:
        """Choose a verified historical action or a safe default."""
        if self._contains_pricing_request(current_message):
            return (
                NextAction(
                    action="pricing_review",
                    reason=(
                        "Pricing requires studio approval before client use."
                    ),
                    priority="high",
                ),
                [],
            )
        if price_estimate is not None and price_estimate.requires_consultation:
            return (
                NextAction(
                    action="offer_consultation",
                    reason="The matching pricing rule requires consultation.",
                ),
                [],
            )

        votes: dict[DecisionAction, int] = defaultdict(int)
        support: dict[DecisionAction, list[str]] = defaultdict(list)
        for score, example in ranked_examples:
            votes[example.final_action] += score
            support[example.final_action].append(example.example_id)
        if votes:
            best = max(votes.values())
            winners = [key for key, score in votes.items() if score == best]
            if len(winners) == 1:
                action = winners[0]
                return (
                    NextAction(
                        action=action,
                        reason=(
                            "Verified similar decisions support this action."
                        ),
                    ),
                    support[action],
                )
        return self._default_action(analysis), []

    def _default_action(self, analysis: AIExtractionOutput) -> NextAction:
        """Derive a conservative action from the existing analysis."""
        if analysis.risk_level == "high":
            return NextAction(
                action="artist_review",
                reason="High-risk requests require manual studio review.",
                priority="high",
            )
        if analysis.missing_information:
            missing = ", ".join(analysis.missing_information)
            return NextAction(
                action="request_more_information",
                reason=f"Request missing details before proceeding: {missing}.",
            )
        if analysis.suggested_artist == "Unclear":
            return NextAction(
                action="offer_consultation",
                reason="Artist suitability remains unclear from current details.",
            )
        return NextAction(
            action="ready_to_book",
            reason="Request is complete enough for staff booking review.",
        )

    def _contains_pricing_request(self, current_message: str) -> bool:
        """Detect direct client requests for price information."""
        normalized = current_message.lower()
        return any(term in normalized for term in _PRICE_TERMS)
