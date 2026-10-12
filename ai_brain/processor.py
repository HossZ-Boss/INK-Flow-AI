"""Unified AI Brain processor used by backend services."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from .decision import StudioDecisionEngine
from .decision_schemas import (
    ArtistSuggestion,
    DecisionHistoryExample,
    NextAction,
    StudioDecisionContext,
    StudioDecisionFeedback,
    StudioDecisionOutput,
    StudioLearningRecord,
)
from .errors import AnalysisPipelineError
from .extraction import TattooTextExtractor
from .llm_health import llm_quota_exhausted, reset_llm_health
from .outlook_classification import OutlookInquiryClassifier
from .price_suggestion import TattooPriceSuggester
from .routing import TattooRouter
from .schemas import (
    AIExtractionOutput,
    Message,
    MessageSource,
    PriceSuggestion,
    TattooInquiryInput,
)
from .vision import TattooVisionAnalyzer

if TYPE_CHECKING:
    from .vector_store import VectorStoreManager

LOGGER = logging.getLogger(__name__)

_STATE_IMAGE_KEYS = ("reference_images", "previous_image_urls", "image_urls")

_QUOTA_EXHAUSTED_REASON = (
    "AI provider quota exhausted: no reply was generated. "
    "A staff member must reply manually."
)
_AUTOMATION_PAUSED_DRAFT = "A staff member will reply to you shortly."
_AUTOMATION_PAUSED_EMAIL_DRAFT = (
    "Hello,\n\n"
    "Thank you for contacting Tattoo Hysteria. We have received your "
    "message. A member of our studio team will review it and reply as soon as "
    "possible.\n\n"
    "Kind regards,\n"
    "Tattoo Hysteria"
)


class StudioAIBrain:
    """Single entry-point for tattoo inquiry analysis."""

    def __init__(
        self,
        vision_analyzer: TattooVisionAnalyzer | None = None,
        text_extractor: TattooTextExtractor | None = None,
        router: TattooRouter | None = None,
        decision_engine: StudioDecisionEngine | None = None,
        outlook_classifier: OutlookInquiryClassifier | None = None,
        price_suggester: TattooPriceSuggester | None = None,
        vector_store: VectorStoreManager | None = None,
    ) -> None:
        """Build the pipeline; ``vector_store`` is the shared learning index."""
        self.vision_analyzer = vision_analyzer or TattooVisionAnalyzer()
        self.text_extractor = text_extractor or TattooTextExtractor()
        self.router = router or TattooRouter(vector_store=vector_store)
        self.decision_engine = decision_engine or StudioDecisionEngine(
            vector_store=vector_store
        )
        self.outlook_classifier = (
            outlook_classifier or OutlookInquiryClassifier()
        )
        chat_model = getattr(self.router, "chat_model", None)
        self.price_suggester = price_suggester or (
            TattooPriceSuggester(
                llm=chat_model,
                vector_store=vector_store,
                image_loader=getattr(
                    self.vision_analyzer,
                    "load_images_as_data_uris",
                    None,
                ),
            )
            if chat_model is not None
            else None
        )

    def process_inquiry(
        self,
        current_message: str | None = None,
        new_image_urls: list[str] | None = None,
        existing_db_state: dict[str, Any] | None = None,
        recent_chat_history: list[Message] | None = None,
        message_source: MessageSource | str | None = None,
        *,
        text: str | None = None,
        image_urls: list[str] | None = None,
    ) -> AIExtractionOutput:
        """Process latest-message and supplied hybrid context.

        The text and image_urls keywords remain temporarily supported until
        the API endpoint migrates to the canonical payload in Step 4.
        """
        inquiry = self._build_inquiry(
            current_message=current_message,
            new_image_urls=new_image_urls,
            existing_db_state=existing_db_state,
            recent_chat_history=recent_chat_history,
            message_source=message_source,
            text=text,
            image_urls=image_urls,
        )
        if self._is_automation_paused(inquiry):
            return self._build_automation_paused_output(inquiry)

        reset_llm_health()
        outlook_classification = self.outlook_classifier.classify(inquiry)
        if (
            inquiry.message_source == "outlook"
            and not outlook_classification.is_tattoo_inquiry
        ):
            return self._build_suppressed_outlook_output(
                inquiry,
                outlook_classification.reason,
            )

        vision_output = self.vision_analyzer.analyze_images(
            inquiry.new_image_urls
        )
        extracted = self.text_extractor.extract(
            current_message=inquiry.current_message,
            style_tags=vision_output.style_tags,
            visual_color_preference=vision_output.color_preference,
            visual_subjects=vision_output.design_subjects,
            visual_description=vision_output.visual_description,
            new_image_urls=inquiry.new_image_urls,
            existing_db_state=inquiry.existing_db_state,
            recent_chat_history=inquiry.recent_chat_history,
        )
        analysis = self.router.route(
            extracted,
            current_message=inquiry.current_message,
            recent_chat_history=inquiry.recent_chat_history,
            existing_db_state=inquiry.existing_db_state,
            message_source=inquiry.message_source,
        )
        if llm_quota_exhausted():
            return self._suppress_reply_for_quota(analysis)
        if analysis.risk_level == "high":
            analysis = analysis.model_copy(
                update={
                    "price_suggestion": self._suggest_price(
                        inquiry,
                        analysis,
                        vision_output.visual_description,
                    )
                }
            )
        is_reply_channel = inquiry.message_source == "whatsapp" or (
            inquiry.message_source == "outlook"
            and outlook_classification.is_tattoo_inquiry
        )
        return analysis.model_copy(
            update={
                "auto_reply": (
                    is_reply_channel and analysis.auto_reply_allowed
                )
            }
        )

    def process_studio_decision(
        self,
        inquiry: TattooInquiryInput,
        context: StudioDecisionContext,
    ) -> StudioDecisionOutput:
        """Create an internal decision from caller-supplied studio context."""
        analysis = self.process_inquiry(
            current_message=inquiry.current_message,
            new_image_urls=inquiry.new_image_urls,
            existing_db_state=inquiry.existing_db_state,
            recent_chat_history=inquiry.recent_chat_history,
            message_source=context.channel,
        )
        if self._is_automation_paused(inquiry):
            return self._build_automation_paused_decision(analysis)
        return self.decision_engine.decide(
            analysis=analysis,
            current_message=inquiry.current_message,
            context=context,
        )

    def _suggest_price(
        self,
        inquiry: TattooInquiryInput,
        analysis: AIExtractionOutput,
        visual_description: str,
    ) -> PriceSuggestion | None:
        """Return a staff-only price range without ever failing the request."""
        if self.price_suggester is None:
            return None
        try:
            return self.price_suggester.suggest(
                analysis,
                image_urls=self._reference_image_urls(inquiry),
                visual_description=visual_description,
                channel=inquiry.message_source,
            )
        except Exception:
            LOGGER.exception("Price suggestion failed; continuing without it.")
            return None

    def _reference_image_urls(self, inquiry: TattooInquiryInput) -> list[str]:
        """Collect new and previously stored reference image URLs."""
        urls = list(inquiry.new_image_urls)
        state = inquiry.existing_db_state
        for record in (state, state.get("intake"), state.get("lead")):
            if not isinstance(record, dict):
                continue
            for key in _STATE_IMAGE_KEYS:
                value = record.get(key)
                if isinstance(value, str):
                    value = [value]
                if isinstance(value, list):
                    urls.extend(
                        item.strip()
                        for item in value
                        if isinstance(item, str)
                        and item.strip().startswith(("http://", "https://"))
                    )
        return list(dict.fromkeys(urls))

    def _suppress_reply_for_quota(
        self,
        analysis: AIExtractionOutput,
    ) -> AIExtractionOutput:
        """Send nothing to the client when the AI provider has no quota."""
        return analysis.model_copy(
            update={
                "draft_reply": "",
                "auto_reply_allowed": False,
                "auto_reply": False,
                "staff_review_required": True,
                "telegram_review_required": True,
                "review_reasons": [
                    *analysis.review_reasons,
                    _QUOTA_EXHAUSTED_REASON,
                ][-20:],
                "ai_reasoning": _QUOTA_EXHAUSTED_REASON,
            }
        )

    def _is_automation_paused(self, inquiry: TattooInquiryInput) -> bool:
        """Return whether backend state explicitly disables automation."""
        return inquiry.existing_db_state.get("automation_paused") is True

    def _build_automation_paused_output(
        self,
        inquiry: TattooInquiryInput,
    ) -> AIExtractionOutput:
        """Return a deterministic hold response without calling AI services."""
        return AIExtractionOutput(
            client_name="",
            tattoo_idea=inquiry.current_message,
            style_tags=["unknown"],
            placement="",
            size_estimate_cm="",
            color_preference="",
            date="",
            time="",
            preferred_artist="",
            appointment_type="",
            availability="",
            tattoo_project_type="",
            suggested_artist="Unclear",
            confidence_level="low",
            ai_reasoning=(
                "Automation paused: manual staff response required."
            ),
            missing_information=[],
            risk_level="high",
            draft_reply=(
                _AUTOMATION_PAUSED_EMAIL_DRAFT
                if inquiry.message_source == "outlook"
                else _AUTOMATION_PAUSED_DRAFT
            ),
            auto_reply_allowed=False,
            auto_reply=False,
            telegram_review_required=True,
        )

    def _build_suppressed_outlook_output(
        self,
        inquiry: TattooInquiryInput,
        reason: str,
    ) -> AIExtractionOutput:
        """Return an empty draft for Outlook mail outside tattoo intake."""
        return AIExtractionOutput(
            client_name=self._state_client_name(inquiry.existing_db_state),
            tattoo_idea="",
            style_tags=["unknown"],
            placement="",
            size_estimate_cm="",
            color_preference="",
            suggested_artist="Unclear",
            confidence_level="high",
            ai_reasoning=reason,
            missing_information=[],
            risk_level="low",
            draft_reply="",
            auto_reply_allowed=False,
            auto_reply=False,
            telegram_review_required=False,
        )

    def _state_client_name(self, state: dict[str, Any]) -> str:
        """Read a client name without treating unrelated mail as intake."""
        for record in (state.get("lead"), state.get("intake"), state):
            if not isinstance(record, dict):
                continue
            for key in ("name", "client_name", "lead_name"):
                value = record.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return ""

    def _build_automation_paused_decision(
        self,
        analysis: AIExtractionOutput,
    ) -> StudioDecisionOutput:
        """Return a staff-only decision without retrieval or pricing work."""
        reasoning = "Automation paused: manual staff response required."
        return StudioDecisionOutput(
            analysis=analysis,
            artist_suggestion=ArtistSuggestion(
                artist_key=None,
                artist_name=None,
                confidence_level="low",
                reasoning=reasoning,
                source="unresolved",
            ),
            suggested_next_action=NextAction(
                action="artist_review",
                reason=reasoning,
                priority="high",
                requires_human_review=True,
            ),
            internal_price_estimate=None,
            applied_history_example_ids=[],
        )

    def build_learning_record(
        self,
        inquiry: TattooInquiryInput,
        context: StudioDecisionContext,
        decision: StudioDecisionOutput,
        feedback: StudioDecisionFeedback,
    ) -> StudioLearningRecord:
        """Build a feedback record without storing or transmitting it."""
        return self.decision_engine.build_learning_record(
            inquiry=inquiry,
            context=context,
            decision=decision,
            feedback=feedback,
        )

    def build_history_example(
        self,
        record: StudioLearningRecord,
        example_id: str,
    ) -> DecisionHistoryExample:
        """Convert a completed record into future decision evidence."""
        return self.decision_engine.build_history_example(
            record=record,
            example_id=example_id,
        )

    def _build_inquiry(
        self,
        current_message: str | None,
        new_image_urls: list[str] | None,
        existing_db_state: dict[str, Any] | None,
        recent_chat_history: list[Message] | None,
        message_source: MessageSource | str | None,
        text: str | None,
        image_urls: list[str] | None,
    ) -> TattooInquiryInput:
        """Validate canonical values with temporary legacy fallbacks."""
        resolved_message = current_message
        if resolved_message is None:
            resolved_message = text

        resolved_images = new_image_urls
        if resolved_images is None:
            resolved_images = image_urls or []

        payload: dict[str, Any] = {
            "current_message": resolved_message,
            "new_image_urls": resolved_images,
            "existing_db_state": existing_db_state or {},
            "recent_chat_history": recent_chat_history or [],
        }
        if message_source is not None:
            payload["message_source"] = message_source
        try:
            return TattooInquiryInput.model_validate(payload)
        except ValidationError as exc:
            raise AnalysisPipelineError("Invalid inquiry payload.") from exc


if __name__ == "__main__":
    brain = StudioAIBrain()
    mock_text = (
        "Hey, I want a tiny fine-line lotus on my inner wrist. "
        "Maybe around 4 cm. Black ink only. Can I come next Tuesday?"
    )
    mock_images = [
        "https://example.com/reference-1.jpg",
        "https://example.com/reference-2.jpg",
    ]

    try:
        output = brain.process_inquiry(
            current_message=mock_text,
            new_image_urls=mock_images,
            existing_db_state={},
            recent_chat_history=[],
        )
        print(output.model_dump_json(indent=2))
    except Exception as exc:
        print(f"Processor test run failed: {exc}")
