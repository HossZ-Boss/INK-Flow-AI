"""Transport-specific response schemas for the HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator

from ai_brain.decision_schemas import (
    DecisionAction,
    DecisionOutcome,
    InquiryChannel,
    MoneyRange,
)
from ai_brain.schemas import (
    AIExtractionOutput,
    Message,
    PriceSuggestion,
    RiskLevel,
    StyleTag,
    TattooInquiryInput,
)

__all__ = [
    "AIExtractionOutput",
    "ErrorResponse",
    "HealthResponse",
    "LearningFeedbackInput",
    "LearningFeedbackResponse",
    "LearningStatusResponse",
    "TattooInquiryInput",
    "TelegramSummaryInput",
    "TelegramSummaryResponse",
]


class HealthResponse(BaseModel):
    """Service health or readiness response."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "ready"] = Field(
        description="Current service state.",
    )
    service: str = Field(
        description="Stable service identifier.",
    )
    version: str = Field(
        description="Deployed API version.",
    )


class ErrorResponse(BaseModel):
    """Standard API error response."""

    model_config = ConfigDict(extra="forbid")

    detail: str = Field(
        description="Safe, client-facing error explanation.",
    )


class TelegramSummaryInput(TattooInquiryInput):
    """Summary input with compatibility for the legacy backend payload."""

    current_message: str = Field(
        default="",
        description=(
            "Latest client text, or the latest user history message when "
            "omitted."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_payload(cls, value: Any) -> Any:
        """Map legacy summary keys and infer the latest user message."""
        if not isinstance(value, dict):
            return value

        data = dict(value)
        current_message = data.get("current_message")
        for alias in ("message", "text"):
            alias_value = data.pop(alias, None)
            if not current_message and alias_value:
                current_message = alias_value
        if current_message is not None:
            data["current_message"] = current_message

        history = data.get("recent_chat_history")
        for alias in ("chat_history", "history"):
            alias_value = data.pop(alias, None)
            if history is None and alias_value is not None:
                history = alias_value
        if history is not None:
            data["recent_chat_history"] = history

        lead = data.pop("lead", None)
        if "existing_db_state" not in data and lead is not None:
            data["existing_db_state"] = lead

        if not data.get("current_message") and isinstance(history, (list, tuple)):
            for item in reversed(history):
                if isinstance(item, Message):
                    role = item.role
                    content = item.content
                elif isinstance(item, dict):
                    role = item.get("role")
                    content = item.get("content")
                else:
                    continue
                if role == "user" and isinstance(content, str) and content.strip():
                    data["current_message"] = content
                    break

        data.setdefault("current_message", "")
        return data


class TelegramSummaryResponse(BaseModel):
    """Staff-review summary plus the separately generated client draft."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    risk_level: RiskLevel
    staff_review_required: bool
    review_reasons: list[StrictStr] = Field(default_factory=list, max_length=20)
    reference_image_urls: list[StrictStr] = Field(
        default_factory=list,
        max_length=20,
    )
    summary: StrictStr = Field(min_length=1, max_length=5000)
    draft_reply: StrictStr = Field(default="", max_length=2000)
    telegram_message: StrictStr = Field(min_length=1, max_length=7000)
    price_suggestion: PriceSuggestion | None = Field(
        default=None,
        description="Staff-only AI price suggestion; not approved for clients.",
    )


class LearningFeedbackInput(BaseModel):
    """Staff-reviewed outcome the backend sends after a request is handled."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    inquiry_id: StrictStr | None = Field(
        default=None,
        max_length=60,
        pattern=r"^[A-Za-z0-9._-]+$",
        description=(
            "Backend request ID. Re-sending the same ID with the same content "
            "is ignored; different content for the same ID is rejected."
        ),
    )
    channel: InquiryChannel = "whatsapp"
    style_tags: list[StyleTag] = Field(default_factory=list, max_length=10)
    placement: StrictStr = Field(default="", max_length=100)
    size_estimate_cm: StrictStr = Field(default="", max_length=100)
    color_preference: StrictStr = Field(default="", max_length=100)
    ai_suggested_artist: StrictStr = Field(
        default="",
        max_length=80,
        description="Artist the AI suggested (display name or key).",
    )
    final_artist: StrictStr = Field(
        default="",
        max_length=80,
        description="Artist staff assigned (display name or key).",
    )
    ai_suggested_action: DecisionAction = "artist_review"
    final_action: DecisionAction = "accept_request"
    ai_suggestion_outcome: DecisionOutcome = Field(
        description="correct, corrected, or rejected.",
    )
    correction_reason: StrictStr | None = Field(default=None, max_length=500)
    approved_price_range: MoneyRange | None = Field(
        default=None,
        description="Price staff approved, used to anchor future estimates.",
    )


class LearningFeedbackResponse(BaseModel):
    """Result of adding one staff-reviewed example to the learning index."""

    model_config = ConfigDict(extra="forbid")

    example_id: str
    record_count: int


class LearningStatusResponse(BaseModel):
    """Current size of the learning index."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool
    record_count: int
    records_with_approved_price: int
