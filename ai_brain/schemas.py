"""Strict data contracts for the Tattoo Studio AI Brain."""

from __future__ import annotations

from datetime import date as calendar_date
from datetime import datetime
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from .email_cleaning import strip_quoted_email_content

STYLE_TAG_OPTIONS: tuple[str, ...] = (
    "fine-line",
    "watercolor",
    "minimal",
    "floral",
    "botanical",
    "illustrative",
    "fine-illustrative",
    "abstract",
    "blackwork",
    "realism",
    "micro-realism",
    "pixel-art",
    "new-school",
    "neo-traditional",
    "black-and-grey",
    "calligraphy",
    "traditional",
    "geometric",
    "unknown",
)

MISSING_INFORMATION_OPTIONS: tuple[str, ...] = (
    "client full name",
    "tattoo idea",
    "size in cm",
    "placement",
    "color preference",
    "tattoo style",
    "reference images",
    "preferred artist",
    "appointment type",
    "preferred dates or availability",
    "tattoo project type",
)

ARTIST_PREFERENCE_OPTIONS: tuple[str, ...] = (
    "Lana",
    "Nina",
    "Hossam",
    "Sliva",
    "Sandra",
    "Mila",
)

MESSAGE_SOURCE_OPTIONS: tuple[str, ...] = (
    "whatsapp",
    "outlook",
    "vcita",
    "other",
)
_MESSAGE_SOURCE_ALIASES = {
    "email": "outlook",
    "microsoft outlook": "outlook",
    "ms outlook": "outlook",
}

StyleTag = Literal[
    "fine-line",
    "watercolor",
    "minimal",
    "floral",
    "botanical",
    "illustrative",
    "fine-illustrative",
    "abstract",
    "blackwork",
    "realism",
    "micro-realism",
    "pixel-art",
    "new-school",
    "neo-traditional",
    "black-and-grey",
    "calligraphy",
    "traditional",
    "geometric",
    "unknown",
]

SuggestedArtist = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=80,
    ),
]
ConfidenceLevel = Literal["high", "medium", "low"]
RiskLevel = Literal["low", "high"]
MessageSource = Literal["whatsapp", "outlook", "vcita", "other"]
VisualColorPreference = Literal["black-and-grey", "color", "unknown"]
SizeStatus = Literal["exact", "approximate", "unknown"]
ArtistPreferenceMode = Literal[
    "specific",
    "recommend",
    "no_preference",
    "unknown",
]
ClientIntent = Literal[
    "continue_intake",
    "pricing_question",
    "size_guidance",
    "artist_guidance",
    "availability_question",
    "status_update",
    "complaint",
    "withdrawal",
]
ConversationStatus = Literal["active", "closed"]
IntakeStatus = Literal[
    "collecting_info",
    "needs_staff_review",
    "ready_for_review",
    "closed",
]
PreferredArtist = Literal[
    "",
    "Hossam",
    "Hoss",
    "Nina",
    "Lana",
    "Sandra",
    "Sliva",
    "Silva",
    "Mila",
    "No preference",
]
AppointmentType = Literal["", "online", "studio_visit"]
TattooProjectType = Literal[
    "",
    "new tattoo",
    "cover-up",
    "continuation",
    "touch-up",
]

MissingInformationItem = Literal[
    "client full name",
    "tattoo idea",
    "size in cm",
    "placement",
    "color preference",
    "tattoo style",
    "reference images",
    "preferred artist",
    "appointment type",
    "preferred dates or availability",
    "tattoo project type",
    # Legacy values remain accepted while existing backend records migrate.
    "preferred date",
    "preferred time",
]


def _canonical_preferred_artist(value: object) -> object:
    """Normalize legacy studio artist names without rejecting old records."""
    if not isinstance(value, str):
        return value
    aliases = {
        "hoss": "Hossam",
        "silva": "Sliva",
    }
    normalized = value.strip()
    return aliases.get(normalized.casefold(), normalized)


class Message(BaseModel):
    """Single recent conversation message supplied by the backend."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    role: Literal["user", "assistant"] = Field(
        description="Conversation participant that produced the message.",
    )
    content: str = Field(
        min_length=1,
        description="Message text used to resolve conversational context.",
    )


class TattooInquiryInput(BaseModel):
    """Hybrid-context inquiry payload assembled by the backend."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    current_message: str = Field(
        description=(
            "Latest client text, or an empty string for an image-only message."
        ),
    )
    new_image_urls: list[str] = Field(
        default_factory=list,
        description="Image URLs attached only to the latest client message.",
    )
    existing_db_state: dict[str, Any] = Field(
        default_factory=dict,
        description="Previously collected inquiry data supplied by the backend.",
    )
    recent_chat_history: list[Message] = Field(
        default_factory=list,
        max_length=30,
        description="Up to 30 recent messages used for context resolution.",
    )
    message_source: MessageSource = Field(
        default="whatsapp",
        description=(
            "Source channel used to select WhatsApp chat or Outlook email "
            "reply formatting."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def infer_nested_message_source(cls, value: Any) -> Any:
        """Use the backend's nested lead/intake source when not explicit."""
        if not isinstance(value, dict) or "message_source" in value:
            return value

        existing_db_state = value.get("existing_db_state")
        if not isinstance(existing_db_state, dict):
            return value

        candidates: list[Any] = []
        for key in ("intake", "lead"):
            record = existing_db_state.get(key)
            if isinstance(record, dict):
                candidates.append(record.get("source"))
        candidates.append(existing_db_state.get("source"))

        for candidate in candidates:
            normalized = cls._normalize_source_value(candidate)
            if normalized in MESSAGE_SOURCE_OPTIONS:
                data = dict(value)
                data["message_source"] = normalized
                return data
        return value

    @field_validator("message_source", mode="before")
    @classmethod
    def normalize_message_source(cls, value: Any) -> Any:
        """Accept source names without making casing a backend concern."""
        return cls._normalize_source_value(value)

    @staticmethod
    def _normalize_source_value(value: Any) -> Any:
        """Normalize known backend source values and email aliases."""
        if not isinstance(value, str):
            return value
        normalized = value.strip().casefold()
        return _MESSAGE_SOURCE_ALIASES.get(normalized, normalized)

    @field_validator("new_image_urls")
    @classmethod
    def normalize_image_urls(cls, value: list[str]) -> list[str]:
        """Drop empty URL items and trim surrounding whitespace."""
        return [item.strip() for item in value if item and item.strip()]

    @field_validator("recent_chat_history", mode="before")
    @classmethod
    def keep_latest_chat_history(cls, value: Any) -> Any:
        """Drop empty media-only text entries and keep the latest 30."""
        if not isinstance(value, (list, tuple)):
            return value

        non_empty_messages: list[Any] = []
        for item in value:
            if isinstance(item, Message):
                if item.content.strip():
                    non_empty_messages.append(item)
                continue
            if isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, str) and not content.strip():
                    continue
            non_empty_messages.append(item)
        return non_empty_messages[-30:]

    @model_validator(mode="after")
    def strip_outlook_quoted_content(self) -> Self:
        """Exclude quoted email threads from new client input and history."""
        if self.message_source != "outlook":
            return self

        self.current_message = strip_quoted_email_content(
            self.current_message
        )
        cleaned_history: list[Message] = []
        for message in self.recent_chat_history:
            if message.role != "user":
                cleaned_history.append(message)
                continue
            cleaned_content = strip_quoted_email_content(message.content)
            if cleaned_content:
                cleaned_history.append(
                    message.model_copy(update={"content": cleaned_content})
                )
        self.recent_chat_history = cleaned_history
        return self

    @model_validator(mode="after")
    def require_text_or_image(self) -> Self:
        """Require usable text or at least one attached image URL."""
        if not self.current_message and not self.new_image_urls:
            raise ValueError(
                "current_message or at least one new_image_url is required."
            )
        return self


class TattooVisionOutput(BaseModel):
    """Strict visual signals extracted from client reference images."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    style_tags: list[StyleTag] = Field(min_length=1)
    color_preference: VisualColorPreference
    design_subjects: list[str] = Field(default_factory=list, max_length=10)
    visual_description: str = Field(default="", max_length=500)


class TattooProjectDetail(BaseModel):
    """One tattoo request within an individual or group inquiry."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    person_label: str = Field(
        default="Client",
        max_length=80,
        description="Person receiving this tattoo, such as Client or Girlfriend.",
    )
    tattoo_idea: str = Field(default="", max_length=500)
    style_tags: list[StyleTag] = Field(default_factory=list, max_length=10)
    placement: str = Field(default="", max_length=100)
    size_estimate_cm: str = Field(default="", max_length=100)
    size_description: str = Field(default="", max_length=100)
    size_status: SizeStatus = "unknown"
    color_preference: str = Field(default="", max_length=100)
    tattoo_project_type: TattooProjectType = ""
    reference_image_urls: list[str] = Field(default_factory=list, max_length=20)


class TattooExtractionDraft(BaseModel):
    """Intermediate extraction output before routing and risk enrichment."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    client_name: str = Field(
        default="",
        description="Client's full name.",
    )
    tattoo_idea: str = Field(
        description="Core tattoo concept extracted from client text.",
    )
    style_tags: list[StyleTag] = Field(
        default_factory=list,
        description="Tattoo style tags from vision analysis.",
    )
    placement: str = Field(
        description="Requested body placement for the tattoo.",
    )
    size_estimate_cm: str = Field(
        description="Approximate tattoo size in centimeters.",
    )
    color_preference: str = Field(
        description="Client color preference for the tattoo.",
    )
    date: str = Field(
        default="",
        description="Preferred appointment date in YYYY-MM-DD format.",
    )
    time: str = Field(
        default="",
        description="Preferred appointment time in 24-hour HH:MM format.",
    )
    preferred_artist: PreferredArtist = Field(
        default="",
        description="Client's preferred artist or No preference.",
    )
    appointment_type: AppointmentType = Field(
        default="",
        description="Online or studio-visit appointment preference.",
    )
    availability: str = Field(
        default="",
        description="Preferred dates or general availability.",
    )
    date_to_confirm: str = Field(
        default="",
        description=(
            "YYYY-MM-DD date inferred from a bare weekday that the client "
            "must confirm before it is saved as the preferred date."
        ),
    )
    tattoo_project_type: TattooProjectType = Field(
        default="",
        description="New tattoo, cover-up, continuation, or touch-up.",
    )
    party_size: int = Field(
        default=1,
        ge=1,
        le=20,
        description="Number of people represented by this inquiry.",
    )
    multi_entity_detected: bool = Field(default=False)
    complexity_notes: str = Field(
        default="",
        description=(
            "Notes on complex requests, e.g., multiple people or colors."
        ),
    )
    projects: list[TattooProjectDetail] = Field(
        default_factory=list,
        max_length=20,
        description="Per-person or per-tattoo details for group requests.",
    )
    size_description: str = Field(
        default="",
        max_length=100,
        description="Natural size wording such as hand-sized or not sure.",
    )
    size_status: SizeStatus = "unknown"
    artist_preference_mode: ArtistPreferenceMode = "unknown"
    pricing_requested: bool = False
    client_intent: ClientIntent = "continue_intake"
    conversation_status: ConversationStatus = "active"
    missing_information: list[MissingInformationItem] = Field(
        default_factory=list,
        description=(
            "Missing intake items from the required checklist: "
            "client full name, tattoo idea, size in cm, placement, color "
            "preference, tattoo style, reference images, preferred artist, "
            "appointment type, preferred dates or availability, tattoo project "
            "type."
        ),
    )

    @field_validator("date", "date_to_confirm")
    @classmethod
    def validate_date_format(cls, value: str) -> str:
        """Require a real calendar date in the public YYYY-MM-DD format."""
        return _validate_preferred_date(value)

    @field_validator("preferred_artist", mode="before")
    @classmethod
    def normalize_preferred_artist(cls, value: object) -> object:
        """Convert legacy artist names to current client-facing names."""
        return _canonical_preferred_artist(value)

    @field_validator("time")
    @classmethod
    def validate_time_format(cls, value: str) -> str:
        """Require a real clock time in the public 24-hour HH:MM format."""
        return _validate_preferred_time(value)


PriceSuggestionBasis = Literal[
    "ai_market_estimate",
    "ai_with_approved_history",
    "approved_history",
]


class PriceSuggestion(BaseModel):
    """Staff-only price range suggested for a high-risk request."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    currency: Annotated[
        str,
        StringConstraints(pattern=r"^[A-Z]{3}$"),
    ] = "EUR"
    minimum: int = Field(ge=0, le=50000)
    maximum: int = Field(ge=0, le=50000)
    estimated_hours: str = Field(
        default="",
        max_length=40,
        description="Approximate tattooing time, e.g. 3-4 hours.",
    )
    complexity: Literal["low", "medium", "high", "unknown"] = "unknown"
    confidence_level: ConfidenceLevel = "low"
    basis: PriceSuggestionBasis = "ai_market_estimate"
    reasoning: str = Field(default="", max_length=600)
    similar_example_ids: list[str] = Field(
        default_factory=list,
        max_length=12,
    )
    internal_only: Literal[True] = True
    client_safe_to_share: Literal[False] = False

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        """Require the upper amount to be at least the lower amount."""
        if self.maximum < self.minimum:
            raise ValueError("maximum must be at least minimum.")
        return self

    def label(self) -> str:
        """Render the range for staff, e.g. "EUR 350-500"."""
        amount = (
            f"{self.minimum}"
            if self.minimum == self.maximum
            else f"{self.minimum}-{self.maximum}"
        )
        symbol = "€" if self.currency == "EUR" else f"{self.currency} "
        return f"{symbol}{amount}"


class AIExtractionOutput(BaseModel):
    """Final strict JSON contract returned to backend services."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    client_name: str = Field(
        default="",
        description="Client's full name.",
    )
    tattoo_idea: str = Field(
        description="Short summary of the tattoo request from the client.",
    )
    style_tags: list[StyleTag] = Field(
        min_length=1,
        description="Detected style tags from the approved style taxonomy.",
    )
    placement: str = Field(
        description="Body placement requested by the client.",
    )
    size_estimate_cm: str = Field(
        description="Tattoo size estimate expressed in centimeters.",
    )
    color_preference: str = Field(
        description="Color preference such as black-and-grey or full color.",
    )
    date: str = Field(
        default="",
        description="Preferred appointment date in YYYY-MM-DD format.",
    )
    time: str = Field(
        default="",
        description="Preferred appointment time in 24-hour HH:MM format.",
    )
    preferred_artist: PreferredArtist = Field(
        default="",
        description="Client's preferred artist or No preference.",
    )
    appointment_type: AppointmentType = Field(
        default="",
        description="Online or studio-visit appointment preference.",
    )
    availability: str = Field(
        default="",
        description="Preferred dates or general availability.",
    )
    tattoo_project_type: TattooProjectType = Field(
        default="",
        description="New tattoo, cover-up, continuation, or touch-up.",
    )
    party_size: int = Field(default=1, ge=1, le=20)
    multi_entity_detected: bool = Field(default=False)
    complexity_notes: str = Field(
        default="",
        description=(
            "Notes on complex requests, e.g., multiple people or colors."
        ),
    )
    size_description: str = Field(default="", max_length=100)
    size_status: SizeStatus = "unknown"
    artist_preference_mode: ArtistPreferenceMode = "unknown"
    pricing_requested: bool = False
    client_intent: ClientIntent = "continue_intake"
    conversation_status: ConversationStatus = "active"
    intake_status: IntakeStatus = "collecting_info"
    staff_review_required: bool = False
    review_reasons: list[str] = Field(default_factory=list, max_length=20)
    suggested_artist: SuggestedArtist = Field(
        description="Configured artist display name or Unclear.",
    )
    confidence_level: ConfidenceLevel = Field(
        description="Confidence in routing and extraction quality.",
    )
    ai_reasoning: str = Field(
        description="Brief operational reasoning behind routing and risk output.",
    )
    missing_information: list[MissingInformationItem] = Field(
        default_factory=list,
        description="Checklist items still required before booking follow-up.",
    )
    risk_level: RiskLevel = Field(
        description="Low or high risk triage label for this inquiry.",
    )
    draft_reply: str = Field(
        description="Draft text requiring the delivery controls below.",
    )
    auto_reply_allowed: bool = Field(
        description="Whether backend automation may send the draft directly.",
    )
    auto_reply: bool = Field(
        default=False,
        alias="Auto-reply",
        description=(
            "Whether this is a reply-capable Outlook tattoo inquiry that "
            "the backend may send automatically."
        ),
    )
    telegram_review_required: bool = Field(
        description="Whether the draft must be routed to staff in Telegram.",
    )
    price_suggestion: PriceSuggestion | None = Field(
        default=None,
        description=(
            "Staff-only suggested price range, produced for high-risk "
            "requests. Never send it to the client without staff approval."
        ),
    )

    @field_validator("date")
    @classmethod
    def validate_date_format(cls, value: str) -> str:
        """Require a real calendar date in the public YYYY-MM-DD format."""
        return _validate_preferred_date(value)

    @field_validator("preferred_artist", mode="before")
    @classmethod
    def normalize_preferred_artist(cls, value: object) -> object:
        """Convert legacy artist names to current client-facing names."""
        return _canonical_preferred_artist(value)

    @field_validator("time")
    @classmethod
    def validate_time_format(cls, value: str) -> str:
        """Require a real clock time in the public 24-hour HH:MM format."""
        return _validate_preferred_time(value)

    @model_validator(mode="before")
    @classmethod
    def default_delivery_controls(cls, value: Any) -> Any:
        """Derive safe delivery defaults from the declared risk level."""
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        is_high_risk = payload.get("risk_level") == "high"
        requires_review = (
            is_high_risk or payload.get("staff_review_required") is True
        )
        payload.setdefault("auto_reply_allowed", not requires_review)
        payload.setdefault("telegram_review_required", requires_review)
        return payload

    @model_validator(mode="after")
    def enforce_high_risk_delivery(self) -> Self:
        """Forbid high-risk outputs from entering the auto-reply channel."""
        if self.risk_level == "high" and self.auto_reply_allowed:
            raise ValueError("High-risk drafts cannot allow auto-replies.")
        if self.auto_reply and not self.auto_reply_allowed:
            raise ValueError("Auto-reply requires auto_reply_allowed.")
        if self.risk_level == "high" and not self.telegram_review_required:
            raise ValueError("High-risk drafts require Telegram review.")
        if self.staff_review_required and self.auto_reply_allowed:
            raise ValueError("Staff-review drafts cannot allow auto-replies.")
        if self.staff_review_required and not self.telegram_review_required:
            raise ValueError("Staff-review drafts require Telegram review.")
        return self


def _validate_preferred_date(value: str) -> str:
    """Validate and preserve an optional ISO calendar date string."""
    if not value:
        return value
    try:
        parsed = calendar_date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("date must use YYYY-MM-DD format.") from exc
    if parsed.isoformat() != value:
        raise ValueError("date must use YYYY-MM-DD format.")
    return value


def _validate_preferred_time(value: str) -> str:
    """Validate and preserve an optional 24-hour clock time string."""
    if not value:
        return value
    try:
        parsed = datetime.strptime(value, "%H:%M")
    except ValueError as exc:
        raise ValueError("time must use 24-hour HH:MM format.") from exc
    if parsed.strftime("%H:%M") != value:
        raise ValueError("time must use 24-hour HH:MM format.")
    return value
