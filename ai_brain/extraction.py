"""Text extraction and missing-information detection module."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from datetime import date as calendar_date
from datetime import timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field

from .errors import AnalysisPipelineError
from .email_cleaning import strip_quoted_email_content
from .llm import get_chat_model
from .llm_health import llm_failure_context, record_llm_failure
from .prompts import EXTRACTION_SYSTEM_PROMPT, build_extraction_human_prompt
from .review_policy import (
    is_reference_led_style_request,
    is_status_update_request,
)
from .schemas import (
    AppointmentType,
    ArtistPreferenceMode,
    ClientIntent,
    ConversationStatus,
    MISSING_INFORMATION_OPTIONS,
    STYLE_TAG_OPTIONS,
    Message,
    MissingInformationItem,
    PreferredArtist,
    StyleTag,
    TattooExtractionDraft,
    TattooProjectDetail,
    TattooProjectType,
    VisualColorPreference,
)

LOGGER = logging.getLogger(__name__)

_IMAGE_ONLY_MESSAGE = "Client sent reference image(s) without a text caption."
_STYLE_TAG_SET = set(STYLE_TAG_OPTIONS)
_STYLE_TAGS_THAT_RESOLVE_PREFERENCE = _STYLE_TAG_SET - {
    "unknown",
    "floral",
    "black-and-grey",
}
_SPECIFIC_STYLE_PARENTS: dict[str, str] = {
    "fine-illustrative": "illustrative",
    "micro-realism": "realism",
    "neo-traditional": "traditional",
}
_MISSING_SET = set(MISSING_INFORMATION_OPTIONS)
_ARTIST_NAME_ALIASES: dict[str, str] = {
    "hossam": "Hossam",
    "hoss": "Hossam",
    "nina": "Nina",
    "lana": "Lana",
    "sandra": "Sandra",
    "sliva": "Sliva",
    "silva": "Sliva",
    "mila": "Mila",
}
_ARTIST_ONLY_IDEA_PATTERN = re.compile(
    r"^(?:"
    + "|".join(sorted(_ARTIST_NAME_ALIASES, key=len, reverse=True))
    + r")(?:\s+(?:for|to|as|by|with|do|does|doing|done|my|me|this|that|the|"
    r"a|an|it|one|piece|work|artist|tattoo|tattooist|tattoos|please))*$",
    flags=re.IGNORECASE,
)
_EXISTING_TATTOO_BEFORE_PATTERN = re.compile(
    r"\b(?:i\s+(?:already\s+)?(?:have|had|got)|i've\s+(?:already\s+)?"
    r"(?:got|had)|i\s+already\s+have|my\s+(?:existing|old|current|other)|"
    r"existing|already\s+have)\s+(?:a\s+|an\s+|one\s+|some\s+|two\s+|"
    r"(?:an?\s+)?(?:old|small|little|existing)\s+)?tattoos?\s+"
    r"(?:on|in)\s+(?:my\s+|the\s+)?$"
)
_EXISTING_TATTOO_WORK_PATTERN = re.compile(
    r"\b(?:cover(?:[- ]?up|\s+up|ing)?|rework|re-work|touch[- ]?up|"
    r"extend|extension|add\s+to)\b"
)
_CLARIFICATION_QUESTION_PATTERN = re.compile(
    r"[^.!?\n]*\b(?:(?:are|were)\s+you\s+(?:talking|asking|referring)\s+"
    r"about|(?:what|which)\s+do\s+you\s+mean|do\s+you\s+mean|"
    r"did\s+you\s+mean|you\s+mean)\b[^.!?\n]*\?",
    flags=re.IGNORECASE,
)
_DIRECTIONAL_BACK_BEFORE_PATTERN = re.compile(
    r"\b(?:front|side|top|left|right|forth)\s+(?:to|and|&|or)\s+$"
)
_DIRECTIONAL_BACK_AFTER_PATTERN = re.compile(
    r"^\s+(?:to|and|&|or)\s+(?:front|forth)\b"
)
_VERB_BACK_BEFORE_PATTERN = re.compile(
    r"\b(?:come|comes|coming|came|get|gets|getting|got|go|going|went|"
    r"write|writing|reply|replying|call|calling|text|texting|message|"
    r"bring|be|being|been|step|look|looking|hear|hold|holding|"
    r"heading|way)\s+$"
)
_SUGGESTED_ARTIST_OFFER_PATTERN = re.compile(
    r"\b(?:go\s+(?:ahead\s+)?with|book\s+(?:you\s+)?(?:in\s+)?with|"
    r"work\s+with)\s+(?P<artist>[A-Za-z]+)\b",
    flags=re.IGNORECASE,
)
_AFFIRMATIVE_REPLY_PATTERN = re.compile(
    r"^\s*(?:yes|yeah|yep|yup|sure|ok|okay|alright|sounds\s+good|"
    r"sounds\s+great|perfect|great|let'?s\s+do\s+it|go\s+ahead|"
    r"that\s+works|fine)\b(?!.*\?)",
    flags=re.IGNORECASE,
)
_HESITANT_REPLY_PATTERN = re.compile(
    r"\b(?:but|not|no|think|maybe|another|other|instead|someone|else|"
    r"wait|hold\s+on)\b",
    flags=re.IGNORECASE,
)
_PRICING_PATTERN = re.compile(
    r"\b(?:price|pricing|cost|quote|how much|budget)\b",
    flags=re.IGNORECASE,
)
_WITHDRAWAL_PATTERN = re.compile(
    r"\b(?:i(?:'m|\s+am)?\s+no\s+longer\s+interested|"
    r"i(?:'m|\s+am)?\s+not\s+interested\s+anymore|"
    r"never\s*mind|forget\s+it|do\s+not\s+proceed|don't\s+proceed|"
    r"dont\s+proceed|stop\s+(?:this|the)\s+(?:inquiry|request)|"
    r"close\s+(?:this|my)\s+(?:inquiry|request))\b",
    flags=re.IGNORECASE,
)
_REOPEN_PATTERN = re.compile(
    r"\b(?:i\s+changed\s+my\s+mind|i(?:'d|\s+would)\s+like\s+to\s+"
    r"continue|please\s+reopen|let(?:'s|\s+us)\s+continue)\b",
    flags=re.IGNORECASE,
)
_ARTIST_GUIDANCE_INTENT_PATTERN = re.compile(
    r"\b(?:recommend|suggest|best|better|which\s+artist|who\s+would|"
    r"who\s+are\s+(?:the|your)\s+artists|artists?\s+in\s+(?:the|your)\s+"
    r"(?:shop|studio)|guide\s+me\s+about\s+(?:them|the\s+artists)|"
    r"portfolio|speciali[sz]|tell\s+me\s+about)\b",
    flags=re.IGNORECASE,
)
_AVAILABILITY_QUESTION_PATTERN = re.compile(
    r"\b(?:what|which|when)\b.{0,45}\b(?:available|availability|open|"
    r"slots?|appointments?)\b",
    flags=re.IGNORECASE,
)
_SIZE_GUIDANCE_PATTERN = re.compile(
    r"\b(?:suggest|recommend|advise|help)\b.{0,35}\b(?:size|large|big)\b|"
    r"\bsize\b.{0,50}\b(?:suggest|recommend|advise|help)\b|"
    r"\b(?:what|which)\s+size\b.{0,35}\b(?:recommend|suggest|best)\b",
    flags=re.IGNORECASE,
)
_COMPLAINT_PATTERN = re.compile(
    r"\b(?:i\s+said|already\s+(?:said|told)|how\s+many\s+times|"
    r"why\s+are\s+you\s+asking|not\s+listening|frustrated|upset)\b",
    flags=re.IGNORECASE,
)
_COMPLEX_ROUTING_NOTE = "Complex routing required"
_REFERENCE_LED_STYLE_NOTE = (
    "Style preference: reference-led design; no named style was inferred."
)
_REFERENCE_SIZE_NOTE = (
    "Size guidance: reference-informed planning estimate; client confirmation "
    "is required."
)
_DEFAULT_PLACEMENT_SIZE_RANGE = "5-20 cm"
_PLACEMENT_SIZE_RANGES: dict[str, str] = {
    "scalp": "5-20 cm",
    "head": "5-20 cm",
    "temple": "2-8 cm",
    "face": "2-10 cm",
    "eyebrow": "1-5 cm",
    "eyelid": "1-4 cm",
    "lip": "1-4 cm",
    "tongue": "1-4 cm",
    "ear": "1-5 cm",
    "behind the ear": "2-6 cm",
    "neck": "5-15 cm",
    "throat": "5-15 cm",
    "nape": "5-15 cm",
    "collarbone": "8-20 cm",
    "shoulder": "8-20 cm",
    "shoulder blade": "10-25 cm",
    "armpit": "5-15 cm",
    "chest": "10-25 cm",
    "sternum": "8-20 cm",
    "breast": "5-20 cm",
    "rib cage": "10-30 cm",
    "side": "10-30 cm",
    "stomach": "10-30 cm",
    "abdomen": "10-30 cm",
    "belly": "10-30 cm",
    "upper belly": "10-25 cm",
    "lower belly": "10-25 cm",
    "waist": "8-25 cm",
    "back": "15-35 cm",
    "upper back": "15-35 cm",
    "lower back": "10-30 cm",
    "spine": "10-35 cm",
    "arm": "8-20 cm",
    "upper arm": "8-20 cm",
    "lower arm": "8-20 cm",
    "bicep": "8-20 cm",
    "tricep": "8-20 cm",
    "elbow": "5-12 cm",
    "forearm": "8-20 cm",
    "inner forearm": "8-20 cm",
    "outer forearm": "8-20 cm",
    "wrist": "3-8 cm",
    "inner wrist": "3-8 cm",
    "outer wrist": "3-8 cm",
    "hand": "10-15 cm",
    "palm": "8-10 cm",
    "finger": "1-5 cm",
    "knuckle": "1-4 cm",
    "hip": "8-20 cm",
    "groin": "5-15 cm",
    "buttock": "10-25 cm",
    "thigh": "12-30 cm",
    "knee": "8-18 cm",
    "kneecap": "8-15 cm",
    "leg": "10-30 cm",
    "calf": "10-25 cm",
    "shin": "10-25 cm",
    "ankle": "3-10 cm",
    "foot": "5-15 cm",
    "toe": "1-4 cm",
}
_QUALITATIVE_SIZE_RANGES: dict[str, str] = {
    "hand-sized": "10-15 cm",
    "palm-sized": "8-10 cm",
    "full-chest": "30-40 cm",
    "full-back": "35-60 cm",
    "full-sleeve": "45-65 cm",
    "half-sleeve": "20-35 cm",
    "full-leg": "50-90 cm",
    "half-leg": "25-45 cm",
    "credit-card-sized": "8-9 cm",
    "coin-sized": "2-3 cm",
    "matchbox-sized": "5-6 cm",
    **{
        f"{placement}-sized": size_range
        for placement, size_range in _PLACEMENT_SIZE_RANGES.items()
    },
}
_NAMED_COLOR_PATTERN = re.compile(
    r"\b(?:red|blue|green|yellow|purple|orange|pink|black|white)\b",
    flags=re.IGNORECASE,
)
_MULTIPLE_TATTOOS_PATTERN = re.compile(
    r"\b(?:two|three|four|five|six|seven|eight|nine|ten|\d{1,2})\s+"
    r"(?:different\s+|matching\s+)?tattoos?\b|"
    r"\b(?:multiple|several)\s+tattoos?\b",
    flags=re.IGNORECASE,
)
_COUNTED_BODY_PART_PATTERN = re.compile(
    r"\b(?P<count>two|three|four|five|six|seven|eight|nine|ten|\d{1,2})\s+"
    r"(?P<part>fingers?|toes?|hands?|wrists?|arms?|legs?)\b",
    flags=re.IGNORECASE,
)
_MATCHING_EXISTING_TATTOO_PATTERN = re.compile(
    r"\bmy\s+(?P<relation>girlfriend|boyfriend|partner|wife|husband|"
    r"friend|sister|brother)(?:'s|\s+has\s+(?:a\s+)?)\s*tattoo\b"
    r".{0,120}\b(?:i\s+)?(?:want\s+to\s+)?match\b|"
    r"\b(?:i\s+)?(?:want\s+to\s+)?match\b.{0,120}"
    r"\bmy\s+(?P<reverse_relation>girlfriend|boyfriend|partner|wife|"
    r"husband|friend|sister|brother)(?:'s)?\s+tattoo\b",
    flags=re.IGNORECASE,
)
_GENERIC_TATTOO_IDEA_PATTERN = re.compile(
    r"^(?:(?:i|we|the client)\s+)?"
    r"(?:(?:want|wants|need|needs|would like|request|requests|"
    r"is interested in)\s+)?"
    r"(?:a\s+|an\s+|some\s+)?(?:new\s+)?tattoo"
    r"(?:\s+(?:idea|design|request|inquiry))?$",
    flags=re.IGNORECASE,
)
_GENERAL_HELP_PATTERN = re.compile(
    r"^(?:hi|hello|hey|good morning|good afternoon|good evening|"
    r"(?:can|could|would)\s+you\s+help(?:\s+me)?(?:\s+out)?|"
    r"(?:i\s+)?need\s+help|please\s+help)(?:\s+with\s+(?:a\s+)?"
    r"tattoo)?$",
    flags=re.IGNORECASE,
)
_CONCEPTLESS_INQUIRY_WORDS = {
    "a",
    "about",
    "an",
    "appointment",
    "book",
    "booking",
    "can",
    "consultation",
    "cost",
    "could",
    "do",
    "does",
    "estimate",
    "for",
    "get",
    "getting",
    "give",
    "have",
    "make",
    "making",
    "my",
    "new",
    "some",
    "hello",
    "hey",
    "hi",
    "how",
    "i",
    "inquiry",
    "is",
    "it",
    "know",
    "me",
    "much",
    "need",
    "of",
    "please",
    "price",
    "pricing",
    "quote",
    "rate",
    "request",
    "tattoo",
    "tattoos",
    "tell",
    "the",
    "this",
    "to",
    "want",
    "what",
    "will",
    "wondering",
    "would",
    "you",
}
_STYLE_ONLY_IDEA_VALUES = {
    "black-and-grey",
    "calligraphy",
    "fine-line",
    "floral",
    "geometric",
    "lettering",
    "micro-realism",
    "minimal",
    "script",
    "traditional",
    "watercolor",
}
_STYLE_TEXT_ALIASES: dict[StyleTag, tuple[str, ...]] = {
    "fine-line": ("fine-line", "fine line", "fineline"),
    "watercolor": ("watercolor", "watercolour"),
    "minimal": ("minimal", "minimalist"),
    "floral": ("floral",),
    "botanical": ("botanical",),
    "illustrative": ("illustrative", "illustration"),
    "fine-illustrative": (
        "fine illustrative",
        "transparent floral",
        "translucent floral",
        "x-ray floral",
        "x ray floral",
    ),
    "abstract": ("abstract",),
    "blackwork": ("blackwork", "black work"),
    "realism": ("realism", "realistic"),
    "micro-realism": ("micro-realism", "micro realism"),
    "pixel-art": ("pixel-art", "pixel art", "pixel tattoo"),
    "new-school": ("new-school", "new school"),
    "neo-traditional": ("neo-traditional", "neo traditional"),
    "calligraphy": ("calligraphy", "calligraphic", "lettering"),
    "traditional": ("traditional",),
    "geometric": ("geometric", "geometry"),
}
_PLACEMENT_ALIASES: tuple[tuple[str, str], ...] = (
    ("behind the ear", "behind the ear"),
    ("behind my ear", "behind the ear"),
    ("shoulder blade", "shoulder blade"),
    ("inner forearm", "inner forearm"),
    ("outer forearm", "outer forearm"),
    ("inner wrist", "inner wrist"),
    ("outer wrist", "outer wrist"),
    ("upper arm", "upper arm"),
    ("lower arm", "lower arm"),
    ("upper back", "upper back"),
    ("lower back", "lower back"),
    ("rib cage", "rib cage"),
    ("ribcage", "rib cage"),
    ("ribs", "rib cage"),
    ("collarbone", "collarbone"),
    ("forearm", "forearm"),
    ("sternum", "sternum"),
    ("shoulder", "shoulder"),
    ("abdomen", "abdomen"),
    ("stomach", "stomach"),
    ("upper belly", "upper belly"),
    ("lower belly", "lower belly"),
    ("belly", "belly"),
    ("tummy", "stomach"),
    ("spine", "spine"),
    ("armpit", "armpit"),
    ("bicep", "bicep"),
    ("biceps", "bicep"),
    ("tricep", "tricep"),
    ("triceps", "tricep"),
    ("elbow", "elbow"),
    ("wrist", "wrist"),
    ("knuckles", "knuckle"),
    ("knuckle", "knuckle"),
    ("fingers", "finger"),
    ("finger", "finger"),
    ("palm", "palm"),
    ("ankle", "ankle"),
    ("chest", "chest"),
    ("breast", "breast"),
    ("thigh", "thigh"),
    ("kneecap", "kneecap"),
    ("knee", "knee"),
    ("calf", "calf"),
    ("shin", "shin"),
    ("back", "back"),
    ("neck", "neck"),
    ("throat", "throat"),
    ("nape", "nape"),
    ("scalp", "scalp"),
    ("temple", "temple"),
    ("eyebrow", "eyebrow"),
    ("eyelid", "eyelid"),
    ("face", "face"),
    ("head", "head"),
    ("ears", "ear"),
    ("ear", "ear"),
    ("lips", "lip"),
    ("lip", "lip"),
    ("tongue", "tongue"),
    ("hand", "hand"),
    ("foot", "foot"),
    ("feet", "foot"),
    ("toes", "toe"),
    ("toe", "toe"),
    ("waist", "waist"),
    ("hips", "hip"),
    ("hip", "hip"),
    ("groin", "groin"),
    ("buttocks", "buttock"),
    ("buttock", "buttock"),
    ("side", "side"),
    ("arm", "arm"),
    ("leg", "leg"),
)
_SIZE_FIELD_TERMS = (
    "size",
    "centimeter",
    "centimetre",
    " cm",
    "inch",
    "bigger",
    "smaller",
)
_PLACEMENT_FIELD_TERMS = (
    "placement",
    "body part",
    "where on",
    *tuple(alias for alias, _ in _PLACEMENT_ALIASES),
)
_COLOR_FIELD_TERMS = (
    "color",
    "colour",
    "black ink",
    "black-and-grey",
    "black and grey",
    "black and gray",
    "black & grey",
    "black & gray",
)
_DATE_FIELD_TERMS = (
    "date",
    "day",
    "today",
    "todaye",
    "tomorrow",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)
_TIME_FIELD_TERMS = (
    "time",
    "morning",
    "afternoon",
    "evening",
    " am",
    " pm",
)
_APPOINTMENT_TYPE_FIELD_TERMS = (
    "appointment",
    "in-person",
    "in person",
    "online",
    "studio visit",
    "studio_visit",
    "visit the studio",
    "visit to the studio",
    "visit to your studio",
    "come to the studio",
    "come to your studio",
    "came to the studio",
    "came to your studio",
)
_AVAILABILITY_FIELD_TERMS = (
    "availability",
    "available",
    "date",
    "day",
    "weekday",
    "weekend",
    "morning",
    "afternoon",
    "evening",
    "anytime",
    "flexible",
    *_DATE_FIELD_TERMS,
)
_TATTOO_SUBJECT_WORDS = (
    "anchor",
    "animal",
    "bird",
    "butterfly",
    "diamond",
    "dragon",
    "eagle",
    "flower",
    "flowers",
    "heart",
    "lettering",
    "lotus",
    "mandala",
    "portrait",
    "quote",
    "rose",
    "script",
    "skeleton",
    "skeleton art",
    "skull",
    "snake",
    "star",
    "stars",
    "sword",
    "symbol",
    "tiger",
    "tulip",
    "wolf",
)
_TATTOO_IDEA_MODIFIERS = (
    "black",
    "black-and-grey",
    "black and grey",
    "black and gray",
    "colored",
    "colorful",
    "coloured",
    "colourful",
    "fine-line",
    "fine line",
    "geometric",
    "large",
    "minimal",
    "minimalist",
    "multiple",
    "realistic",
    "small",
    "traditional",
    "watercolor",
    "watercolour",
)
_MONTH_NUMBERS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}
_WEEKDAY_NUMBERS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}
_NEGATION_PREFIX_PATTERN = re.compile(
    r"(?:\bnot|\bno|instead\s+of|rather\s+than|\bfrom)\s+$",
    flags=re.IGNORECASE,
)


class _ExtractionSubset(BaseModel):
    """Subset schema used to parse extraction fields from the LLM."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    client_name: str = Field(
        default="",
        description="Client's full name.",
    )
    tattoo_idea: str = Field(
        description="Core tattoo idea extracted from client text.",
    )
    placement: str = Field(
        description="Requested body placement for the tattoo.",
    )
    size_estimate_cm: str = Field(
        description="Tattoo size estimate in centimeters.",
    )
    color_preference: str = Field(
        description="Color preference or constraints.",
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
        description="Preferred artist or No preference.",
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
    complexity_notes: str = Field(default="")
    projects: list[TattooProjectDetail] = Field(
        default_factory=list,
        max_length=20,
    )
    size_description: str = Field(default="", max_length=100)
    artist_preference_mode: ArtistPreferenceMode = "unknown"
    pricing_requested: bool = False
    client_intent: ClientIntent = "continue_intake"
    conversation_status: ConversationStatus = "active"
    missing_information: list[MissingInformationItem] = Field(
        default_factory=list,
        description="Missing items from the required intake checklist.",
    )


class TattooTextExtractor:
    """Extract structured tattoo fields from inquiry text."""

    def __init__(
        self,
        llm: ChatOpenAI | None = None,
        model_name: str | None = None,
    ) -> None:
        self._llm = llm or get_chat_model(model_name=model_name)
        self._parser = JsonOutputParser(pydantic_object=_ExtractionSubset)

    def extract(
        self,
        current_message: str,
        style_tags: list[str],
        visual_color_preference: VisualColorPreference = "unknown",
        visual_subjects: list[str] | None = None,
        visual_description: str = "",
        new_image_urls: list[str] | None = None,
        existing_db_state: dict[str, Any] | None = None,
        recent_chat_history: list[Message] | None = None,
    ) -> TattooExtractionDraft:
        """Extract details from the latest message and supplied context."""
        normalized_message = current_message.strip()
        safe_image_urls = list(new_image_urls or [])
        safe_visual_subjects = self._normalize_visual_subjects(
            visual_subjects or []
        )
        if not normalized_message and not safe_image_urls:
            raise AnalysisPipelineError(
                "current_message or new_image_urls must be provided."
            )
        if not normalized_message:
            normalized_message = _IMAGE_ONLY_MESSAGE

        safe_db_state = dict(existing_db_state or {})
        safe_chat_history = list(recent_chat_history or [])
        current_style_tags = self._detect_style_tags_from_text(
            normalized_message
        )
        history_text = self._user_history_text(safe_chat_history)
        history_style_tags = self._detect_style_tags_from_text(history_text)
        stored_style_tags = self._state_style_tags(safe_db_state)
        text_style_tags = (
            current_style_tags or history_style_tags or stored_style_tags
        )
        negated_style_tags = self._detect_negated_style_tags(
            normalized_message
        )
        normalized_tags = self._normalize_style_tags(
            [
                tag
                for tag in [*style_tags, *text_style_tags]
                if tag not in negated_style_tags
            ]
        )

        try:
            llm_output = self._invoke_extraction_llm(
                current_message=normalized_message,
                style_tags=normalized_tags,
                visual_color_preference=visual_color_preference,
                visual_subjects=safe_visual_subjects,
                visual_description=visual_description,
                new_image_urls=safe_image_urls,
                existing_db_state=safe_db_state,
                recent_chat_history=safe_chat_history,
            )
            visual_output = self._apply_visual_color_default(
                llm_output=llm_output,
                visual_color_preference=visual_color_preference,
            )
            resolved_output = self._apply_context_defaults(
                llm_output=visual_output,
                current_message=normalized_message,
                recent_chat_history=safe_chat_history,
                existing_db_state=safe_db_state,
            )
            resolved_output = self._apply_reference_led_style_note(
                output=resolved_output,
                current_message=normalized_message,
                recent_chat_history=safe_chat_history,
                new_image_urls=safe_image_urls,
                existing_db_state=safe_db_state,
                style_tags=normalized_tags,
            )
            resolved_output = self._apply_reference_size_guidance(
                output=resolved_output,
                current_message=normalized_message,
                recent_chat_history=safe_chat_history,
                new_image_urls=safe_image_urls,
                existing_db_state=safe_db_state,
                visual_description=visual_description,
            )
            if self._is_missing_tattoo_idea(resolved_output.tattoo_idea):
                visual_idea = self._visual_subject_idea(safe_visual_subjects)
                if visual_idea:
                    resolved_output = resolved_output.model_copy(
                        update={"tattoo_idea": visual_idea}
                    )
            missing_information = self._finalize_missing_information(
                llm_output=resolved_output,
                current_message=normalized_message,
                new_image_urls=safe_image_urls,
                existing_db_state=safe_db_state,
                recent_chat_history=safe_chat_history,
                style_tags=normalized_tags,
            )
            return TattooExtractionDraft(
                client_name=resolved_output.client_name,
                tattoo_idea=resolved_output.tattoo_idea,
                style_tags=normalized_tags,
                placement=resolved_output.placement,
                size_estimate_cm=self._infer_size(
                    size_estimate_cm=resolved_output.size_estimate_cm,
                    size_description=resolved_output.size_description,
                ),
                color_preference=resolved_output.color_preference,
                date=resolved_output.date,
                time=resolved_output.time,
                preferred_artist=resolved_output.preferred_artist,
                appointment_type=resolved_output.appointment_type,
                availability=resolved_output.availability,
                date_to_confirm=self._weekday_date_state(
                    normalized_message,
                    safe_chat_history,
                )[1],
                tattoo_project_type=resolved_output.tattoo_project_type,
                party_size=resolved_output.party_size,
                multi_entity_detected=(
                    resolved_output.multi_entity_detected
                ),
                complexity_notes=resolved_output.complexity_notes,
                projects=self._resolve_projects(
                    llm_projects=resolved_output.projects,
                    current_message=normalized_message,
                    style_tags=normalized_tags,
                    resolved_output=resolved_output,
                    new_image_urls=safe_image_urls,
                    existing_db_state=safe_db_state,
                ),
                size_description=resolved_output.size_description,
                size_status=self._size_status(
                    size_estimate_cm=resolved_output.size_estimate_cm,
                    size_description=resolved_output.size_description,
                    current_message=normalized_message,
                ),
                artist_preference_mode=resolved_output.artist_preference_mode,
                pricing_requested=resolved_output.pricing_requested,
                client_intent=resolved_output.client_intent,
                conversation_status=resolved_output.conversation_status,
                missing_information=missing_information,
            )
        except Exception as exc:  # pragma: no cover - defensive branch
            record_llm_failure(exc)
            LOGGER.warning(
                "Text extraction fallback used: %s [%s]",
                exc,
                llm_failure_context(exc),
            )
            return self._build_fallback_draft(
                current_message=normalized_message,
                style_tags=normalized_tags,
                visual_color_preference=visual_color_preference,
                visual_subjects=safe_visual_subjects,
                visual_description=visual_description,
                new_image_urls=safe_image_urls,
                existing_db_state=safe_db_state,
                recent_chat_history=safe_chat_history,
            )

    def _invoke_extraction_llm(
        self,
        current_message: str,
        style_tags: list[StyleTag],
        visual_color_preference: VisualColorPreference,
        visual_subjects: list[str],
        visual_description: str,
        new_image_urls: list[str],
        existing_db_state: dict[str, Any],
        recent_chat_history: list[Message],
    ) -> _ExtractionSubset:
        """Call the model and parse strict JSON output for extraction fields."""
        format_instructions = self._parser.get_format_instructions()
        human_prompt = build_extraction_human_prompt(
            current_message=current_message,
            style_tags=style_tags,
            visual_color_preference=visual_color_preference,
            visual_subjects=visual_subjects,
            visual_description=visual_description,
            new_image_urls=new_image_urls,
            existing_db_state=existing_db_state,
            recent_chat_history=recent_chat_history,
            required_items=MISSING_INFORMATION_OPTIONS,
            format_instructions=format_instructions,
        )

        try:
            response = self._llm.invoke(
                [
                    SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
                    HumanMessage(content=human_prompt),
                ]
            )
            response_text = self._coerce_content_to_text(response.content)
            payload = self._parser.parse(response_text)
            return _ExtractionSubset.model_validate(payload)
        except Exception as exc:
            raise AnalysisPipelineError(
                "Tattoo detail extraction failed."
            ) from exc

    def _coerce_content_to_text(self, content: Any) -> str:
        """Normalize LangChain response content to plain text."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text_value = item.get("text")
                    if isinstance(text_value, str):
                        text_parts.append(text_value)
            return "".join(text_parts)
        return str(content)

    def _normalize_style_tags(self, style_tags: list[str]) -> list[StyleTag]:
        """Normalize incoming style tags to the approved taxonomy."""
        cleaned: list[str] = []
        for tag in style_tags:
            normalized = tag.strip().lower()
            if normalized in _STYLE_TAG_SET and normalized not in cleaned:
                cleaned.append(normalized)

        if not cleaned:
            return ["unknown"]
        if "unknown" in cleaned and len(cleaned) > 1:
            cleaned = [tag for tag in cleaned if tag != "unknown"]
        for specific, parent in _SPECIFIC_STYLE_PARENTS.items():
            if specific in cleaned and parent in cleaned:
                cleaned.remove(parent)

        return cast(list[StyleTag], cleaned)

    def _state_style_tags(
        self,
        existing_db_state: dict[str, Any],
    ) -> list[str]:
        """Read previously confirmed style tags from canonical state."""
        for record in self._state_records(existing_db_state):
            value = record.get("style_tags")
            if not isinstance(value, list):
                continue
            normalized = self._normalize_style_tags(
                [item for item in value if isinstance(item, str)]
            )
            if normalized != ["unknown"]:
                return list(normalized)
        return []

    def _apply_reference_led_style_note(
        self,
        output: _ExtractionSubset,
        current_message: str,
        recent_chat_history: list[Message],
        new_image_urls: list[str],
        existing_db_state: dict[str, Any],
        style_tags: list[StyleTag],
    ) -> _ExtractionSubset:
        """Record image-led style without inventing a named technique."""
        if any(
            tag in _STYLE_TAGS_THAT_RESOLVE_PREFERENCE
            for tag in style_tags
        ):
            return output
        conversation = self._user_conversation_text(
            current_message=current_message,
            recent_chat_history=recent_chat_history,
        )
        has_reference = bool(new_image_urls) or self._has_reference_images(
            existing_db_state
        )
        if not has_reference or not is_reference_led_style_request(
            conversation
        ):
            return output
        notes = " ".join(output.complexity_notes.split())
        if _REFERENCE_LED_STYLE_NOTE.casefold() not in notes.casefold():
            notes = " ".join(
                part for part in (notes, _REFERENCE_LED_STYLE_NOTE) if part
            )
        return output.model_copy(update={"complexity_notes": notes})

    def _apply_reference_size_guidance(
        self,
        output: _ExtractionSubset,
        current_message: str,
        recent_chat_history: list[Message],
        new_image_urls: list[str],
        existing_db_state: dict[str, Any],
        visual_description: str,
    ) -> _ExtractionSubset:
        """Suggest a conservative size when image and placement support it.

        A reference image has no reliable physical scale by itself. The
        recommendation therefore combines the visual reference with the
        requested body placement and remains subject to client confirmation.
        """
        has_reference = bool(new_image_urls) or self._has_reference_images(
            existing_db_state
        )
        if not has_reference or self._is_blank(output.placement):
            return output
        if self._has_client_supplied_size(
            current_message=current_message,
            recent_chat_history=recent_chat_history,
            existing_db_state=existing_db_state,
        ):
            return output

        description = self._placement_size_description(output.placement)
        estimate = self._infer_size(
            size_estimate_cm="",
            size_description=description,
        )
        if visual_description.strip():
            estimate = self._validated_visual_size_estimate(
                proposed_estimate=output.size_estimate_cm,
                placement_range=estimate,
            ) or estimate
        if not estimate:
            return output

        notes = " ".join(output.complexity_notes.split())
        if _REFERENCE_SIZE_NOTE.casefold() not in notes.casefold():
            notes = " ".join(
                part for part in (notes, _REFERENCE_SIZE_NOTE) if part
            )
        return output.model_copy(
            update={
                "size_estimate_cm": estimate,
                "size_description": description,
                "complexity_notes": notes,
            }
        )

    def _has_client_supplied_size(
        self,
        current_message: str,
        recent_chat_history: list[Message],
        existing_db_state: dict[str, Any],
    ) -> bool:
        """Return whether the client or persisted intake supplied a size."""
        current_size = self._extract_size_from_text(current_message)
        if current_size:
            return True
        current_description = self._extract_size_description(current_message)
        if current_description == "not sure":
            return False
        if current_description:
            return True

        user_texts = [
            message.content
            for message in recent_chat_history
            if message.role == "user"
        ]
        for text in user_texts:
            if self._extract_size_from_text(text):
                return True
            description = self._extract_size_description(text)
            if description and description != "not sure":
                return True

        stored_size = self._get_state_text(
            existing_db_state,
            ("size_estimate_cm", "size_cm", "size"),
        )
        stored_description = self._get_state_text(
            existing_db_state,
            ("size_description", "size_label", "qualitative_size"),
        )
        return bool(
            stored_size
            or (
                stored_description
                and stored_description.casefold() != "not sure"
            )
        )

    def _validated_visual_size_estimate(
        self,
        proposed_estimate: str,
        placement_range: str,
    ) -> str:
        """Accept a vision-informed model range only within placement limits."""
        normalized = self._extract_size_from_text(proposed_estimate)
        proposed_bounds = self._size_bounds(normalized)
        placement_bounds = self._size_bounds(placement_range)
        if not proposed_bounds or not placement_bounds:
            return ""
        if (
            proposed_bounds[0] < placement_bounds[0]
            or proposed_bounds[1] > placement_bounds[1]
        ):
            return ""
        return normalized

    def _size_bounds(self, value: str) -> tuple[Decimal, Decimal] | None:
        """Return minimum and maximum numeric centimetre values."""
        values = [
            Decimal(match)
            for match in re.findall(r"\d+(?:\.\d+)?", value)
        ]
        if not values:
            return None
        return min(values), max(values)

    def _normalize_visual_subjects(self, subjects: list[str]) -> list[str]:
        """Normalize bounded visual subjects before prompt and state use."""
        normalized: list[str] = []
        for subject in subjects:
            value = " ".join(subject.split()).strip(" .,-")[:80]
            if value and value.casefold() not in {
                item.casefold() for item in normalized
            }:
                normalized.append(value)
        return normalized[:10]

    def _visual_subject_idea(self, subjects: list[str]) -> str:
        """Create a concise tattoo concept from trusted vision subjects."""
        if not subjects:
            return ""
        return " and ".join(subjects[:3]).capitalize()

    def _detect_style_tags_from_text(self, text: str) -> list[str]:
        """Detect approved style names explicitly stated in conversation text."""
        normalized = text.casefold()
        detected: list[str] = []
        for style_tag, aliases in _STYLE_TEXT_ALIASES.items():
            if any(
                self._has_positive_phrase(normalized, alias)
                for alias in aliases
            ):
                detected.append(style_tag)
        return detected

    def _detect_negated_style_tags(self, text: str) -> set[StyleTag]:
        """Return styles explicitly rejected by the latest client message."""
        normalized = text.casefold()
        rejected: set[StyleTag] = set()
        for style_tag, aliases in _STYLE_TEXT_ALIASES.items():
            if any(
                self._has_negated_phrase(normalized, alias)
                for alias in aliases
            ):
                rejected.add(style_tag)
        return rejected

    def _apply_visual_color_default(
        self,
        llm_output: _ExtractionSubset,
        visual_color_preference: VisualColorPreference,
    ) -> _ExtractionSubset:
        """Use image color only when conversation extraction has no color."""
        color_preference = llm_output.color_preference
        if self._is_blank(color_preference):
            if visual_color_preference == "black-and-grey":
                color_preference = "black-and-grey"
            elif visual_color_preference == "color":
                color_preference = "color"
        return _ExtractionSubset(
            client_name=llm_output.client_name,
            tattoo_idea=llm_output.tattoo_idea,
            placement=llm_output.placement,
            size_estimate_cm=llm_output.size_estimate_cm,
            color_preference=color_preference,
            date=llm_output.date,
            time=llm_output.time,
            preferred_artist=llm_output.preferred_artist,
            appointment_type=llm_output.appointment_type,
            availability=llm_output.availability,
            tattoo_project_type=llm_output.tattoo_project_type,
            party_size=llm_output.party_size,
            multi_entity_detected=llm_output.multi_entity_detected,
            complexity_notes=llm_output.complexity_notes,
            projects=llm_output.projects,
            size_description=llm_output.size_description,
            artist_preference_mode=llm_output.artist_preference_mode,
            pricing_requested=llm_output.pricing_requested,
            client_intent=llm_output.client_intent,
            conversation_status=llm_output.conversation_status,
            missing_information=llm_output.missing_information,
        )

    def _finalize_missing_information(
        self,
        llm_output: _ExtractionSubset,
        current_message: str,
        new_image_urls: list[str],
        existing_db_state: dict[str, Any],
        recent_chat_history: list[Message],
        style_tags: list[StyleTag],
    ) -> list[MissingInformationItem]:
        """Reconcile client-actionable gaps across every context source."""
        missing: set[str] = {
            item for item in llm_output.missing_information if item in _MISSING_SET
        }
        inferred_size = self._infer_size(
            size_estimate_cm=llm_output.size_estimate_cm,
            size_description=llm_output.size_description,
        )

        conversation_text = self._user_conversation_text(
            current_message=current_message,
            recent_chat_history=recent_chat_history,
        )
        reference_led_style = (
            _REFERENCE_LED_STYLE_NOTE.casefold()
            in llm_output.complexity_notes.casefold()
        )
        checks: dict[MissingInformationItem, bool] = {
            "client full name": self._is_missing_client_name(
                llm_output.client_name
            ),
            "tattoo idea": self._is_missing_tattoo_idea(
                llm_output.tattoo_idea
            ),
            "size in cm": self._is_blank(inferred_size)
            and (
                self._is_blank(llm_output.size_description)
                or llm_output.size_description == "not sure"
            ),
            "placement": self._is_blank(llm_output.placement),
            "tattoo style": not any(
                tag in _STYLE_TAGS_THAT_RESOLVE_PREFERENCE
                for tag in style_tags
            )
            and not reference_led_style,
            "color preference": self._is_blank(llm_output.color_preference),
            "reference images": not (
                new_image_urls
                or self._has_reference_images(existing_db_state)
                or self._mentions_reference_image(conversation_text)
                or self._declines_reference_image(conversation_text)
            ),
            "preferred artist": (
                self._is_blank(llm_output.preferred_artist)
                and llm_output.artist_preference_mode != "no_preference"
            ),
            "appointment type": self._is_blank(llm_output.appointment_type),
            "preferred dates or availability": self._is_blank(
                llm_output.availability
            ),
            "tattoo project type": self._is_blank(
                llm_output.tattoo_project_type
            ),
        }
        for item, is_missing in checks.items():
            if is_missing:
                missing.add(item)
            else:
                missing.discard(item)

        ordered = [
            item for item in MISSING_INFORMATION_OPTIONS if item in missing
        ]
        return cast(list[MissingInformationItem], ordered)

    def _build_fallback_draft(
        self,
        current_message: str,
        style_tags: list[StyleTag],
        visual_color_preference: VisualColorPreference,
        visual_subjects: list[str],
        visual_description: str,
        new_image_urls: list[str],
        existing_db_state: dict[str, Any],
        recent_chat_history: list[Message],
    ) -> TattooExtractionDraft:
        """Return a safe draft when the extraction call fails."""
        fallback_idea = self._extract_tattoo_idea_from_text(current_message)
        if not fallback_idea:
            fallback_idea = self._visual_subject_idea(visual_subjects)
        fallback = _ExtractionSubset(
            client_name=self._extract_client_name_from_text(current_message),
            tattoo_idea=fallback_idea,
            placement="",
            size_estimate_cm=self._extract_size_from_text(current_message),
            size_description=self._extract_size_description(current_message),
            color_preference="",
            date=self._extract_date_from_text(current_message),
            time=self._extract_time_from_text(current_message),
            preferred_artist=self._extract_preferred_artist_from_text(
                current_message
            ),
            artist_preference_mode=self._extract_artist_preference_mode(
                current_message
            ),
            appointment_type=self._extract_appointment_type_from_text(
                current_message
            ),
            availability=self._extract_availability_from_text(current_message),
            tattoo_project_type=self._extract_project_type_from_text(
                current_message
            ),
            party_size=self._extract_party_size(current_message),
            pricing_requested=bool(_PRICING_PATTERN.search(current_message)),
            missing_information=[],
        )
        visual_fallback = self._apply_visual_color_default(
            llm_output=fallback,
            visual_color_preference=visual_color_preference,
        )
        resolved_fallback = self._apply_context_defaults(
            llm_output=visual_fallback,
            current_message=current_message,
            recent_chat_history=recent_chat_history,
            existing_db_state=existing_db_state,
        )
        resolved_fallback = self._apply_reference_led_style_note(
            output=resolved_fallback,
            current_message=current_message,
            recent_chat_history=recent_chat_history,
            new_image_urls=new_image_urls,
            existing_db_state=existing_db_state,
            style_tags=style_tags,
        )
        resolved_fallback = self._apply_reference_size_guidance(
            output=resolved_fallback,
            current_message=current_message,
            recent_chat_history=recent_chat_history,
            new_image_urls=new_image_urls,
            existing_db_state=existing_db_state,
            visual_description=visual_description,
        )
        missing = self._finalize_missing_information(
            llm_output=resolved_fallback,
            current_message=current_message,
            new_image_urls=new_image_urls,
            existing_db_state=existing_db_state,
            recent_chat_history=recent_chat_history,
            style_tags=style_tags,
        )
        return TattooExtractionDraft(
            client_name=resolved_fallback.client_name,
            tattoo_idea=resolved_fallback.tattoo_idea,
            style_tags=style_tags,
            placement=resolved_fallback.placement,
            size_estimate_cm=self._infer_size(
                size_estimate_cm=resolved_fallback.size_estimate_cm,
                size_description=resolved_fallback.size_description,
            ),
            color_preference=resolved_fallback.color_preference,
            date=resolved_fallback.date,
            time=resolved_fallback.time,
            preferred_artist=resolved_fallback.preferred_artist,
            appointment_type=resolved_fallback.appointment_type,
            availability=resolved_fallback.availability,
            date_to_confirm=self._weekday_date_state(
                current_message,
                recent_chat_history,
            )[1],
            tattoo_project_type=resolved_fallback.tattoo_project_type,
            party_size=resolved_fallback.party_size,
            multi_entity_detected=(
                resolved_fallback.multi_entity_detected
            ),
            complexity_notes=resolved_fallback.complexity_notes,
            projects=self._resolve_projects(
                llm_projects=resolved_fallback.projects,
                current_message=current_message,
                style_tags=style_tags,
                resolved_output=resolved_fallback,
                new_image_urls=new_image_urls,
                existing_db_state=existing_db_state,
            ),
            size_description=resolved_fallback.size_description,
            size_status=self._size_status(
                size_estimate_cm=resolved_fallback.size_estimate_cm,
                size_description=resolved_fallback.size_description,
                current_message=current_message,
            ),
            artist_preference_mode=(
                resolved_fallback.artist_preference_mode
            ),
            pricing_requested=resolved_fallback.pricing_requested,
            client_intent=resolved_fallback.client_intent,
            conversation_status=resolved_fallback.conversation_status,
            missing_information=missing,
        )

    def _apply_context_defaults(
        self,
        llm_output: _ExtractionSubset,
        current_message: str,
        recent_chat_history: list[Message],
        existing_db_state: dict[str, Any],
    ) -> _ExtractionSubset:
        """Resolve fields using current, model, history, then database order."""
        party_size = self._resolve_party_size(
            llm_value=llm_output.party_size,
            current_message=current_message,
            recent_chat_history=recent_chat_history,
            existing_db_state=existing_db_state,
        )
        multi_entity_detected = self._resolve_multi_entity_detected(
            llm_value=llm_output.multi_entity_detected,
            llm_complexity_notes=llm_output.complexity_notes,
            party_size=party_size,
            llm_projects=llm_output.projects,
            current_message=current_message,
            recent_chat_history=recent_chat_history,
            existing_db_state=existing_db_state,
        )
        complexity_notes = self._resolve_complexity_notes(
            llm_value=llm_output.complexity_notes,
            multi_entity_detected=multi_entity_detected,
            party_size=party_size,
            project_count=len(llm_output.projects),
            current_message=current_message,
            recent_chat_history=recent_chat_history,
            existing_db_state=existing_db_state,
        )
        placement = self._resolve_context_field(
            llm_value=llm_output.placement,
            current_message=current_message,
            recent_chat_history=recent_chat_history,
            existing_db_state=existing_db_state,
            state_keys=("placement",),
            value_extractor=self._extract_placement_from_text,
            field_terms=_PLACEMENT_FIELD_TERMS,
        )
        size_description = self._resolve_size_description(
            llm_value=llm_output.size_description,
            current_message=current_message,
            recent_chat_history=recent_chat_history,
            existing_db_state=existing_db_state,
        )
        size_estimate_cm = self._resolve_size_estimate(
            llm_value=llm_output.size_estimate_cm,
            size_description=size_description,
            current_message=current_message,
            recent_chat_history=recent_chat_history,
            existing_db_state=existing_db_state,
        )
        preferred_artist = self._resolve_preferred_artist(
            current_message=current_message,
            recent_chat_history=recent_chat_history,
            existing_db_state=existing_db_state,
            llm_value=llm_output.preferred_artist,
        )
        tattoo_project_type = self._resolve_project_type(
            current_message=current_message,
            recent_chat_history=recent_chat_history,
            existing_db_state=existing_db_state,
        )
        resolved = _ExtractionSubset(
            client_name=self._resolve_client_name(
                llm_value=llm_output.client_name,
                current_message=current_message,
                recent_chat_history=recent_chat_history,
                existing_db_state=existing_db_state,
            ),
            tattoo_idea=self._resolve_tattoo_idea(
                llm_value=llm_output.tattoo_idea,
                current_message=current_message,
                recent_chat_history=recent_chat_history,
                existing_db_state=existing_db_state,
            ),
            placement=placement,
            size_estimate_cm=size_estimate_cm,
            color_preference=self._normalize_color_value(
                self._resolve_context_field(
                    llm_value=self._normalize_color_value(
                        llm_output.color_preference
                    ),
                    current_message=current_message,
                    recent_chat_history=recent_chat_history,
                    existing_db_state=existing_db_state,
                    state_keys=(
                        "color_preference",
                        "colour_preference",
                        "color",
                    ),
                    value_extractor=self._extract_color_from_text,
                    field_terms=_COLOR_FIELD_TERMS,
                )
            ),
            date=self._normalize_date_value(
                self._resolve_context_field(
                    llm_value=self._normalize_date_value(llm_output.date),
                    current_message=current_message,
                    recent_chat_history=recent_chat_history,
                    existing_db_state=existing_db_state,
                    state_keys=(
                        "date",
                        "preferred_date",
                        "appointment_date",
                        "requested_date",
                        "scheduled_date",
                    ),
                    value_extractor=self._extract_date_from_text,
                    field_terms=_DATE_FIELD_TERMS,
                )
            ),
            time=self._normalize_time_value(
                self._resolve_context_field(
                    llm_value=self._normalize_time_value(llm_output.time),
                    current_message=current_message,
                    recent_chat_history=recent_chat_history,
                    existing_db_state=existing_db_state,
                    state_keys=(
                        "time",
                        "preferred_time",
                        "appointment_time",
                        "requested_time",
                        "scheduled_time",
                    ),
                    value_extractor=self._extract_time_from_text,
                    field_terms=_TIME_FIELD_TERMS,
                )
            ),
            preferred_artist=preferred_artist,
            appointment_type=self._normalize_appointment_type(
                self._resolve_context_field(
                    llm_value=self._normalize_appointment_type(
                        llm_output.appointment_type
                    ),
                    current_message=current_message,
                    recent_chat_history=recent_chat_history,
                    existing_db_state=existing_db_state,
                    state_keys=(
                        "appointment_type",
                        "appointment_mode",
                    ),
                    value_extractor=self._extract_appointment_type_from_text,
                    field_terms=_APPOINTMENT_TYPE_FIELD_TERMS,
                )
            ),
            availability=self._resolve_context_field(
                llm_value=llm_output.availability,
                current_message=current_message,
                recent_chat_history=recent_chat_history,
                existing_db_state=existing_db_state,
                state_keys=(
                    "availability",
                    "preferred_availability",
                    "preferred_dates",
                    "date",
                    "preferred_date",
                    "appointment_date",
                    "requested_date",
                    "scheduled_date",
                ),
                value_extractor=self._extract_availability_from_text,
                field_terms=_AVAILABILITY_FIELD_TERMS,
            ),
            tattoo_project_type=tattoo_project_type,
            party_size=party_size,
            multi_entity_detected=multi_entity_detected,
            complexity_notes=complexity_notes,
            projects=llm_output.projects,
            size_description=size_description,
            artist_preference_mode=self._resolve_artist_preference_mode(
                preferred_artist=preferred_artist,
                current_message=current_message,
                recent_chat_history=recent_chat_history,
                existing_db_state=existing_db_state,
            ),
            pricing_requested=self._resolve_pricing_requested(
                llm_value=llm_output.pricing_requested,
                current_message=current_message,
                recent_chat_history=recent_chat_history,
                existing_db_state=existing_db_state,
            ),
            client_intent=self._resolve_client_intent(current_message),
            conversation_status=self._resolve_conversation_status(
                current_message=current_message,
                existing_db_state=existing_db_state,
            ),
            missing_information=llm_output.missing_information,
        )
        return self._apply_weekday_date_confirmation(
            resolved,
            current_message,
            recent_chat_history,
        )

    def _resolve_client_intent(self, current_message: str) -> ClientIntent:
        """Classify the latest client turn before choosing an intake action."""
        if _WITHDRAWAL_PATTERN.search(current_message):
            return "withdrawal"
        if is_status_update_request(current_message):
            return "status_update"
        if _COMPLAINT_PATTERN.search(current_message):
            return "complaint"
        if _SIZE_GUIDANCE_PATTERN.search(current_message):
            return "size_guidance"
        if _ARTIST_GUIDANCE_INTENT_PATTERN.search(current_message):
            return "artist_guidance"
        if _AVAILABILITY_QUESTION_PATTERN.search(current_message):
            return "availability_question"
        if _PRICING_PATTERN.search(current_message):
            return "pricing_question"
        return "continue_intake"

    def _resolve_conversation_status(
        self,
        current_message: str,
        existing_db_state: dict[str, Any],
    ) -> ConversationStatus:
        """Close withdrawn inquiries until the client explicitly reopens them."""
        if _REOPEN_PATTERN.search(current_message):
            return "active"
        if _WITHDRAWAL_PATTERN.search(current_message):
            return "closed"
        stored_status = self._get_state_text(
            existing_db_state,
            ("conversation_status", "status"),
        ).casefold()
        if stored_status in {"closed", "cancelled", "canceled", "withdrawn"}:
            return "closed"
        return "active"

    def _resolve_context_field(
        self,
        llm_value: str,
        current_message: str,
        recent_chat_history: list[Message],
        existing_db_state: dict[str, Any],
        state_keys: tuple[str, ...],
        value_extractor: Callable[[str], str],
        field_terms: tuple[str, ...],
    ) -> str:
        """Resolve one field while enforcing latest-message precedence."""
        current_value = value_extractor(current_message)
        if current_value:
            return current_value
        if self._mentions_field(current_message, field_terms):
            return "" if self._is_blank(llm_value) else llm_value

        history_value = self._latest_history_value(
            recent_chat_history=recent_chat_history,
            value_extractor=value_extractor,
        )
        if history_value:
            return history_value
        stored_value = self._get_state_text(existing_db_state, state_keys)
        if stored_value:
            return stored_value
        return "" if self._is_blank(llm_value) else llm_value

    def _resolve_preferred_artist(
        self,
        current_message: str,
        recent_chat_history: list[Message],
        existing_db_state: dict[str, Any],
        llm_value: str = "",
    ) -> PreferredArtist:
        """Use only an explicit client selection or a persisted preference."""
        current_value = self._extract_preferred_artist_from_text(
            current_message
        )
        if current_value:
            return cast(PreferredArtist, current_value)
        llm_artist = self._client_named_llm_artist(llm_value, current_message)
        if llm_artist:
            return cast(PreferredArtist, llm_artist)
        turns = self._client_turns_with_prompts(
            current_message,
            recent_chat_history,
        )
        if turns:
            confirmed = self._confirmed_offered_artist(*turns[-1])
            if confirmed:
                return cast(PreferredArtist, confirmed)
        for prompt, message in reversed(turns[:-1]):
            history_value = self._extract_preferred_artist_from_text(
                message
            ) or self._confirmed_offered_artist(prompt, message)
            if history_value:
                return cast(PreferredArtist, history_value)
        stored_value = self._get_state_text(
            existing_db_state,
            ("preferred_artist", "requested_artist"),
        )
        normalized = self._normalize_preferred_artist(stored_value)
        return cast(PreferredArtist, normalized)

    def _client_turns_with_prompts(
        self,
        current_message: str,
        recent_chat_history: list[Message],
    ) -> list[tuple[str, str]]:
        """Pair each client message with the assistant message before it."""
        turns: list[tuple[str, str]] = []
        last_assistant = ""
        for message in recent_chat_history:
            if message.role == "assistant":
                last_assistant = message.content
            else:
                turns.append((last_assistant, message.content))
        if not turns or turns[-1][1] != current_message:
            turns.append((last_assistant, current_message))
        return turns

    def _confirmed_offered_artist(self, prompt: str, reply: str) -> str:
        """Accept a short "yes" to an artist the studio just offered."""
        if (
            len(reply.strip()) > 60
            or not _AFFIRMATIVE_REPLY_PATTERN.match(reply)
            or _HESITANT_REPLY_PATTERN.search(reply)
        ):
            return ""
        questions = " ".join(re.findall(r"[^.!?\n]*\?", prompt))
        offered = [
            _ARTIST_NAME_ALIASES[match.group("artist").casefold()]
            for match in _SUGGESTED_ARTIST_OFFER_PATTERN.finditer(questions)
            if match.group("artist").casefold() in _ARTIST_NAME_ALIASES
        ]
        if len(set(offered)) != 1:
            return ""
        return offered[0]

    def _weekday_date_state(
        self,
        current_message: str,
        recent_chat_history: list[Message],
    ) -> tuple[str, str]:
        """Return (confirmed_date, date_to_confirm) for weekday-only dates.

        A bare weekday such as "Thursday" or "next Thursday" is ambiguous, so
        it stays unconfirmed until the client accepts the exact date the
        studio offered back or states an explicit calendar date.
        """
        confirmed = ""
        pending = ""
        for prompt, reply in self._client_turns_with_prompts(
            current_message,
            recent_chat_history,
        ):
            offered = self._confirmed_date_offer(prompt, reply) if pending else ""
            if offered:
                confirmed, pending = offered, ""
                continue
            if not self._extract_date_from_text(reply):
                continue
            if self._extract_date_from_text(reply, allow_weekdays=False):
                confirmed, pending = "", ""
            else:
                confirmed, pending = "", self._extract_date_from_text(reply)
        return confirmed, pending

    def _confirmed_date_offer(self, prompt: str, reply: str) -> str:
        """Accept "yes" (or the same weekday) to a date the studio offered."""
        if len(reply.strip()) > 80 or _HESITANT_REPLY_PATTERN.search(reply):
            return ""
        questions = " ".join(re.findall(r"[^.!?\n]*\?", prompt))
        offered = self._extract_date_from_text(questions, allow_weekdays=False)
        if not offered:
            return ""
        if _AFFIRMATIVE_REPLY_PATTERN.match(reply):
            return offered
        reply_weekday = self._extract_date_from_text(reply)
        if reply_weekday and (
            calendar_date.fromisoformat(reply_weekday).weekday()
            == calendar_date.fromisoformat(offered).weekday()
        ):
            return offered
        return ""

    def _apply_weekday_date_confirmation(
        self,
        output: _ExtractionSubset,
        current_message: str,
        recent_chat_history: list[Message],
    ) -> _ExtractionSubset:
        """Hold back weekday-only dates and apply confirmed ones."""
        confirmed, pending = self._weekday_date_state(
            current_message,
            recent_chat_history,
        )
        if confirmed:
            return output.model_copy(
                update={
                    "date": confirmed,
                    "availability": (
                        f"{confirmed} at {output.time}"
                        if output.time
                        else confirmed
                    ),
                }
            )
        if not pending:
            return output
        availability = output.availability
        if self._extract_date_from_text(availability):
            availability = ""
        return output.model_copy(
            update={"date": "", "availability": availability}
        )

    def _client_named_llm_artist(
        self,
        llm_value: str,
        current_message: str,
    ) -> str:
        """Accept a model-selected artist only when the client named them.

        The model may otherwise echo the studio's own suggestion, so the
        artist must appear in the latest client statement, not a question.
        """
        artist = self._normalize_preferred_artist(llm_value)
        if not artist or artist == "No preference" or "?" in current_message:
            return ""
        normalized = " ".join(current_message.casefold().split())
        names = [
            alias
            for alias, display_name in _ARTIST_NAME_ALIASES.items()
            if display_name == artist
        ]
        if any(re.search(rf"\b{re.escape(name)}\b", normalized) for name in names):
            return artist
        return ""

    def _resolve_project_type(
        self,
        current_message: str,
        recent_chat_history: list[Message],
        existing_db_state: dict[str, Any],
    ) -> TattooProjectType:
        """Use only an explicitly supplied or previously persisted type."""
        current_value = self._extract_project_type_from_text(current_message)
        if current_value:
            return cast(TattooProjectType, current_value)
        for message in reversed(recent_chat_history):
            if message.role != "user":
                continue
            history_value = self._extract_project_type_from_text(
                message.content
            )
            if history_value:
                return cast(TattooProjectType, history_value)
        stored_value = self._get_state_text(
            existing_db_state,
            (
                "tattoo_project_type",
                "tattoo_type",
                "project_type",
                "work_type",
            ),
        )
        normalized = self._normalize_project_type(stored_value)
        return cast(TattooProjectType, normalized)

    def _resolve_party_size(
        self,
        llm_value: int,
        current_message: str,
        recent_chat_history: list[Message],
        existing_db_state: dict[str, Any],
    ) -> int:
        """Preserve the largest explicitly stated party size across the thread."""
        current_value = self._extract_party_size(current_message)
        if current_value > 1:
            return current_value
        for message in reversed(recent_chat_history):
            if message.role != "user":
                continue
            history_value = self._extract_party_size(message.content)
            if history_value > 1:
                return history_value
        for record in self._state_records(existing_db_state):
            for key in ("party_size", "client_count", "people_count"):
                value = record.get(key)
                if isinstance(value, int) and 1 <= value <= 20:
                    return value
                if isinstance(value, str) and value.strip().isdigit():
                    parsed = int(value.strip())
                    if 1 <= parsed <= 20:
                        return parsed
        return min(max(llm_value, 1), 20)

    def _resolve_multi_entity_detected(
        self,
        llm_value: bool,
        llm_complexity_notes: str,
        party_size: int,
        llm_projects: list[TattooProjectDetail],
        current_message: str,
        recent_chat_history: list[Message],
        existing_db_state: dict[str, Any],
    ) -> bool:
        """Detect and remember requests containing multiple routing entities."""
        if party_size > 1 or len(llm_projects) > 1:
            return True
        if llm_value and self._note_indicates_multiple_entities(
            llm_complexity_notes
        ):
            return True

        conversation = self._user_conversation_text(
            current_message=current_message,
            recent_chat_history=recent_chat_history,
        )
        named_colors = {
            match.group(0).casefold()
            for match in _NAMED_COLOR_PATTERN.finditer(conversation)
        }
        if len(named_colors) > 1:
            return True
        if _MULTIPLE_TATTOOS_PATTERN.search(conversation):
            return True
        if len(self._extract_same_client_tattoo_ideas(conversation)) > 1:
            return True
        if _MATCHING_EXISTING_TATTOO_PATTERN.search(conversation):
            return True

        for record in self._state_records(existing_db_state):
            stored_projects = record.get("projects")
            if isinstance(stored_projects, list) and len(stored_projects) > 1:
                return True
            stored_note = record.get("complexity_notes")
            if (
                isinstance(stored_note, str)
                and self._note_indicates_multiple_entities(stored_note)
            ):
                return True
        return False

    def _note_indicates_multiple_entities(self, note: str) -> bool:
        """Reject generic review notes as evidence of multiple entities."""
        return bool(
            re.search(
                r"\b(?:multiple|several|two|three|four|five|six|seven|"
                r"eight|nine|ten|\d{1,2})\s+"
                r"(?:people|clients?|tattoos?|projects?|colou?rs?)\b|"
                r"\bmatching\b.{0,60}\bexisting tattoo\b",
                note,
                flags=re.IGNORECASE,
            )
        )

    def _resolve_complexity_notes(
        self,
        llm_value: str,
        multi_entity_detected: bool,
        party_size: int,
        project_count: int,
        current_message: str,
        recent_chat_history: list[Message],
        existing_db_state: dict[str, Any],
    ) -> str:
        """Build concise internal notes for multi-entity staff routing."""
        stored_note = self._get_state_text(
            existing_db_state,
            ("complexity_notes",),
        )
        base_note = " ".join((llm_value or stored_note).split())
        if not multi_entity_detected:
            return base_note

        conversation = self._user_conversation_text(
            current_message=current_message,
            recent_chat_history=recent_chat_history,
        )
        details: list[str] = []
        enumerated_count = len(
            self._extract_same_client_tattoo_ideas(conversation)
        )
        if party_size > 1:
            details.append(f"{party_size} people")
        if project_count > 1:
            details.append(f"{project_count} tattoo projects")
        elif enumerated_count > 1:
            details.append(f"{enumerated_count} tattoo projects for one client")
        elif _MULTIPLE_TATTOOS_PATTERN.search(conversation):
            details.append("multiple tattoos")

        matching_tattoo = _MATCHING_EXISTING_TATTOO_PATTERN.search(
            conversation
        )
        if matching_tattoo:
            relation = (
                matching_tattoo.group("relation")
                or matching_tattoo.group("reverse_relation")
                or "another person's"
            )
            details.append(
                f"matching the client's {relation}'s existing tattoo"
            )

        named_colors = sorted(
            {
                match.group(0).casefold()
                for match in _NAMED_COLOR_PATTERN.finditer(conversation)
            }
        )
        if len(named_colors) > 1:
            details.append(f"multiple colors: {', '.join(named_colors)}")

        note_parts: list[str] = []
        if base_note:
            cleaned_note = re.sub(
                rf"^{re.escape(_COMPLEX_ROUTING_NOTE)}\s*[:;.-]?\s*",
                "",
                base_note,
                flags=re.IGNORECASE,
            ).rstrip(". ")
            if cleaned_note:
                note_parts.append(cleaned_note)
        for detail in details:
            if detail.casefold() not in base_note.casefold():
                note_parts.append(detail)
        if not note_parts:
            return _COMPLEX_ROUTING_NOTE
        return f"{_COMPLEX_ROUTING_NOTE}: {'; '.join(note_parts)}"

    def _resolve_size_estimate(
        self,
        llm_value: str,
        size_description: str,
        current_message: str,
        recent_chat_history: list[Message],
        existing_db_state: dict[str, Any],
    ) -> str:
        """Resolve exact sizing while honoring a newer qualitative answer."""
        resolved = self._resolve_context_field(
            llm_value=llm_value,
            current_message=current_message,
            recent_chat_history=recent_chat_history,
            existing_db_state=existing_db_state,
            state_keys=("size_estimate_cm", "size_cm", "size"),
            value_extractor=self._extract_size_from_text,
            field_terms=_SIZE_FIELD_TERMS,
        )
        current_description = self._extract_size_description(current_message)
        if current_description:
            return self._extract_size_from_text(current_message)
        if size_description == "not sure" and self._is_uncertain_answer(
            current_message
        ):
            return ""
        return resolved

    def _resolve_size_description(
        self,
        llm_value: str,
        current_message: str,
        recent_chat_history: list[Message],
        existing_db_state: dict[str, Any],
    ) -> str:
        """Resolve qualitative or explicitly uncertain size answers."""
        current_value = self._extract_size_description(current_message)
        if current_value:
            return current_value
        if self._extract_size_from_text(current_message):
            return ""
        if self._is_uncertain_answer(current_message) and any(
            message.role == "assistant"
            and any(
                marker in message.content.casefold()
                for marker in ("size", "centimet", "how large", "how big")
            )
            for message in recent_chat_history[-2:]
        ):
            return "not sure"
        for message in reversed(recent_chat_history):
            if message.role != "user":
                continue
            if self._extract_size_from_text(message.content):
                return ""
            history_value = self._extract_size_description(message.content)
            if history_value:
                return history_value
        stored_value = self._get_state_text(
            existing_db_state,
            ("size_description", "size_label", "qualitative_size"),
        )
        if stored_value:
            return stored_value
        return "" if self._is_blank(llm_value) else llm_value

    def _resolve_artist_preference_mode(
        self,
        preferred_artist: str,
        current_message: str,
        recent_chat_history: list[Message],
        existing_db_state: dict[str, Any],
    ) -> ArtistPreferenceMode:
        """Distinguish a named artist from a request for AI recommendation."""
        current_value = self._extract_artist_preference_mode(current_message)
        if current_value != "unknown":
            return current_value
        if preferred_artist == "No preference":
            return "no_preference"
        if preferred_artist:
            return "specific"
        for message in reversed(recent_chat_history):
            if message.role != "user":
                continue
            history_value = self._extract_artist_preference_mode(
                message.content
            )
            if history_value != "unknown":
                return history_value
        stored = self._get_state_text(
            existing_db_state,
            ("artist_preference_mode",),
        ).casefold()
        if stored in {"specific", "recommend", "no_preference"}:
            return cast(ArtistPreferenceMode, stored)
        return "unknown"

    def _resolve_pricing_requested(
        self,
        llm_value: bool,
        current_message: str,
        recent_chat_history: list[Message],
        existing_db_state: dict[str, Any],
    ) -> bool:
        """Remember a pricing question throughout the complete conversation."""
        if _PRICING_PATTERN.search(current_message):
            return True
        if any(
            message.role == "user"
            and _PRICING_PATTERN.search(message.content)
            for message in recent_chat_history
        ):
            return True
        for record in self._state_records(existing_db_state):
            if record.get("pricing_requested") is True:
                return True
        return llm_value

    def _resolve_client_name(
        self,
        llm_value: str,
        current_message: str,
        recent_chat_history: list[Message],
        existing_db_state: dict[str, Any],
    ) -> str:
        """Resolve explicit names and short answers to the full-name question."""
        current_name = self._extract_client_name_from_text(current_message)
        if current_name:
            return current_name
        if self._last_assistant_asked_for_name(recent_chat_history):
            short_answer = self._name_only_response(current_message)
            if short_answer:
                return short_answer
        if not self._is_missing_client_name(llm_value):
            return llm_value

        for index in range(len(recent_chat_history) - 1, 0, -1):
            message = recent_chat_history[index]
            previous = recent_chat_history[index - 1]
            if message.role != "user" or previous.role != "assistant":
                continue
            if "full name" not in previous.content.casefold():
                continue
            historical_name = self._name_only_response(message.content)
            if historical_name:
                return historical_name
        return self._get_state_text(
            existing_db_state,
            ("client_name", "full_name", "lead_name", "name"),
        )

    def _last_assistant_asked_for_name(
        self,
        recent_chat_history: list[Message],
    ) -> bool:
        """Return whether the most recent assistant turn requested a name."""
        for message in reversed(recent_chat_history):
            if message.role == "assistant":
                return "full name" in message.content.casefold()
        return False

    def _name_only_response(self, text: str) -> str:
        """Accept a two-to-four-word answer containing only name characters."""
        normalized = " ".join(text.split())
        if not re.fullmatch(
            r"[A-Za-z][A-Za-z'’-]*(?:\s+[A-Za-z][A-Za-z'’-]*){1,3}",
            normalized,
        ):
            return ""
        return normalized

    def _latest_history_value(
        self,
        recent_chat_history: list[Message],
        value_extractor: Callable[[str], str],
    ) -> str:
        """Extract the latest known value from prior client messages."""
        for message in reversed(recent_chat_history):
            if message.role != "user":
                continue
            value = value_extractor(message.content)
            if value:
                return value
        return ""

    def _resolve_tattoo_idea(
        self,
        llm_value: str,
        current_message: str,
        recent_chat_history: list[Message],
        existing_db_state: dict[str, Any],
    ) -> str:
        """Update the concept only when the client actually states one."""
        current_idea = self._extract_tattoo_idea_from_text(current_message)
        if current_idea:
            return current_idea

        historical_idea = self._latest_history_value(
            recent_chat_history,
            self._extract_tattoo_idea_from_text,
        )
        if historical_idea:
            return historical_idea

        stored_idea = self._get_state_text(
            existing_db_state,
            ("tattoo_idea", "idea", "concept"),
        )
        stored_idea = strip_quoted_email_content(stored_idea)
        if stored_idea and not self._is_missing_tattoo_idea(stored_idea):
            return stored_idea

        normalized_llm_value = " ".join(llm_value.split())
        normalized_current = " ".join(current_message.split())
        if (
            len(normalized_llm_value) <= 120
            and normalized_llm_value.casefold() != normalized_current.casefold()
            and not self._is_missing_tattoo_idea(normalized_llm_value)
        ):
            return normalized_llm_value
        return ""

    def _get_state_text(
        self,
        existing_db_state: dict[str, Any],
        state_keys: tuple[str, ...],
    ) -> str:
        """Return the first non-empty scalar value for known state keys."""
        for record in self._state_records(existing_db_state):
            for key in state_keys:
                value = record.get(key)
                if isinstance(value, str) and not self._is_blank(value):
                    return value.strip()
                if isinstance(value, (int, float)):
                    return str(value)
        return ""

    def _has_state_value(
        self,
        existing_db_state: dict[str, Any],
        state_keys: tuple[str, ...],
    ) -> bool:
        """Return whether database state contains a meaningful value."""
        for record in self._state_records(existing_db_state):
            for key in state_keys:
                value = record.get(key)
                if isinstance(value, str) and not self._is_blank(value):
                    return True
                if isinstance(value, (list, tuple, set, dict)) and value:
                    return True
                if value is not None and not isinstance(
                    value,
                    (str, list, tuple, set, dict),
                ):
                    return True
        return False

    def _has_reference_images(
        self,
        existing_db_state: dict[str, Any],
    ) -> bool:
        """Find saved reference images at intake or per-project level."""
        if self._has_state_value(
            existing_db_state,
            (
                "reference_images",
                "image_urls",
                "previous_image_urls",
                "new_image_urls",
                "images",
                "references",
            ),
        ):
            return True

        for record in self._state_records(existing_db_state):
            projects = record.get("projects")
            if not isinstance(projects, list):
                continue
            for project in projects:
                if not isinstance(project, dict):
                    continue
                references = project.get("reference_image_urls")
                if isinstance(references, (list, tuple, set)) and references:
                    return True
        return False

    def _state_records(
        self,
        existing_db_state: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Return root and supported nested backend intake records."""
        records = [existing_db_state]
        for key in ("intake", "lead", "latest_ai_analysis"):
            value = existing_db_state.get(key)
            if isinstance(value, dict):
                records.append(value)
        return records

    def _user_conversation_text(
        self,
        current_message: str,
        recent_chat_history: list[Message],
    ) -> str:
        """Combine current and prior user messages for deterministic checks."""
        user_messages = [
            message.content
            for message in recent_chat_history
            if message.role == "user"
        ]
        return " ".join([*user_messages, current_message])

    def _user_history_text(
        self,
        recent_chat_history: list[Message],
    ) -> str:
        """Combine only prior client messages for style fallback checks."""
        return " ".join(
            message.content
            for message in recent_chat_history
            if message.role == "user"
        )

    def _mentions_reference_image(self, text: str) -> bool:
        """Detect previously supplied reference images in conversation text."""
        normalized = text.lower()
        if "http://" in normalized or "https://" in normalized:
            return True
        pattern = (
            r"\b(sent|shared|attached|uploaded)\b.{0,30}"
            r"\b(image|photo|picture|reference)\b"
        )
        return bool(re.search(pattern, normalized))

    def _declines_reference_image(self, text: str) -> bool:
        """Detect that the client cannot or does not have a reference."""
        normalized = " ".join(text.casefold().split())
        patterns = (
            r"\b(?:i\s+)?(?:do not|don't|dont|does not|doesn't|doesnt)\s+"
            r"have\b.{0,25}\b(?:reference|image|photo|picture)s?\b",
            r"\b(?:i\s+)?(?:have|got)\s+no\b.{0,25}"
            r"\b(?:reference|image|photo|picture)s?\b",
            r"\b(?:no|without)\s+(?:a\s+|any\s+)?(?:reference|"
            r"reference image|image|photo|picture)s?\b",
            r"\b(?:cannot|can't|cant|unable to)\s+"
            r"(?:send|share|provide|upload)\b.{0,25}"
            r"\b(?:reference|image|photo|picture)s?\b",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    def _is_missing_tattoo_idea(self, value: str) -> bool:
        """Reject blank and generic values that contain no tattoo concept."""
        if self._is_blank(value):
            return True
        normalized = " ".join(
            re.sub(r"[^\w\s'-]", " ", value.casefold()).split()
        )
        if normalized in {
            "unknown",
            "not provided",
            "not sure",
            "unsure",
            "no idea",
            "i have no idea",
            "i don't know",
            "i dont know",
            "do not know",
            "general inquiry",
            "general tattoo inquiry",
            "tattoo help",
        }:
            return True
        if _ARTIST_ONLY_IDEA_PATTERN.fullmatch(normalized):
            return True
        style_only = normalized.removesuffix(" tattoo").strip()
        if style_only in _STYLE_ONLY_IDEA_VALUES:
            return True
        words = set(re.findall(r"[a-z]+", normalized))
        if words and words.issubset(_CONCEPTLESS_INQUIRY_WORDS):
            return True
        return bool(
            _GENERIC_TATTOO_IDEA_PATTERN.fullmatch(normalized)
            or _GENERAL_HELP_PATTERN.fullmatch(normalized)
        )

    def _is_missing_client_name(self, value: str) -> bool:
        """Require at least two name parts for the requested full name."""
        if self._is_blank(value):
            return True
        return len(value.split()) < 2

    def _extract_client_name_from_text(self, text: str) -> str:
        """Extract an explicitly introduced client full name."""
        pattern = re.compile(
            r"\b(?:my\s+full\s+name\s+is|my\s+name\s+is|call\s+me)\s+"
            r"([A-Za-z][A-Za-z'’-]*(?:\s+[A-Za-z][A-Za-z'’-]*){1,3})"
            r"(?=\s+(?:and|but|i|my|the)\b|[,.;!?\n]|$)",
            flags=re.IGNORECASE,
        )
        matches = list(pattern.finditer(text))
        if not matches:
            return ""
        return " ".join(matches[-1].group(1).split())

    def _extract_tattoo_idea_from_text(self, text: str) -> str:
        """Extract a design subject without treating logistics as the idea."""
        normalized = " ".join(text.split())
        if not normalized or normalized == _IMAGE_ONLY_MESSAGE:
            return ""
        normalized = " ".join(
            _CLARIFICATION_QUESTION_PATTERN.sub(" ", normalized).split()
        )
        if not normalized:
            return ""

        separate_ideas = self._extract_same_client_tattoo_ideas(normalized)
        if len(separate_ideas) > 1:
            placement = self._extract_placement_from_text(normalized)
            return self._combined_tattoo_idea(separate_ideas, placement)

        quoted_wording = self._extract_quoted_wording_idea(normalized)
        if quoted_wording:
            return quoted_wording

        reference_idea = self._extract_reference_based_idea(normalized)

        candidates: list[tuple[int, str]] = []
        if reference_idea:
            candidates.append((0, reference_idea))
        idea_patterns = (
            re.compile(
                r"\b(?:tattoo\s+)?(?:idea|concept|design|background story)"
                r"\s*(?:is|would be|:)\s+(?:a|an|the)?\s*"
                r"(?P<idea>[^,.;!?]{1,120})",
                flags=re.IGNORECASE,
            ),
            re.compile(
                r"\btattoo\s+(?:of|featuring|with)\s+"
                r"(?:a|an|the)?\s*(?P<idea>[^,.;!?]{1,120})",
                flags=re.IGNORECASE,
            ),
            re.compile(
                r"\b(?:want|would like|need|planning|get|getting|have)\s+"
                r"(?:to\s+(?:get|have)\s+)?(?:a|an|the)?\s*"
                r"(?P<idea>[^,.;!?]{1,100}?)\s+tattoo\b",
                flags=re.IGNORECASE,
            ),
        )
        for pattern in idea_patterns:
            for match in pattern.finditer(normalized):
                candidates.append((match.start(), match.group("idea")))

        modifier_pattern = "|".join(
            re.escape(value)
            for value in sorted(
                _TATTOO_IDEA_MODIFIERS,
                key=len,
                reverse=True,
            )
        )
        subject_pattern = "|".join(
            re.escape(value)
            for value in sorted(
                _TATTOO_SUBJECT_WORDS,
                key=len,
                reverse=True,
            )
        )
        recognizable_subject = re.compile(
            rf"\b(?P<idea>(?:(?:{modifier_pattern})\s+){{0,4}}"
            rf"(?:{subject_pattern}))\b",
            flags=re.IGNORECASE,
        )
        for match in recognizable_subject.finditer(normalized):
            candidates.append((match.start(), match.group("idea")))

        cleaned_candidates = [
            (position, self._clean_tattoo_idea_candidate(candidate))
            for position, candidate in candidates
        ]
        usable = [item for item in cleaned_candidates if item[1]]
        if not usable:
            return ""
        return max(usable, key=lambda item: item[0])[1]

    def _extract_same_client_tattoo_ideas(self, text: str) -> list[str]:
        """Extract enumerated tattoos without treating them as people."""
        normalized = " ".join(text.split())
        clauses = re.split(
            r"\s*,\s*(?:and\s+)?(?=another\b)|"
            r"\s+and\s+(?=another\b)",
            normalized,
            flags=re.IGNORECASE,
        )
        ideas: list[str] = []
        for clause in clauses:
            candidate = self._enumerated_idea_from_clause(clause)
            if candidate and candidate.casefold() not in {
                value.casefold() for value in ideas
            }:
                ideas.append(candidate)
        if len(ideas) > 1:
            return ideas

        placed_idea_pattern = re.compile(
            r"(?P<idea>[a-z][a-z -]{1,45}?)\s+on\s+"
            r"(?:the\s+)?(?:first|second|third|fourth|one|another)\s+"
            r"(?:finger|toe)\b",
            flags=re.IGNORECASE,
        )
        for match in placed_idea_pattern.finditer(normalized):
            candidate = self._clean_enumerated_idea(match.group("idea"))
            if candidate:
                ideas.append(candidate)
        return ideas if len(ideas) > 1 else []

    def _enumerated_idea_from_clause(self, clause: str) -> str:
        """Extract one design from an enumerated same-client clause."""
        patterns = (
            re.compile(
                r"\b(?:on\s+)?(?:one|first|second|third|fourth|another)"
                r"(?:\s+one|\s+finger|\s+toe)?\s+(?:it\s+)?"
                r"(?:will|would|should)?\s*(?:be|have)?\s*"
                r"(?P<idea>.+)$",
                flags=re.IGNORECASE,
            ),
            re.compile(
                r"\b(?P<idea>[a-z][a-z -]{1,45}?)\s+on\s+"
                r"(?:the\s+)?(?:first|second|third|fourth|one|another)\s+"
                r"(?:finger|toe)\b",
                flags=re.IGNORECASE,
            ),
        )
        for pattern in patterns:
            match = pattern.search(clause)
            if match is None:
                continue
            candidate = self._clean_enumerated_idea(match.group("idea"))
            if candidate:
                return candidate
        return ""

    def _clean_enumerated_idea(self, value: str) -> str:
        """Normalize one short item from a multi-tattoo description."""
        candidate = " ".join(value.split()).strip(" ,.;:-")
        candidate = re.sub(
            r"^(?:it\s+will\s+be\s+|will\s+be\s+|be\s+|in\s+|"
            r"a\s+|an\s+|the\s+)+",
            "",
            candidate,
            flags=re.IGNORECASE,
        )
        candidate = re.sub(
            r"\s+(?:shaped?\s+)?tattoos?$|\s+shapes?$",
            "",
            candidate,
            flags=re.IGNORECASE,
        )
        candidate = re.sub(
            r"\bdiamond\s+shaped\b",
            "diamond",
            candidate,
            flags=re.IGNORECASE,
        )
        candidate = re.split(
            r"\b(?:maybe|approximately|about|around)\b.{0,25}"
            r"\b(?:cm|centimet)",
            candidate,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip(" ,.;:-")
        candidate = re.sub(
            r"\s+(?:shaped?\s+)?tattoos?$|\s+shapes?$",
            "",
            candidate,
            flags=re.IGNORECASE,
        )
        if not candidate or len(candidate.split()) > 8:
            return ""
        return candidate.casefold()

    def _combined_tattoo_idea(
        self,
        ideas: list[str],
        placement: str,
    ) -> str:
        """Create one flat concept that preserves every separate tattoo."""
        count = self._count_label(len(ideas)).capitalize()
        placement_label = ""
        if "finger" in placement.casefold():
            placement_label = " finger"
        elif "toe" in placement.casefold():
            placement_label = " toe"
        joined = self._natural_join(ideas)
        return f"{count}{placement_label} tattoos: {joined}"

    def _natural_join(self, values: list[str]) -> str:
        """Join short extraction values using natural punctuation."""
        if len(values) == 1:
            return values[0]
        if len(values) == 2:
            return f"{values[0]} and {values[1]}"
        return f"{', '.join(values[:-1])}, and {values[-1]}"

    def _count_label(self, count: int) -> str:
        """Return a readable label for a small count."""
        words = {
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five",
            6: "six",
            7: "seven",
            8: "eight",
            9: "nine",
            10: "ten",
        }
        return words.get(count, str(count))

    def _extract_reference_based_idea(self, text: str) -> str:
        """Recognize a clearly named tattoo reference as a usable concept."""
        patterns = (
            re.compile(
                r"\b(?:look|looks|looking)\s+like\s+"
                r"(?P<reference>[A-Z][A-Za-z'\u2019-]+(?:\s+"
                r"[A-Z][A-Za-z'\u2019-]+){0,3})(?:'s|\u2019s)\s+"
                r"(?:one|tattoo|design)\b",
            ),
            re.compile(
                r"\b(?:inspired\s+by|based\s+on)\s+"
                r"(?P<reference>[A-Z][A-Za-z'\u2019-]+(?:\s+"
                r"[A-Z][A-Za-z'\u2019-]+){0,3})(?:'s|\u2019s)?\s+"
                r"(?:tattoo|design)\b",
            ),
        )
        for pattern in patterns:
            match = pattern.search(text)
            if match is None:
                continue
            reference = " ".join(match.group("reference").split())
            return f"{reference}-inspired design"
        return ""

    def _extract_quoted_wording_idea(self, text: str) -> str:
        """Preserve explicitly quoted tattoo wording as one complete concept."""
        matches = list(
            re.finditer(
                r'["\u201c](?P<wording>[^"\u201d\n]{2,200})["\u201d]',
                text,
            )
        )
        for match in reversed(matches):
            prefix = text[max(0, match.start() - 180) : match.start()]
            suffix = text[match.end() : match.end() + 100]
            context = f"{prefix} {suffix}".casefold()
            describes_wording = re.search(
                r"\b(?:tattoo\s+(?:idea|concept|design)|wording|words?|text|"
                r"phrase|quote|written|write|saying|say)\b",
                context,
            )
            tattoo_request = (
                "tattoo" in prefix.casefold()
                and re.search(
                    r"\b(?:want|would like|need|planning|get|getting)\b",
                    prefix.casefold(),
                )
            )
            if not describes_wording and not tattoo_request:
                continue

            wording = " ".join(match.group("wording").split()).strip()
            if not wording:
                continue
            descriptor = (
                "Calligraphy wording"
                if "calligraph" in context
                else "Wording"
            )
            return f'{descriptor}: "{wording}"'
        return ""

    def _clean_tattoo_idea_candidate(self, value: str) -> str:
        """Normalize a short extracted concept and reject generic wording."""
        candidate = " ".join(value.split())
        candidate = re.sub(
            r"^(?:(?:it(?:'s|\s+is)|this\s+is)\s+)?"
            r"(?:basically|actually|just)\s+",
            "",
            candidate,
            flags=re.IGNORECASE,
        )
        candidate = re.sub(
            r"\bfeathres?\b",
            "feathers",
            candidate,
            flags=re.IGNORECASE,
        )
        candidate = re.sub(
            r"\b(?:the\s+)?eagle(?:'s|s)?\s+"
            r"(?:fins?\s*/\s*)?feathers\b",
            "eagle feathers",
            candidate,
            flags=re.IGNORECASE,
        )
        candidate = re.split(
            r"\s+(?:on|for)\s+(?:my|the)\b|\s+instead\b",
            candidate,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        candidate = re.sub(
            r"^(?:a|an|the)\s+|\s+(?:please|instead)$",
            "",
            candidate,
            flags=re.IGNORECASE,
        ).strip(" -")
        if not candidate or candidate.casefold() in {"new", "new tattoo"}:
            return ""
        if self._is_missing_tattoo_idea(candidate):
            return ""
        return candidate[:1].upper() + candidate[1:120]

    def _extract_party_size(self, text: str) -> int:
        """Extract common individual and group wording without guessing."""
        normalized = " ".join(text.casefold().split())
        number_words = {
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
        }
        candidates = [1]
        for match in re.finditer(
            r"\b(?:group\s+of|for|we\s+are|there\s+are)\s+"
            r"(?P<count>\d{1,2}|two|three|four|five|six|seven|eight|nine|ten)"
            r"\s+(?:people|persons?|friends?|clients?)\b",
            normalized,
        ):
            raw = match.group("count")
            count = int(raw) if raw.isdigit() else number_words[raw]
            if 1 <= count <= 20:
                candidates.append(count)
        if re.search(
            r"\b(?:me|myself)\s+and\s+my\s+"
            r"(?:girlfriend|boyfriend|partner|wife|husband|friend|sister|brother)\b",
            normalized,
        ) or re.search(
            r"\bmy\s+(?:girlfriend|boyfriend|partner|wife|husband|friend|"
            r"sister|brother)\s+and\s+(?:me|i)\b",
            normalized,
        ):
            candidates.append(2)
        if re.search(r"\b(?:both\s+of\s+us|the\s+two\s+of\s+us)\b", normalized):
            candidates.append(2)
        if re.search(
            r"\bmatch(?:ing)?\b.{0,50}\bwith\s+my\s+"
            r"(?:girlfriend|boyfriend|partner|wife|husband|friend|"
            r"sister|brother)\b",
            normalized,
        ):
            candidates.append(2)
        return max(candidates)

    def _extract_size_description(self, text: str) -> str:
        """Return natural size wording when an exact measurement is unavailable."""
        normalized = " ".join(text.casefold().split())
        descriptions = (
            (
                r"\b(?:hand[- ]?sized?|size\s+of\s+(?:a|my|your)\s+hand|"
                r"fit(?:s|ting)?\s+(?:on|across)\s+"
                r"(?:a|my|the|your)\s+hand)\b",
                "hand-sized",
            ),
            (
                r"\b(?:palm[- ]?sized?|size\s+of\s+(?:a|my|your)\s+palm|"
                r"fit(?:s|ting)?\s+(?:on|across)\s+"
                r"(?:a|my|the|your)\s+palm)\b",
                "palm-sized",
            ),
            (
                r"\b(?:full[- ]?chest|whole\s+chest|entire\s+chest|"
                r"across\s+(?:my|the|your)\s+(?:whole\s+)?chest|"
                r"cover(?:s|ing)?\s+(?:my|the|your)\s+(?:both|whole|"
                r"entire|full)\s+chest)\b",
                "full-chest",
            ),
            (
                r"\b(?:full[- ]?back|whole\s+back|entire\s+back|"
                r"cover(?:s|ing)?\s+(?:my|the|your)\s+(?:whole|entire|"
                r"full)\s+back)\b",
                "full-back",
            ),
            (r"\bfull[- ]?sleeve\b", "full-sleeve"),
            (r"\bhalf[- ]?sleeve\b", "half-sleeve"),
            (
                r"\b(?:full[- ]?leg|whole\s+leg|entire\s+leg)\b",
                "full-leg",
            ),
            (r"\bhalf[- ]?leg\b", "half-leg"),
            (r"\bcredit[- ]?card[- ]?sized?\b", "credit-card-sized"),
            (r"\bcoin[- ]?sized?\b", "coin-sized"),
            (r"\bmatchbox[- ]?sized?\b", "matchbox-sized"),
        )
        for pattern, description in descriptions:
            if re.search(pattern, normalized):
                return description

        placement = self._extract_placement_from_text(normalized)
        if placement and self._placement_is_used_as_size(
            text=normalized,
            placement=placement,
        ):
            return self._placement_size_description(placement)
        if self._is_uncertain_answer(normalized) and re.search(
            r"\b(?:size|large|big|small|centimet|\bcm\b)\b",
            normalized,
        ):
            placement_description = self._placement_size_description(placement)
            if placement_description:
                return placement_description
            return "not sure"
        return ""

    def _placement_is_used_as_size(self, text: str, placement: str) -> bool:
        """Detect when a body area describes coverage rather than location."""
        aliases = [
            alias
            for alias, canonical in _PLACEMENT_ALIASES
            if canonical == placement
        ]
        alias_pattern = "|".join(
            re.escape(alias) for alias in sorted(aliases, key=len, reverse=True)
        )
        if not alias_pattern:
            return False
        patterns = (
            rf"\b(?:{alias_pattern})[- ]sized?\b",
            rf"\bsize\s+of\s+(?:a|my|the|your)\s+(?:{alias_pattern})\b",
            rf"\bcover(?:s|ing)?\s+(?:my|the|your)\s+"
            rf"(?:whole\s+|entire\s+|full\s+|both\s+)?"
            rf"(?:{alias_pattern})\b",
            rf"\bacross\s+(?:my|the|your)\s+(?:{alias_pattern})\b",
        )
        return any(re.search(pattern, text) for pattern in patterns)

    def _placement_size_description(self, placement: str) -> str:
        """Return a stable qualitative key for any resolved body placement."""
        normalized = " ".join(placement.casefold().split())
        if not normalized:
            return ""
        return f"{normalized}-sized"

    def _infer_size(
        self,
        size_estimate_cm: str,
        size_description: str,
    ) -> str:
        """Infer a conservative centimetre range from qualitative sizing."""
        if not self._is_blank(size_estimate_cm):
            return size_estimate_cm
        normalized = " ".join(size_description.casefold().split())
        configured_range = _QUALITATIVE_SIZE_RANGES.get(normalized)
        if configured_range:
            return configured_range
        if normalized.endswith("-sized"):
            return _DEFAULT_PLACEMENT_SIZE_RANGE
        return ""

    def _is_uncertain_answer(self, text: str) -> bool:
        """Detect an explicit statement that the client does not know."""
        normalized = " ".join(text.casefold().split())
        return bool(
            re.search(
                r"\b(?:not sure|unsure|"
                r"(?:do not|don't|dont)\s+(?:really\s+|exactly\s+)?know|"
                r"no idea|you decide|please advise)\b",
                normalized,
            )
        )

    def _size_status(
        self,
        size_estimate_cm: str,
        size_description: str,
        current_message: str,
    ) -> str:
        """Classify whether a supplied size is exact, approximate, or unknown."""
        if size_description:
            if size_description == "not sure":
                return "unknown"
            return "approximate"
        if size_estimate_cm:
            is_range = bool(
                re.search(
                    r"\d(?:\.\d+)?\s*(?:-|to)\s*\d",
                    size_estimate_cm,
                    flags=re.IGNORECASE,
                )
            )
            if is_range or re.search(
                r"\b(?:about|around|approximately|approx|roughly|maybe)\b",
                current_message,
                flags=re.IGNORECASE,
            ):
                return "approximate"
            return "exact"
        return "unknown"

    def _extract_artist_preference_mode(
        self,
        text: str,
    ) -> ArtistPreferenceMode:
        """Interpret recommendation requests as a valid artist answer."""
        normalized = " ".join(text.casefold().split())
        if re.search(
            r"\b(?:recommend|suggest)\b.{0,45}\b(?:artist|best fit|who)\b|"
            r"\bwhich\s+artist\b.{0,45}\b(?:recommend|suggest|best)\b|"
            r"\b(?:best fit|best artist)\b|"
            r"\b(?:do not|don't|dont)\s+know\b.{0,35}\bartist\b|"
            r"\bwho\b.{0,30}\b(?:best|better)\b|"
            r"\b(?:best|better)\b.{0,30}\b(?:among them|for me)\b|"
            r"\b(?:do not|don't|dont)\s+know\b.{0,25}"
            r"\bthem\b.{0,20}\bpersonally\b",
            normalized,
        ):
            return "recommend"
        if re.search(
            r"\b(?:no\s+(?:artist\s+)?preference|any\s+artist|"
            r"whoever\s+(?:is|you\s+think)|you\s+(?:can\s+)?choose)\b",
            normalized,
        ):
            return "no_preference"
        if self._extract_preferred_artist_from_text(normalized) not in {
            "",
            "No preference",
        }:
            return "specific"
        return "unknown"

    def _resolve_projects(
        self,
        llm_projects: list[TattooProjectDetail],
        current_message: str,
        style_tags: list[StyleTag],
        resolved_output: _ExtractionSubset,
        new_image_urls: list[str],
        existing_db_state: dict[str, Any],
    ) -> list[TattooProjectDetail]:
        """Build per-person tattoo details while preserving prior project state."""
        projects = [project.model_copy(deep=True) for project in llm_projects]
        enumerated_ideas = self._extract_same_client_tattoo_ideas(
            current_message
        )
        if len(enumerated_ideas) > 1:
            individual_placement = self._individual_placement(
                resolved_output.placement
            )
            projects = [
                TattooProjectDetail(
                    person_label="Client",
                    tattoo_idea=idea.capitalize(),
                    placement=individual_placement,
                )
                for idea in enumerated_ideas
            ]
        if not projects:
            projects = self._stored_projects(existing_db_state)

        party_size = resolved_output.party_size
        labels = self._party_labels(current_message, party_size)
        if party_size > 1 and len(projects) < party_size:
            existing_by_label = {
                project.person_label.casefold(): project for project in projects
            }
            projects = [
                existing_by_label.get(label.casefold())
                or TattooProjectDetail(person_label=label)
                for label in labels
            ]
        elif not projects and party_size <= 1:
            return []

        person_colors = self._extract_person_colors(current_message)
        inferred_size = self._infer_size(
            size_estimate_cm=resolved_output.size_estimate_cm,
            size_description=resolved_output.size_description,
        )
        resolved: list[TattooProjectDetail] = []
        for index, project in enumerate(projects):
            label = project.person_label or (
                labels[index] if index < len(labels) else f"Person {index + 1}"
            )
            color = person_colors.get(label.casefold())
            if not color:
                color = project.color_preference or resolved_output.color_preference
            project_size_description = (
                project.size_description or resolved_output.size_description
            )
            project_size = self._infer_size(
                size_estimate_cm=project.size_estimate_cm or inferred_size,
                size_description=project_size_description,
            )
            project_size_status = self._size_status(
                size_estimate_cm=project_size,
                size_description=project_size_description,
                current_message=current_message,
            )
            resolved.append(
                project.model_copy(
                    update={
                        "person_label": label,
                        "tattoo_idea": (
                            project.tattoo_idea or resolved_output.tattoo_idea
                        ),
                        "style_tags": project.style_tags or style_tags,
                        "placement": (
                            project.placement or resolved_output.placement
                        ),
                        "size_estimate_cm": project_size,
                        "size_description": project_size_description,
                        "size_status": (
                            project.size_status
                            if project.size_status != "unknown"
                            else project_size_status
                        ),
                        "color_preference": color,
                        "tattoo_project_type": (
                            project.tattoo_project_type
                            or resolved_output.tattoo_project_type
                        ),
                        "reference_image_urls": (
                            new_image_urls or project.reference_image_urls
                        ),
                    }
                )
            )
        return resolved

    def _individual_placement(self, placement: str) -> str:
        """Convert a counted placement into one project's body placement."""
        normalized = " ".join(placement.casefold().split())
        normalized = re.sub(
            r"^(?:two|three|four|five|six|seven|eight|nine|ten|\d+)\s+",
            "",
            normalized,
        )
        if normalized.endswith("s"):
            normalized = normalized[:-1]
        return normalized

    def _stored_projects(
        self,
        existing_db_state: dict[str, Any],
    ) -> list[TattooProjectDetail]:
        """Read valid project objects from root or nested intake state."""
        for record in self._state_records(existing_db_state):
            raw_projects = record.get("projects")
            if not isinstance(raw_projects, list):
                continue
            projects: list[TattooProjectDetail] = []
            for item in raw_projects:
                try:
                    projects.append(TattooProjectDetail.model_validate(item))
                except (TypeError, ValueError):
                    continue
            if projects:
                return projects
        return []

    def _party_labels(self, text: str, party_size: int) -> list[str]:
        """Create stable human-readable labels for common relationship wording."""
        normalized = text.casefold()
        relationships = (
            ("girlfriend", "Girlfriend"),
            ("boyfriend", "Boyfriend"),
            ("partner", "Partner"),
            ("wife", "Wife"),
            ("husband", "Husband"),
            ("sister", "Sister"),
            ("brother", "Brother"),
            ("friend", "Friend"),
        )
        labels = ["Client"]
        for token, label in relationships:
            if re.search(rf"\b(?:my\s+)?{token}\b", normalized):
                labels.append(label)
                break
        while len(labels) < party_size:
            labels.append(f"Person {len(labels) + 1}")
        return labels[:party_size]

    def _extract_person_colors(self, text: str) -> dict[str, str]:
        """Attach named colours to the person they describe in group requests."""
        normalized = " ".join(text.casefold().split())
        color_names = r"red|blue|green|yellow|purple|orange|pink|black"
        labels = {
            "Client": r"(?:mine|my\s+one|me|i)",
            "Girlfriend": r"(?:my\s+)?girlfriend(?:'s)?",
            "Boyfriend": r"(?:my\s+)?boyfriend(?:'s)?",
            "Partner": r"(?:my\s+)?partner(?:'s)?",
            "Wife": r"(?:my\s+)?wife(?:'s)?",
            "Husband": r"(?:my\s+)?husband(?:'s)?",
            "Friend": r"(?:my\s+)?friend(?:'s)?",
        }
        found: dict[str, str] = {}
        for label, subject in labels.items():
            match = re.search(
                rf"\b{subject}\b.{{0,45}}?\b(?P<color>{color_names})\b",
                normalized,
            )
            if match:
                found[label.casefold()] = match.group("color")
        return found

    def _extract_preferred_artist_from_text(self, text: str) -> str:
        """Extract a canonical artist preference, including legacy aliases."""
        normalized = " ".join(text.casefold().replace("’", "'").split())
        aliases = _ARTIST_NAME_ALIASES
        matches: list[tuple[int, str]] = []
        no_preference_pattern = re.compile(
            r"\b(?:no\s+(?:artist\s+)?preference|any\s+artist|"
            r"you\s+(?:can\s+)?choose)\b"
        )
        for match in no_preference_pattern.finditer(normalized):
            matches.append((match.start(), "No preference"))
        for alias, display_name in aliases.items():
            escaped = re.escape(alias)
            selection_patterns = (
                rf"\b(?:prefer|preferring|preferred|choose|choosing|chose|"
                rf"chosen|select|selecting|selected)\s+"
                rf"(?:artist\s+)?{escaped}\b",
                rf"\b(?:opt|opting|opted|go|going|went)\s+for\s+{escaped}\b",
                rf"\b(?:go|going)\s+with\s+{escaped}\b",
                rf"\b(?:move|moving|go|going)\s+(?:forward|ahead)\s+with\s+"
                rf"{escaped}\b",
                rf"\b(?:proceed|proceeding|continue|continuing|stick|sticking|"
                rf"work|working)\s+with\s+{escaped}\b",
                rf"\b(?:request|requesting|pick|picking|take|taking)\s+"
                rf"{escaped}\b",
                rf"\b(?:i'd|i\s+would)\s+like\s+(?:to\s+book\s+(?:with\s+)?)?"
                rf"{escaped}\b",
                rf"\b{escaped}\s+(?:it\s+is|sounds\s+(?:good|great|perfect))\b",
                rf"\b(?:want|book)\s+(?:with\s+)?{escaped}\b",
                rf"\b(?:want|need|like|have|get)\s+{escaped}\s+"
                rf"(?:for|to|as)\b",
                rf"\b{escaped}\b.{{0,25}}\b(?:is|as)\s+my\s+"
                rf"(?:preferred\s+)?(?:tattoo\s+)?artist\b",
                rf"\bmy\s+(?:preferred\s+)?(?:tattoo\s+)?artist\s+"
                rf"(?:is|would\s+be)\s+{escaped}\b",
                rf"^(?:i(?:'d|\s+would)\s+(?:prefer\s+)?)?"
                rf"{escaped}(?:\s+please)?[.!\s]*$",
            )
            for pattern in selection_patterns:
                for match in re.finditer(pattern, normalized):
                    matches.append((match.start(), display_name))
        if matches:
            return max(matches, key=lambda item: item[0])[1]

        return ""

    def _normalize_preferred_artist(self, value: str) -> str:
        """Normalize artist spelling and reject unsupported preferences."""
        if self._is_blank(value):
            return ""
        extracted = self._extract_preferred_artist_from_text(value)
        return extracted

    def _extract_appointment_type_from_text(self, text: str) -> str:
        """Extract the client's online or studio-visit preference."""
        normalized = " ".join(text.casefold().split())
        matches: list[tuple[int, str]] = []
        patterns = (
            (r"\bonline(?:\s+(?:appointment|consultation))?\b", "online"),
            (
                r"\b(?:studio[_ -]?visit|visit(?:ing)?\s+(?:the\s+)?studio|"
                r"visit(?:ing)?\s+to\s+(?:(?:the|your)\s+)?studio|"
                r"(?:come|came|coming)\s+to\s+(?:(?:the|your)\s+)?studio|"
                r"in[- ]person)\b",
                "studio_visit",
            ),
        )
        for pattern, appointment_type in patterns:
            for match in re.finditer(pattern, normalized):
                matches.append((match.start(), appointment_type))
        if not matches:
            return ""
        return max(matches, key=lambda item: item[0])[1]

    def _normalize_appointment_type(self, value: str) -> str:
        """Normalize the appointment preference to the public taxonomy."""
        if self._is_blank(value):
            return ""
        return self._extract_appointment_type_from_text(value)

    def _extract_project_type_from_text(self, text: str) -> str:
        """Extract the tattoo project category requested by the client."""
        normalized = text.casefold()
        patterns = (
            (r"\bcover[- ]?up\b", "cover-up"),
            (r"\btouch[- ]?up\b", "touch-up"),
            (r"\b(?:continuation|continue|continuing)\b", "continuation"),
            (r"\bnew\s+tattoo\b", "new tattoo"),
        )
        matches: list[tuple[int, str]] = []
        for pattern, project_type in patterns:
            for match in re.finditer(pattern, normalized):
                matches.append((match.start(), project_type))
        if not matches:
            return ""
        return max(matches, key=lambda item: item[0])[1]

    def _normalize_project_type(self, value: str) -> str:
        """Normalize the tattoo project type to the response taxonomy."""
        if self._is_blank(value):
            return ""
        return self._extract_project_type_from_text(value)

    def _extract_availability_from_text(self, text: str) -> str:
        """Extract an exact date or a useful general availability phrase."""
        date_value = self._extract_date_from_text(text)
        if date_value:
            time_value = self._extract_time_from_text(text)
            if time_value:
                return f"{date_value} at {time_value}"
            return date_value

        patterns = (
            r"\b(?:weekdays?|weekends?)(?:\s+(?:mornings?|afternoons?|"
            r"evenings?))?\b",
            r"\b(?:mornings?|afternoons?|evenings?)\b",
            r"\b(?:any\s+day|any\s+time|anytime|flexible)\b",
            r"\bnext\s+(?:week|month)\b",
            r"\bafter\s+work\b",
        )
        matches: list[re.Match[str]] = []
        for pattern in patterns:
            matches.extend(re.finditer(pattern, text, flags=re.IGNORECASE))
        if not matches:
            return ""
        latest = max(matches, key=lambda match: match.start())
        return " ".join(latest.group(0).split())

    def _extract_size_from_text(self, text: str) -> str:
        """Extract a size and normalize imperial measurements to centimetres."""
        candidates: list[tuple[int, str]] = []
        range_spans: list[tuple[int, int]] = []
        range_pattern = re.compile(
            r"\b(?P<minimum>\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*"
            r"(?P<maximum>\d+(?:\.\d+)?)\s*"
            r"(?:cm|centimeters?|centimetres?)\b",
            flags=re.IGNORECASE,
        )
        for match in range_pattern.finditer(text):
            minimum = self._normalize_decimal(match.group("minimum"))
            maximum = self._normalize_decimal(match.group("maximum"))
            if minimum is None or maximum is None:
                continue
            candidates.append(
                (match.start(), f"{minimum}-{maximum} cm")
            )
            range_spans.append(match.span())

        cm_pattern = (
            r"\b(?P<value>\d+(?:\.\d+)?)\s*"
            r"(?:cm|centimeters?|centimetres?)\b"
        )
        for match in re.finditer(cm_pattern, text, flags=re.IGNORECASE):
            if any(
                start <= match.start() < end
                for start, end in range_spans
            ):
                continue
            normalized = self._normalize_decimal(match.group("value"))
            if normalized is None:
                continue
            candidates.append((match.start(), f"{normalized} cm"))

        inch_pattern = re.compile(
            r"\b(?P<value>\d+(?:\.\d+)?)\s*(?:inches?|inch|in)\b|"
            r"(?P<quoted>\d+(?:\.\d+)?)\s*\"",
            flags=re.IGNORECASE,
        )
        for match in inch_pattern.finditer(text):
            raw_value = match.group("value") or match.group("quoted")
            try:
                centimeters = Decimal(raw_value) * Decimal("2.54")
            except InvalidOperation:
                continue
            normalized = format(
                centimeters.quantize(Decimal("0.01")).normalize(),
                "f",
            )
            candidates.append((match.start(), f"{normalized} cm"))

        if not candidates:
            return ""
        return max(candidates, key=lambda item: item[0])[1]

    def _normalize_decimal(self, value: str) -> str | None:
        """Normalize one decimal number without insignificant zeroes."""
        try:
            return format(Decimal(value).normalize(), "f")
        except InvalidOperation:
            return None

    def _extract_placement_from_text(self, text: str) -> str:
        """Extract the latest positively stated common body placement."""
        normalized = _CLARIFICATION_QUESTION_PATTERN.sub(" ", text.casefold())
        normalized = re.sub(
            r'["\u201c][^"\u201d\n]{1,500}["\u201d]',
            " ",
            normalized,
        )
        counted_matches = list(_COUNTED_BODY_PART_PATTERN.finditer(normalized))
        if counted_matches:
            latest = counted_matches[-1]
            count = latest.group("count")
            part = latest.group("part")
            plural_part = part if part.endswith("s") else f"{part}s"
            return f"{count} {plural_part}"

        if re.search(r"\bone\s+finger\b", normalized):
            additional = len(
                re.findall(r"\banother(?:\s+one)?\b", normalized)
            )
            if additional:
                return f"{self._count_label(1 + additional)} fingers"

        reworks_existing_tattoo = bool(
            _EXISTING_TATTOO_WORK_PATTERN.search(normalized)
        )
        matches: list[tuple[int, int, str]] = []
        for alias, canonical in _PLACEMENT_ALIASES:
            pattern = rf"\b{re.escape(alias)}\b"
            for match in re.finditer(pattern, normalized):
                if not reworks_existing_tattoo and (
                    _EXISTING_TATTOO_BEFORE_PATTERN.search(
                        normalized[max(0, match.start() - 50) : match.start()]
                    )
                ):
                    continue
                suffix = normalized[match.end() : match.end() + 8]
                if alias in {"hand", "palm"} and re.match(
                    r"[- ]siz(?:e|ed)\b",
                    suffix,
                ):
                    continue
                if alias == "back" and self._is_non_body_back(
                    normalized,
                    match.start(),
                    match.end(),
                ):
                    continue
                if not self._phrase_is_negated(normalized, match.start()):
                    matches.append((match.start(), match.end(), canonical))
        if not matches:
            return ""
        specific_matches = [
            candidate
            for candidate in matches
            if not any(
                other[0] <= candidate[0]
                and other[1] >= candidate[1]
                and (other[1] - other[0]) > (candidate[1] - candidate[0])
                for other in matches
            )
        ]
        return max(specific_matches, key=lambda item: item[0])[2]

    def _is_non_body_back(self, text: str, start: int, end: int) -> bool:
        """Detect "back" used as a direction or verb rather than placement."""
        before = text[max(0, start - 30) : start]
        after = text[end : end + 20]
        return bool(
            _DIRECTIONAL_BACK_BEFORE_PATTERN.search(before)
            or _DIRECTIONAL_BACK_AFTER_PATTERN.match(after)
            or _VERB_BACK_BEFORE_PATTERN.search(before)
        )

    def _normalize_color_value(self, value: str) -> str:
        """Collapse free-text colour notes to a canonical client value."""
        if self._is_blank(value):
            return ""
        canonical = self._extract_color_from_text(value)
        if canonical:
            return canonical
        return re.split(r"[,;(]|\s+inferred\b", value, maxsplit=1)[0].strip()

    def _extract_color_from_text(self, text: str) -> str:
        """Normalize an explicit latest-message color preference."""
        normalized = text.casefold()
        specific_patterns = (
            (
                r"\b(?:(?:no|without)\s+colou?r|"
                r"black[- ]and[- ]gr[ae]y|black\s*&\s*gr[ae]y|"
                r"gr[ae]y\s*(?:and|&)?\s*black"
                r"(?:\s+colou?r(?:\s+combination)?)?|"
                r"black\s+ink(?:\s+only)?|black\s+only|"
                r"black(?:\s+colou?r)?)\b",
                "black-and-grey",
            ),
            (
                r"\b(?:full\s+colou?r|colou?r(?:ed|ful)|"
                r"red|blue|green|yellow|purple|orange|pink)\b",
                "color",
            ),
        )
        specific_matches: list[tuple[int, str]] = []
        for pattern, value in specific_patterns:
            for match in re.finditer(pattern, normalized):
                if not self._phrase_is_negated(normalized, match.start()):
                    specific_matches.append((match.start(), value))
        if specific_matches:
            return max(specific_matches, key=lambda item: item[0])[1]

        if re.search(r"\bwatercolou?r\b", normalized):
            return "color"

        generic_matches = list(re.finditer(r"\bcolou?r\b", normalized))
        if generic_matches:
            return "color"
        return ""

    def _extract_date_from_text(
        self,
        text: str,
        allow_weekdays: bool = True,
    ) -> str:
        """Extract and normalize the latest explicit preferred date."""
        today = calendar_date.today()
        candidates: list[tuple[int, calendar_date]] = []

        for match in re.finditer(r"\b\d{4}-\d{1,2}-\d{1,2}\b", text):
            parsed = self._safe_calendar_date(match.group(0))
            if parsed is not None:
                candidates.append((match.start(), parsed))

        numeric_pattern = r"\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})\b"
        for match in re.finditer(numeric_pattern, text):
            day, month, year = (int(value) for value in match.groups())
            if year < 100:
                year += 2000
            parsed = self._safe_calendar_date(
                f"{year:04d}-{month:02d}-{day:02d}"
            )
            if parsed is not None:
                candidates.append((match.start(), parsed))

        month_names = "|".join(_MONTH_NUMBERS)
        month_first = re.compile(
            rf"\b({month_names})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b"
            rf"(?!\s*:)"
            rf"(?:,?\s+(\d{{4}}))?\b",
            flags=re.IGNORECASE,
        )
        day_first = re.compile(
            rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({month_names})"
            rf"(?:\s+(\d{{4}}))?\b",
            flags=re.IGNORECASE,
        )
        for pattern, month_index, day_index, year_index in (
            (month_first, 1, 2, 3),
            (day_first, 2, 1, 3),
        ):
            for match in pattern.finditer(text):
                month = _MONTH_NUMBERS[match.group(month_index).casefold()]
                day = int(match.group(day_index))
                year_text = match.group(year_index)
                year = int(year_text) if year_text else today.year
                parsed = self._safe_calendar_date(
                    f"{year:04d}-{month:02d}-{day:02d}"
                )
                if parsed is None:
                    continue
                if year_text is None and parsed < today:
                    parsed = self._safe_calendar_date(
                        f"{year + 1:04d}-{month:02d}-{day:02d}"
                    )
                if parsed is not None:
                    candidates.append((match.start(), parsed))

        for match in re.finditer(
            r"\b(todaye?|tomorrow)\b",
            text,
            flags=re.IGNORECASE,
        ):
            offset = 1 if match.group(1).casefold() == "tomorrow" else 0
            candidates.append((match.start(), today + timedelta(days=offset)))

        weekday_names = "|".join(_WEEKDAY_NUMBERS)
        weekday_pattern = re.compile(
            rf"\b(next\s+)?({weekday_names})\b",
            flags=re.IGNORECASE,
        )
        for match in weekday_pattern.finditer(text if allow_weekdays else ""):
            weekday = _WEEKDAY_NUMBERS[match.group(2).casefold()]
            offset = (weekday - today.weekday()) % 7
            if offset == 0:
                offset += 7
            candidates.append((match.start(), today + timedelta(days=offset)))

        if not candidates:
            return ""
        return max(candidates, key=lambda item: item[0])[1].isoformat()

    def _extract_time_from_text(self, text: str) -> str:
        """Extract and normalize the latest explicit preferred time."""
        candidates: list[tuple[int, str]] = []
        twelve_hour_pattern = re.compile(
            r"\b(1[0-2]|0?[1-9])(?::([0-5]\d))?\s*(am|pm)\b",
            flags=re.IGNORECASE,
        )
        for match in twelve_hour_pattern.finditer(text):
            hour = int(match.group(1)) % 12
            if match.group(3).casefold() == "pm":
                hour += 12
            minute = int(match.group(2) or 0)
            candidates.append((match.start(), f"{hour:02d}:{minute:02d}"))

        twenty_four_hour_pattern = re.compile(
            r"(?<!\d)([01]?\d|2[0-3]):([0-5]\d)(?!\d)"
        )
        for match in twenty_four_hour_pattern.finditer(text):
            candidates.append(
                (
                    match.start(),
                    f"{int(match.group(1)):02d}:{int(match.group(2)):02d}",
                )
            )

        if not candidates:
            return ""
        return max(candidates, key=lambda item: item[0])[1]

    def _normalize_date_value(self, value: str) -> str:
        """Normalize model or backend date text to the public contract."""
        if self._is_blank(value):
            return ""
        return self._extract_date_from_text(value)

    def _normalize_time_value(self, value: str) -> str:
        """Normalize model or backend time text to the public contract."""
        if self._is_blank(value):
            return ""
        return self._extract_time_from_text(value)

    def _safe_calendar_date(self, value: str) -> calendar_date | None:
        """Parse an ISO-like calendar date without allowing invalid dates."""
        try:
            return calendar_date.fromisoformat(value)
        except ValueError:
            return None

    def _mentions_field(
        self,
        text: str,
        field_terms: tuple[str, ...],
    ) -> bool:
        """Return whether the latest message intentionally addresses a field."""
        normalized = text.casefold()
        return any(
            re.search(
                rf"(?<!\w){re.escape(term.strip().casefold())}(?!\w)",
                normalized,
            )
            is not None
            for term in field_terms
            if term.strip()
        )

    def _has_positive_phrase(self, text: str, phrase: str) -> bool:
        """Return whether a phrase appears outside a rejection expression."""
        pattern = rf"\b{re.escape(phrase)}\b"
        return any(
            not self._phrase_is_negated(text, match.start())
            for match in re.finditer(pattern, text)
        )

    def _has_negated_phrase(self, text: str, phrase: str) -> bool:
        """Return whether a phrase is explicitly rejected in nearby text."""
        pattern = rf"\b{re.escape(phrase)}\b"
        return any(
            self._phrase_is_negated(text, match.start())
            for match in re.finditer(pattern, text)
        )

    def _phrase_is_negated(self, text: str, phrase_start: int) -> bool:
        """Detect a rejection marker immediately before a matched phrase."""
        prefix = text[max(0, phrase_start - 24):phrase_start]
        return bool(_NEGATION_PREFIX_PATTERN.search(prefix))

    def _is_blank(self, value: str) -> bool:
        """Return True when the extracted field is effectively empty."""
        normalized = value.strip().lower()
        return normalized in {"", "unknown", "n/a", "none", "not provided"}
