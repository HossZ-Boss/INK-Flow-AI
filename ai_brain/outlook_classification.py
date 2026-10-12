"""Conservative Outlook tattoo-inquiry classification."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .review_policy import is_proceed_confirmation, is_status_update_request
from .schemas import Message, TattooInquiryInput

_NO_REPLY_ADDRESS_PATTERN = re.compile(
    r"(?:^|[^a-z0-9])(?:no[-_ ]?reply|do[-_ ]?not[-_ ]?reply|"
    r"mailer[-_ ]?daemon|postmaster|bounce|notifications?|alerts?)"
    r"(?=[^a-z0-9]|$)",
    flags=re.IGNORECASE,
)
_AUTOMATED_MESSAGE_PATTERN = re.compile(
    r"\b(?:automatic\s+reply|automated\s+(?:email|message)|out\s+of\s+office|"
    r"delivery\s+(?:status\s+notification|failure)|undeliverable|"
    r"message\s+(?:was\s+)?not\s+delivered|mail\s+delivery\s+subsystem|"
    r"do\s+not\s+reply|don'?t\s+reply|list-unsubscribe|unsubscribe|"
    r"verification\s+code|one[- ]time\s+(?:code|password)|"
    r"password\s+reset|security\s+alert)\b",
    flags=re.IGNORECASE,
)
_UNRELATED_MESSAGE_PATTERN = re.compile(
    r"\b(?:job\s+application|curriculum\s+vitae|resume|recruitment|vacancy|"
    r"marketing\s+(?:proposal|services?)|seo\s+services?|guest\s+post|"
    r"sponsorship|partnership\s+proposal|supplier|purchase\s+order|"
    r"invoice|payment\s+receipt|shipping\s+notification|newsletter|"
    r"collaboration|guest\s+spot|apprenticeship|gift\s+card|voucher|"
    r"tattoo\s+(?:machine|supplies)|needles?\s+for\s+sale)\b",
    flags=re.IGNORECASE,
)
_MANUAL_ONLY_PATTERN = re.compile(
    r"\b(?:infect(?:ed|ion)|allerg(?:y|ic)|pregnan(?:t|cy)|underage|minor|"
    r"medical|aftercare|healing|redness|swelling|legal|complaint|"
    r"bad\s+experience|refund|cancel(?:lation)?|"
    r"reschedul(?:e|ing)|deposit\s+refund|tattoo\s+removal|laser\s+removal)\b",
    flags=re.IGNORECASE,
)
_COURTESY_ONLY_PATTERN = re.compile(
    r"^(?:hi|hello|hey|thanks|thank\s+you|many\s+thanks|okay|ok|got\s+it|"
    r"sounds\s+good|perfect|great)[.!\s]*$",
    flags=re.IGNORECASE,
)
_DIRECT_TATTOO_PATTERN = re.compile(
    r"\b(?:tattoo|tattoos|tattooing|tattooed|tatto|body\s+art|inked|"
    r"get\s+(?:some\s+)?ink|new\s+piece|cover[- ]?up|touch[- ]?up)\b",
    flags=re.IGNORECASE,
)
_REQUEST_INTENT_PATTERN = re.compile(
    r"\b(?:i|we)\s+(?:want|need|plan|hope|are\s+looking)|"
    r"\bi(?:'d|\s+would)\s+like|\bi(?:'m|\s+am)\s+looking|"
    r"\b(?:interested\s+in|looking\s+to\s+get)|"
    r"\b(?:can|could|would)\s+(?:i|we|you)|"
    r"\b(?:book|booking|appointment|consultation|how\s+much|cost|price|"
    r"quote|availability|available)\b",
    flags=re.IGNORECASE,
)
_STYLE_PATTERN = re.compile(
    r"\b(?:fine[- ]?line|watercolou?r|minimal(?:ist)?|micro[- ]?realism|"
    r"botanical|illustrative|abstract|black\s*work|realism|pixel[- ]?art|"
    r"new[- ]?school|neo[- ]?traditional|calligraph(?:y|ic)|traditional|"
    r"geometric|black[- ]and[- ]gr[ae]y|black\s+and\s+gr[ae]y)\b",
    flags=re.IGNORECASE,
)
_PLACEMENT_PATTERN = re.compile(
    r"\b(?:wrist|hand|finger|arm|forearm|bicep|shoulder|chest|sternum|"
    r"back|spine|rib|stomach|abdomen|belly|tummy|hip|thigh|leg|calf|knee|ankle|"
    r"foot|toe|neck|throat|face|head|scalp|ear|tongue)\b",
    flags=re.IGNORECASE,
)
_DESIGN_PATTERN = re.compile(
    r"\b(?:design|reference\s+(?:image|photo)|flower|rose|lotus|skeleton|"
    r"skull|dragon|eagle|tiger|wolf|snake|heart|sword|diamond|stars?|"
    r"portrait|lettering|quote|script|mandala|symbol)\b",
    flags=re.IGNORECASE,
)
_STUDIO_PATTERN = re.compile(
    r"\b(?:tattoo\s+hysteria|tattoo\s+studio|studio\s+visit|"
    r"visit\s+(?:to\s+)?(?:the|your)\s+studio|(?:the|your)\s+studio|"
    r"tattoo\s+artist|"
    r"hoss|hossam|nina|lana|sandra|silva|sliva|mila)\b",
    flags=re.IGNORECASE,
)
_MEASUREMENT_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:cm|centimet(?:er|re)s?|inches?|inch)\b",
    flags=re.IGNORECASE,
)
_INTAKE_ANSWER_PATTERN = re.compile(
    r"\b(?:colou?r|black|gr[ae]y|new\s+tattoo|continuation|online|"
    r"studio\s+visit|no\s+preference|reference|available|availability|"
    r"today|tomorrow|monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday|\d{1,2}(?::\d{2})?\s*(?:am|pm))\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class OutlookClassification:
    """Result of deciding whether an Outlook email may enter intake."""

    is_tattoo_inquiry: bool
    reason: str


class OutlookInquiryClassifier:
    """Fail closed unless an Outlook email belongs to tattoo intake."""

    def classify(self, inquiry: TattooInquiryInput) -> OutlookClassification:
        """Classify one validated message before expensive AI processing."""
        if inquiry.message_source != "outlook":
            return OutlookClassification(
                is_tattoo_inquiry=False,
                reason="Automatic email replies apply only to Outlook.",
            )

        sender = self._sender_email(inquiry.existing_db_state)
        if sender and _NO_REPLY_ADDRESS_PATTERN.search(sender):
            return OutlookClassification(
                is_tattoo_inquiry=False,
                reason="Outlook sender is a no-reply or automated address.",
            )

        subject = self._email_subject(inquiry.existing_db_state)
        body = " ".join(inquiry.current_message.split())
        message = " ".join(value for value in (subject, body) if value)
        if _AUTOMATED_MESSAGE_PATTERN.search(message):
            return OutlookClassification(
                is_tattoo_inquiry=False,
                reason="Outlook message is an automated notification.",
            )
        if _UNRELATED_MESSAGE_PATTERN.search(message):
            return OutlookClassification(
                is_tattoo_inquiry=False,
                reason="Outlook message is unrelated to tattoo intake.",
            )
        if _MANUAL_ONLY_PATTERN.search(message):
            return OutlookClassification(
                is_tattoo_inquiry=False,
                reason="Outlook message requires manual studio handling.",
            )
        if _COURTESY_ONLY_PATTERN.fullmatch(message):
            return OutlookClassification(
                is_tattoo_inquiry=False,
                reason="Outlook message is only a courtesy acknowledgement.",
            )
        if self._has_direct_tattoo_request(message):
            return OutlookClassification(
                is_tattoo_inquiry=True,
                reason="Outlook message contains a tattoo inquiry.",
            )
        if self._is_intake_follow_up(inquiry, message):
            return OutlookClassification(
                is_tattoo_inquiry=True,
                reason="Outlook message continues an existing tattoo intake.",
            )
        return OutlookClassification(
            is_tattoo_inquiry=False,
            reason="No reliable tattoo-intake intent was detected.",
        )

    def _has_direct_tattoo_request(self, message: str) -> bool:
        """Recognize explicit and strongly implied tattoo requests."""
        evidence = (
            bool(_STYLE_PATTERN.search(message)),
            bool(_PLACEMENT_PATTERN.search(message)),
            bool(_DESIGN_PATTERN.search(message)),
            bool(_STUDIO_PATTERN.search(message)),
            bool(_MEASUREMENT_PATTERN.search(message)),
        )
        if _DIRECT_TATTOO_PATTERN.search(message):
            return bool(_REQUEST_INTENT_PATTERN.search(message)) or any(evidence)
        if (
            _REQUEST_INTENT_PATTERN.search(message)
            and _STUDIO_PATTERN.search(message)
        ):
            return True
        return sum(evidence) >= 2

    def _is_intake_follow_up(
        self,
        inquiry: TattooInquiryInput,
        message: str,
    ) -> bool:
        """Recognize short answers only inside an established intake thread."""
        if not self._has_established_thread(
            inquiry.recent_chat_history,
            inquiry.existing_db_state,
        ):
            return False
        if inquiry.new_image_urls:
            return True
        if is_status_update_request(message):
            return True
        if is_proceed_confirmation(message):
            return True
        if _INTAKE_ANSWER_PATTERN.search(message):
            return True
        return self._last_assistant_requested_information(
            inquiry.recent_chat_history
        ) and len(message.split()) >= 2

    def _has_established_thread(
        self,
        history: list[Message],
        state: dict[str, Any],
    ) -> bool:
        """Require thread evidence before accepting an ambiguous follow-up."""
        assistant_context = any(
            message.role == "assistant" for message in history
        )
        user_context = any(
            message.role == "user"
            and self._has_direct_tattoo_request(message.content)
            for message in history
        )
        if assistant_context and user_context:
            return True

        intake = state.get("intake")
        if not isinstance(intake, dict) or not assistant_context:
            return False
        return any(
            self._known(intake.get(key))
            for key in ("tattoo_idea", "placement", "style_tags")
        )

    def _last_assistant_requested_information(
        self,
        history: list[Message],
    ) -> bool:
        """Return whether the latest assistant email contains a question."""
        for message in reversed(history):
            if message.role == "assistant":
                return "?" in message.content
        return False

    def _sender_email(self, state: dict[str, Any]) -> str:
        """Read the sender address from supported backend state locations."""
        candidates: list[Any] = [
            state.get("sender_email"),
            state.get("from_email"),
        ]
        for key in ("lead", "sender", "message"):
            record = state.get(key)
            if not isinstance(record, dict):
                continue
            candidates.extend(
                record.get(field)
                for field in ("email", "address", "sender_email", "from_email")
            )
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return ""

    def _email_subject(self, state: dict[str, Any]) -> str:
        """Read an optional Outlook subject supplied in backend state."""
        candidates: list[Any] = [
            state.get("subject"),
            state.get("email_subject"),
        ]
        message = state.get("message")
        if isinstance(message, dict):
            candidates.extend(
                (message.get("subject"), message.get("email_subject"))
            )
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return " ".join(candidate.split())
        return ""

    def _known(self, value: Any) -> bool:
        """Return whether a stored intake value contains useful context."""
        if isinstance(value, str):
            return value.strip().casefold() not in {"", "unknown", "none"}
        if isinstance(value, (list, tuple, set)):
            return bool(value)
        return value is not None
