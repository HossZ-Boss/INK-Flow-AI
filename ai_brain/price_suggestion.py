"""Staff-only AI price suggestions for high-risk tattoo requests."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Sequence
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .decision_schemas import DecisionHistoryExample
from .llm_health import llm_failure_context
from .pricing import parse_size_cm
from .prompts import PRICE_SUGGESTION_SYSTEM_PROMPT
from .schemas import AIExtractionOutput, ConfidenceLevel, PriceSuggestion

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

    from .vector_store import VectorStoreManager

LOGGER = logging.getLogger(__name__)

_MAX_IMAGES = 3
_MAX_ANCHORS = 5
_PRICE_CEILING = 20000
_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", flags=re.DOTALL)

ImageLoader = Callable[[list[str]], list[str]]


class _PriceLLMOutput(BaseModel):
    """Raw model estimate before rounding and sanity checks."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    minimum: float = Field(gt=0)
    maximum: float = Field(gt=0)
    estimated_hours: str = ""
    complexity: str = "unknown"
    confidence_level: ConfidenceLevel = "low"
    reasoning: str = ""


class TattooPriceSuggester:
    """Estimate a price range from tattoo details, images, and approved cases.

    The language model provides the market-based estimate. Prices staff have
    approved for similar requests (from the learning index) are passed to the
    model as anchors, and are used on their own when the model is unavailable.
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        vector_store: VectorStoreManager | None = None,
        image_loader: ImageLoader | None = None,
    ) -> None:
        self._llm = llm
        self._vector_store = vector_store
        self._image_loader = image_loader

    def suggest(
        self,
        analysis: AIExtractionOutput,
        image_urls: Sequence[str] = (),
        visual_description: str = "",
        channel: str = "whatsapp",
    ) -> PriceSuggestion | None:
        """Return a staff-only price range, or None when none is possible."""
        anchors = self._approved_anchors(analysis)
        try:
            estimate = self._llm_estimate(
                analysis=analysis,
                images=self._load_images(image_urls),
                visual_description=visual_description,
                anchors=anchors,
                channel=channel,
            )
        except Exception as exc:
            LOGGER.warning(
                "Price suggestion fallback used: %s [%s]",
                exc,
                llm_failure_context(exc),
            )
            return self._history_only_suggestion(anchors)
        return estimate

    def _llm_estimate(
        self,
        analysis: AIExtractionOutput,
        images: list[str],
        visual_description: str,
        anchors: list[DecisionHistoryExample],
        channel: str,
    ) -> PriceSuggestion:
        """Ask the model for a range and normalize it to whole euros."""
        details = {
            "tattoo_idea": analysis.tattoo_idea,
            "style_tags": [tag for tag in analysis.style_tags if tag != "unknown"],
            "placement": analysis.placement,
            "size_cm": analysis.size_estimate_cm,
            "color_preference": analysis.color_preference,
            "tattoo_project_type": analysis.tattoo_project_type,
            "suggested_artist": analysis.suggested_artist,
            "reference_image_description": visual_description,
            "reference_images_attached": len(images),
            "channel": channel,
        }
        approved = [
            {
                "style_tags": list(example.style_tags),
                "placement": example.placement,
                "size_cm": example.size_estimate_cm,
                "color_preference": example.color_preference,
                "approved_price": {
                    "currency": example.approved_price_range.currency,
                    "minimum": str(example.approved_price_range.minimum),
                    "maximum": str(example.approved_price_range.maximum),
                },
            }
            for example in anchors
            if example.approved_price_range is not None
        ]
        text = (
            "Estimate the price for this tattoo request.\n"
            f"Request details:\n{json.dumps(details, indent=2)}\n\n"
            "Studio-approved prices for similar past requests "
            "(empty when none exist yet):\n"
            f"{json.dumps(approved, indent=2)}"
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        content.extend(
            {"type": "image_url", "image_url": {"url": image}}
            for image in images
        )
        response = self._llm.invoke(
            [
                SystemMessage(content=PRICE_SUGGESTION_SYSTEM_PROMPT),
                HumanMessage(content=content),
            ]
        )
        raw = self._parse(self._coerce_text(response.content))
        minimum, maximum = sorted(
            (self._round_price(raw.minimum), self._round_price(raw.maximum))
        )
        if maximum > _PRICE_CEILING:
            raise ValueError("Price estimate exceeds the sanity ceiling.")
        complexity = (
            raw.complexity
            if raw.complexity in {"low", "medium", "high"}
            else "unknown"
        )
        return PriceSuggestion(
            minimum=minimum,
            maximum=maximum,
            estimated_hours=raw.estimated_hours[:40],
            complexity=complexity,
            confidence_level=raw.confidence_level,
            basis=(
                "ai_with_approved_history" if approved else "ai_market_estimate"
            ),
            reasoning=raw.reasoning[:600],
            similar_example_ids=[example.example_id for example in anchors],
        )

    def _history_only_suggestion(
        self,
        anchors: list[DecisionHistoryExample],
    ) -> PriceSuggestion | None:
        """Use approved similar prices when the model cannot answer."""
        ranges = [
            example.approved_price_range
            for example in anchors
            if example.approved_price_range is not None
            and example.approved_price_range.currency == "EUR"
        ]
        if not ranges:
            return None
        return PriceSuggestion(
            minimum=self._round_price(min(item.minimum for item in ranges)),
            maximum=self._round_price(max(item.maximum for item in ranges)),
            confidence_level="medium" if len(ranges) > 1 else "low",
            basis="approved_history",
            reasoning=(
                "Based only on studio-approved prices for similar requests; "
                "the AI estimate was unavailable."
            ),
            similar_example_ids=[example.example_id for example in anchors],
        )

    def _approved_anchors(
        self,
        analysis: AIExtractionOutput,
    ) -> list[DecisionHistoryExample]:
        """Find similar staff-approved prices in the learning index."""
        if self._vector_store is None:
            return []
        styles = ", ".join(analysis.style_tags) or "unknown"
        query = (
            f"Tattoo styles: {styles}. "
            f"Placement: {analysis.placement or 'unknown'}. "
            f"Size in centimeters: {analysis.size_estimate_cm or 'unknown'}. "
            f"Color preference: {analysis.color_preference or 'unknown'}."
        )
        try:
            matches = self._vector_store.search_similar_cases(query, top_k=12)
        except Exception as exc:  # pragma: no cover - defensive branch
            LOGGER.warning("Price anchor search failed: %s", exc)
            return []
        anchors: list[DecisionHistoryExample] = []
        for _, example in matches:
            if example.approved_price_range is None:
                continue
            if not self._is_comparable(example, analysis):
                continue
            anchors.append(example)
            if len(anchors) == _MAX_ANCHORS:
                break
        return anchors

    def _is_comparable(
        self,
        example: DecisionHistoryExample,
        analysis: AIExtractionOutput,
    ) -> bool:
        """Require a shared style or placement and a similar size."""
        shared_style = bool(
            set(example.style_tags).intersection(analysis.style_tags)
            - {"unknown"}
        )
        same_placement = bool(
            example.placement
            and example.placement.casefold() == analysis.placement.casefold()
        )
        if not shared_style and not same_placement:
            return False
        current = parse_size_cm(analysis.size_estimate_cm)
        previous = parse_size_cm(example.size_estimate_cm)
        if current is None or previous is None:
            return True
        return abs(current - previous) <= max(Decimal("5"), current / 2)

    def _load_images(self, image_urls: Sequence[str]) -> list[str]:
        """Download up to three reference images as data URIs."""
        urls = list(dict.fromkeys(url for url in image_urls if url))[:_MAX_IMAGES]
        if not urls or self._image_loader is None:
            return []
        try:
            loaded = self._image_loader(urls)
        except Exception as exc:  # pragma: no cover - defensive branch
            LOGGER.warning("Price suggestion images unavailable: %s", exc)
            return []
        if not isinstance(loaded, list):
            return []
        return [item for item in loaded if isinstance(item, str)][:_MAX_IMAGES]

    def _parse(self, raw_text: str) -> _PriceLLMOutput:
        """Parse the first JSON object in the model response."""
        match = _JSON_OBJECT_PATTERN.search(raw_text)
        if match is None:
            raise ValueError("Price response did not contain JSON.")
        try:
            return _PriceLLMOutput.model_validate_json(match.group(0))
        except ValidationError as exc:
            raise ValueError("Price response failed validation.") from exc

    def _coerce_text(self, content: Any) -> str:
        """Normalize LangChain response content to plain text."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            )
        return str(content)

    def _round_price(self, value: float | Decimal) -> int:
        """Round to the nearest 10 euros, never below 10."""
        return max(10, int(round(float(value) / 10.0)) * 10)
