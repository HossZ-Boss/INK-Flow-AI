"""Vision analysis module for tattoo style detection."""

from __future__ import annotations

import base64
import json
import logging
import re
from typing import Any, cast

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import AnyHttpUrl, TypeAdapter, ValidationError

from .errors import AnalysisPipelineError
from .llm import get_chat_model
from .llm_health import llm_failure_context, record_llm_failure
from .prompts import VISION_SYSTEM_PROMPT
from .schemas import (
    STYLE_TAG_OPTIONS,
    StyleTag,
    TattooVisionOutput,
    VisualColorPreference,
)

LOGGER = logging.getLogger(__name__)

_STYLE_TAG_SET = set(STYLE_TAG_OPTIONS)


class TattooVisionAnalyzer:
    """Analyze reference images for normalized tattoo style and color."""

    def __init__(
        self,
        llm: ChatOpenAI | None = None,
        model_name: str | None = None,
        timeout_seconds: float = 20.0,
    ) -> None:
        self._llm = llm or get_chat_model(model_name=model_name)
        self._timeout_seconds = timeout_seconds
        self._url_adapter = TypeAdapter(AnyHttpUrl)

    def analyze_images(self, image_urls: list[str]) -> TattooVisionOutput:
        """Detect tattoo style and color from image URLs.

        Invalid URLs, download issues, or API failures return unknown values.
        """
        if not image_urls:
            return self._unknown_output()

        data_uris: list[str] = []
        for image_url in image_urls:
            data_uri = self._download_as_data_uri(image_url)
            if data_uri:
                data_uris.append(data_uri)

        if not data_uris:
            return self._unknown_output()

        try:
            raw_text = self._invoke_vision_model(data_uris)
            return self._parse_vision_output(raw_text)
        except Exception as exc:  # pragma: no cover - defensive branch
            record_llm_failure(exc)
            LOGGER.warning(
                "Vision analysis failed: %s [%s]",
                exc,
                llm_failure_context(exc),
            )

        return self._unknown_output()

    def load_images_as_data_uris(self, image_urls: list[str]) -> list[str]:
        """Download images for other model calls, skipping failures."""
        return [
            data_uri
            for image_url in image_urls
            if (data_uri := self._download_as_data_uri(image_url))
        ]

    def analyze_styles(self, image_urls: list[str]) -> list[StyleTag]:
        """Return only style tags for legacy callers."""
        return self.analyze_images(image_urls).style_tags

    def _download_as_data_uri(self, image_url: str) -> str | None:
        """Download an image URL and convert it into a data URI."""
        try:
            normalized_url = str(self._url_adapter.validate_python(image_url))
        except ValidationError:
            LOGGER.warning("Invalid image URL skipped: %s", image_url)
            return None

        try:
            response = httpx.get(
                normalized_url,
                timeout=self._timeout_seconds,
                follow_redirects=True,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            LOGGER.warning("Unable to fetch image %s: %s", normalized_url, exc)
            return None

        if not response.content:
            LOGGER.warning("Image URL returned empty data: %s", normalized_url)
            return None

        content_type = self._extract_content_type(
            response.headers.get("content-type", "")
        )
        encoded = base64.b64encode(response.content).decode("ascii")
        return f"data:{content_type};base64,{encoded}"

    def _extract_content_type(self, header_value: str) -> str:
        """Return a safe image MIME type from response headers."""
        normalized = header_value.split(";")[0].strip().lower()
        if normalized.startswith("image/"):
            return normalized
        return "image/jpeg"

    def _invoke_vision_model(self, data_uris: list[str]) -> str:
        """Call the vision-capable model and return text content."""
        content: list[dict[str, Any]] = []
        for data_uri in data_uris:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_uri},
                }
            )

        try:
            response = self._llm.invoke(
                [
                    SystemMessage(content=VISION_SYSTEM_PROMPT),
                    HumanMessage(content=content),
                ]
            )
            return self._coerce_content_to_text(response.content)
        except Exception as exc:
            raise AnalysisPipelineError("Vision model analysis failed.") from exc

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

    def _parse_vision_output(self, raw_text: str) -> TattooVisionOutput:
        """Parse and sanitize model output into approved style and color."""
        parsed = self._load_json_payload(raw_text)
        if isinstance(parsed, list):
            raw_tags = parsed
            raw_color = "unknown"
            raw_subjects: Any = []
            raw_description: Any = ""
        elif isinstance(parsed, dict):
            raw_tags = parsed.get("style_tags", [])
            raw_color = parsed.get("color_preference", "unknown")
            raw_subjects = parsed.get("design_subjects", [])
            raw_description = parsed.get("visual_description", "")
        else:
            return self._unknown_output()

        cleaned: list[str] = []
        if not isinstance(raw_tags, list):
            raw_tags = []
        for item in raw_tags:
            if not isinstance(item, str):
                continue
            normalized = item.strip().lower()
            if normalized in _STYLE_TAG_SET and normalized not in cleaned:
                cleaned.append(normalized)

        if not cleaned:
            cleaned = ["unknown"]
        if "unknown" in cleaned and len(cleaned) > 1:
            cleaned = [tag for tag in cleaned if tag != "unknown"]

        color = self._normalize_color(raw_color)
        if color == "unknown" and "black-and-grey" in cleaned:
            color = "black-and-grey"
        return TattooVisionOutput(
            style_tags=cast(list[StyleTag], cleaned),
            color_preference=color,
            design_subjects=self._normalize_subjects(raw_subjects),
            visual_description=self._normalize_description(raw_description),
        )

    def _normalize_subjects(self, raw_subjects: Any) -> list[str]:
        """Return unique, concise visual subjects from model output."""
        if not isinstance(raw_subjects, list):
            return []
        subjects: list[str] = []
        for item in raw_subjects:
            if not isinstance(item, str):
                continue
            normalized = " ".join(item.split()).strip(" .,-")[:80]
            if normalized and normalized.casefold() not in {
                value.casefold() for value in subjects
            }:
                subjects.append(normalized)
        return subjects[:10]

    def _normalize_description(self, raw_description: Any) -> str:
        """Return one bounded factual image description."""
        if not isinstance(raw_description, str):
            return ""
        return " ".join(raw_description.split())[:500]

    def _load_json_payload(self, raw_text: str) -> Any:
        """Load a JSON object, with support for fenced explanatory output."""
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", raw_text)
            if not match:
                return None
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None

    def _normalize_color(self, raw_color: Any) -> VisualColorPreference:
        """Normalize common model color labels to the strict taxonomy."""
        if not isinstance(raw_color, str):
            return "unknown"
        normalized = raw_color.strip().lower()
        aliases = {
            "black and grey": "black-and-grey",
            "black & grey": "black-and-grey",
            "black and gray": "black-and-grey",
            "black & gray": "black-and-grey",
            "full color": "color",
            "full colour": "color",
            "colour": "color",
            "colored": "color",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized == "black-and-grey":
            return "black-and-grey"
        if normalized == "color":
            return "color"
        return "unknown"

    def _unknown_output(self) -> TattooVisionOutput:
        """Return the safe result used when vision evidence is unavailable."""
        return TattooVisionOutput(
            style_tags=["unknown"],
            color_preference="unknown",
            design_subjects=[],
            visual_description="",
        )
