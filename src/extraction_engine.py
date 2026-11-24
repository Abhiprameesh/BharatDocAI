"""
BharatDoc AI - Extraction Engine Module
Handles JSON parsing, schema enforcement, validation, and structured extraction generation.
"""

import re
import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from src.config import CATEGORY_SCHEMAS

@dataclass
class ExtractionResult:
    """Container for parsed and validated structured extraction output."""
    category: str
    language: str
    prediction: Dict[str, str]
    raw_response: str
    json_valid: bool
    schema_valid: bool
    missing_fields: List[str] = field(default_factory=list)
    unexpected_fields: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    is_historical_benchmark: bool = False

class ExtractionEngine:
    """Robust output parser and schema validator."""

    @staticmethod
    def parse_model_json(text: str) -> Optional[Dict[str, Any]]:
        """
        Extract and parse JSON object from model raw response.
        Handles surrounding Markdown code blocks, preamble, and postscript text.
        """
        if not text or not isinstance(text, str):
            return None

        cleaned = text.strip()

        # Remove markdown code fences if present
        if "```" in cleaned:
            # Handle ```json ... ``` or bare ``` ... ```
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()

        # Locate outermost JSON object braces
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start == -1 or end == -1 or end < start:
            return None

        candidate = cleaned[start : end + 1]

        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
            return None
        except (json.JSONDecodeError, TypeError):
            # Attempt recovery from common JSON formatting issues (e.g. trailing commas)
            try:
                fixed_candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
                parsed = json.loads(fixed_candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
            return None

    @classmethod
    def validate_schema(
        cls, parsed_dict: Optional[Dict[str, Any]], category: str
    ) -> Tuple[Dict[str, str], bool, List[str], List[str]]:
        """
        Validate parsed dictionary against the required category schema.
        Returns:
            (sanitized_prediction_dict, is_schema_valid, missing_fields, unexpected_fields)
        """
        expected_fields = CATEGORY_SCHEMAS.get(category.lower(), [])
        
        if parsed_dict is None:
            # All fields missing
            return {f: "" for f in expected_fields}, False, list(expected_fields), []

        actual_keys = set(parsed_dict.keys())
        expected_set = set(expected_fields)

        missing = [f for f in expected_fields if f not in actual_keys]
        unexpected = [k for k in parsed_dict.keys() if k not in expected_set]

        # Sanitize values to strings
        sanitized = {}
        for f in expected_fields:
            raw_val = parsed_dict.get(f, "")
            if raw_val is None:
                sanitized[f] = ""
            elif isinstance(raw_val, (int, float, bool)):
                sanitized[f] = str(raw_val)
            else:
                sanitized[f] = str(raw_val).strip()

        # Schema is strictly valid only if all required fields are present and no unexpected fields
        is_schema_valid = (len(missing) == 0 and len(unexpected) == 0)

        return sanitized, is_schema_valid, missing, unexpected

    @classmethod
    def process_response(
        cls, raw_response: str, category: str, language: str
    ) -> ExtractionResult:
        """
        End-to-end processing of model response into ExtractionResult.
        """
        cat = category.lower()
        lang = language.lower()

        parsed = cls.parse_model_json(raw_response)
        json_valid = parsed is not None

        sanitized_pred, schema_valid, missing, unexpected = cls.validate_schema(parsed, cat)

        error_msg = None
        if not json_valid:
            error_msg = "Model output could not be parsed as valid JSON."
        elif not schema_valid:
            parts = []
            if missing:
                parts.append(f"Missing schema keys: {missing}")
            if unexpected:
                parts.append(f"Unexpected keys: {unexpected}")
            error_msg = "; ".join(parts)

        return ExtractionResult(
            category=cat,
            language=lang,
            prediction=sanitized_pred,
            raw_response=raw_response or "",
            json_valid=json_valid,
            schema_valid=schema_valid,
            missing_fields=missing,
            unexpected_fields=unexpected,
            error_message=error_msg,
        )
