"""
BharatDoc AI - Modular OCR Engine Interface
Provides abstraction for EasyOCR (default), Tesseract, and PaddleOCR.
Includes benchmark simulation fallback to inspect historical research recognitions.
"""

import abc
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from PIL import Image
import numpy as np

from src.config import (
    EASYOCR_RESULTS_FILE,
    PADDLEOCR_RESULTS_FILE,
    TESSERACT_CLEAN_FILE,
    TESSERACT_DEGRADED_FILE,
    ANNOTATIONS_FILE,
    OCR_ENGINES,
)

# Language code mappings verified from research notebooks
EASYOCR_LANG_MAP = {
    "english": ["en"],
    "hindi": ["hi", "en"],
    "kannada": ["kn", "en"],
}

TESSERACT_LANG_MAP = {
    "english": "eng",
    "hindi": "hin",
    "kannada": "kan",
}

PADDLEOCR_LANG_MAP = {
    "english": "en",
    "hindi": "hi",
    "kannada": "ka",
}

class BaseOCREngine(abc.ABC):
    """Abstract interface for document OCR engines."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Engine display name."""
        pass

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Whether the engine package/binary is installed and functional."""
        pass

    @abc.abstractmethod
    def extract_text(self, image: Image.Image, language: str) -> str:
        """Extract text from a PIL Image given a target language."""
        pass


class EasyOCREngine(BaseOCREngine):
    """EasyOCR engine implementation with reader caching."""

    def __init__(self):
        self._readers: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "EasyOCR"

    def is_available(self) -> bool:
        try:
            import easyocr
            return True
        except ImportError:
            return False

    def _get_reader(self, language: str):
        import easyocr
        import torch
        lang_codes = EASYOCR_LANG_MAP.get(language.lower(), ["en"])
        key = "-".join(lang_codes)
        if key not in self._readers:
            use_gpu = torch.cuda.is_available()
            self._readers[key] = easyocr.Reader(lang_codes, gpu=use_gpu, verbose=False)
        return self._readers[key]

    def extract_text(self, image: Image.Image, language: str) -> str:
        if not self.is_available():
            raise RuntimeError(
                "EasyOCR is not installed in this Python environment. "
                "Install it using: pip install easyocr torch"
            )
        reader = self._get_reader(language)
        np_image = np.array(image)
        results = reader.readtext(np_image, detail=0, paragraph=False)
        return "\n".join(results).strip()


class TesseractEngine(BaseOCREngine):
    """Tesseract OCR engine wrapper."""

    @property
    def name(self) -> str:
        return "Tesseract"

    def is_available(self) -> bool:
        try:
            import pytesseract
            # Check if tesseract binary can be called
            _ = pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def extract_text(self, image: Image.Image, language: str) -> str:
        if not self.is_available():
            raise RuntimeError(
                "Tesseract is not installed or not in PATH. "
                "Install Tesseract OCR and appropriate language packs (eng, hin, kan)."
            )
        import pytesseract
        lang_code = TESSERACT_LANG_MAP.get(language.lower(), "eng")
        text = pytesseract.image_to_string(image, lang=lang_code, config="--oem 3 --psm 3")
        return text.strip()


class PaddleOCREngine(BaseOCREngine):
    """PaddleOCR engine wrapper."""

    def __init__(self):
        self._readers: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "PaddleOCR"

    def is_available(self) -> bool:
        try:
            import paddleocr
            return True
        except ImportError:
            return False

    def _get_reader(self, language: str):
        from paddleocr import PaddleOCR
        lang_code = PADDLEOCR_LANG_MAP.get(language.lower(), "en")
        if lang_code not in self._readers:
            self._readers[lang_code] = PaddleOCR(
                lang=lang_code,
                device="cpu",
                enable_mkldnn=False,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        return self._readers[lang_code]

    def extract_text(self, image: Image.Image, language: str) -> str:
        if not self.is_available():
            raise RuntimeError(
                "PaddleOCR is not installed in this Python environment. "
                "Install it using: pip install paddlepaddle paddleocr"
            )
        reader = self._get_reader(language)
        np_image = np.array(image)
        results = reader.predict(np_image)
        extracted_lines = []
        for res in results:
            texts = res.get("rec_texts", [])
            if texts:
                extracted_lines.extend(texts)
        return "\n".join(extracted_lines).strip()


class BenchmarkLookupEngine(BaseOCREngine):
    """
    Retrieves the verified historical OCR recognition from the research archive
    for any synthetic benchmark document (001 - 090).
    Guarantees instant, 100% reproducible testing on CPU without requiring heavy OCR downloads.
    """

    def __init__(self, target_engine: str = "easyocr"):
        self.target_engine = target_engine.lower()
        self._records_cache: Optional[Dict[str, str]] = None

    @property
    def name(self) -> str:
        return f"Research Benchmark ({self.target_engine.title()})"

    def is_available(self) -> bool:
        return True

    def _load_cache(self):
        if self._records_cache is not None:
            return

        self._records_cache = {}
        target_file = None

        if self.target_engine == "easyocr":
            target_file = EASYOCR_RESULTS_FILE
        elif self.target_engine == "paddleocr":
            target_file = PADDLEOCR_RESULTS_FILE
        elif self.target_engine == "tesseract":
            target_file = TESSERACT_CLEAN_FILE

        if target_file and target_file.exists():
            with open(target_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        doc_id = item.get("document_id")
                        # Only take clean baseline or first match
                        if doc_id and doc_id not in self._records_cache:
                            self._records_cache[doc_id] = item.get("ocr_text", "")

    def lookup_by_doc_id(self, doc_id: str) -> Optional[str]:
        """Look up OCR text by synthetic document_id."""
        self._load_cache()
        return self._records_cache.get(doc_id)

    def extract_text(self, image: Image.Image, language: str) -> str:
        return "[Benchmark Lookup requires a synthetic document identifier]"


def get_ocr_engine(engine_name: str) -> BaseOCREngine:
    """Factory to get the requested OCR engine instance."""
    engine_name = engine_name.lower()
    if engine_name == "easyocr":
        return EasyOCREngine()
    elif engine_name == "tesseract":
        return TesseractEngine()
    elif engine_name == "paddleocr":
        return PaddleOCREngine()
    raise ValueError(f"Unknown OCR engine: '{engine_name}'. Supported: {list(OCR_ENGINES.keys())}")
