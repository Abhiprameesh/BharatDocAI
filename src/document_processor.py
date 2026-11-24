"""
BharatDoc AI - Document Processor Module
Handles file validation, image loading (PNG, JPG, JPEG), and PDF page extraction.
"""

import io
from pathlib import Path
from typing import Tuple, List, Optional, Union
from PIL import Image
import pypdf

from src.config import ALLOWED_EXTENSIONS

MAX_FILE_SIZE_MB = 20
MAX_IMAGE_DIMENSION = 5000

class DocumentValidationError(Exception):
    """Custom exception for invalid document formats or sizes."""
    pass

class DocumentProcessor:
    """Handles loading, validation, and conversion of document files."""

    @staticmethod
    def validate_file(file_name: str, file_bytes: bytes) -> str:
        """
        Validate file extension and size.
        Returns the sanitized lowercase extension.
        """
        ext = Path(file_name).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise DocumentValidationError(
                f"Unsupported file format '{ext}'. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
            )

        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise DocumentValidationError(
                f"File size ({size_mb:.2f} MB) exceeds maximum limit of {MAX_FILE_SIZE_MB} MB."
            )

        return ext

    @classmethod
    def load_document(
        cls, file_name: str, file_bytes: bytes
    ) -> Tuple[List[Image.Image], Optional[str]]:
        """
        Loads document from bytes.
        Returns a tuple of (List[PIL.Image], Optional[extracted_pdf_text]).
        For single images, returns a 1-element list.
        For PDFs, extracts pages as images or text.
        """
        ext = cls.validate_file(file_name, file_bytes)

        if ext in [".png", ".jpg", ".jpeg"]:
            try:
                image = Image.open(io.BytesIO(file_bytes))
                # Verify image integrity
                image.verify()
                # Reopen since verify closes the stream
                image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

                # Dimension sanity check
                w, h = image.size
                if w > MAX_IMAGE_DIMENSION or h > MAX_IMAGE_DIMENSION:
                    image.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.Resampling.LANCZOS)

                return [image], None
            except Exception as e:
                raise DocumentValidationError(f"Invalid or corrupted image file: {str(e)}")

        elif ext == ".pdf":
            try:
                reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                total_pages = len(reader.pages)
                if total_pages == 0:
                    raise DocumentValidationError("The uploaded PDF has 0 pages.")

                extracted_text_pages = []
                extracted_images = []

                for page_idx, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    extracted_text_pages.append(f"--- Page {page_idx + 1} ---\n{text}")

                    # Attempt to extract embedded images if available
                    for img_file in page.images:
                        try:
                            img = Image.open(io.BytesIO(img_file.data)).convert("RGB")
                            extracted_images.append(img)
                        except Exception:
                            pass

                # If no embedded images found, create a placeholder canvas with page info
                if not extracted_images:
                    for page_idx in range(total_pages):
                        placeholder = Image.new("RGB", (800, 1000), color=(255, 255, 255))
                        extracted_images.append(placeholder)

                full_pdf_text = "\n\n".join(extracted_text_pages).strip()
                return extracted_images, full_pdf_text

            except Exception as e:
                raise DocumentValidationError(f"Failed to process PDF document: {str(e)}")

        raise DocumentValidationError(f"Unsupported extension: {ext}")
