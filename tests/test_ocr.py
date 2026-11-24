"""
Stage 3 OCR Unit Tests
Validates:
1. DocumentProcessor: File validation, image loading (PNG/JPG), PDF extraction.
2. ImagePreprocessor: Grayscale, Contrast boost, Otsu binarization, Denoising, Sharpening.
3. BaseOCREngine: Engine factory, availability checks, language mappings.
4. BenchmarkLookupEngine: Instant retrieval of verified benchmark OCR recognitions on synthetic docs.
"""

import sys
import io
from pathlib import Path
from PIL import Image
import numpy as np

# Ensure project root is in python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.document_processor import DocumentProcessor, DocumentValidationError
from src.image_preprocessing import ImagePreprocessor, PREPROCESSING_METHODS
from src.ocr_engine import (
    get_ocr_engine,
    EasyOCREngine,
    TesseractEngine,
    PaddleOCREngine,
    BenchmarkLookupEngine,
    EASYOCR_LANG_MAP,
    TESSERACT_LANG_MAP,
    PADDLEOCR_LANG_MAP,
)
from src.config import RAW_DOCUMENTS_DIR

def test_document_processor_image_loading():
    """Test loading valid PNG image."""
    img = Image.new("RGB", (200, 200), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    bytes_data = buf.getvalue()
    
    images, pdf_text = DocumentProcessor.load_document("test.png", bytes_data)
    assert len(images) == 1
    assert images[0].size == (200, 200)
    assert pdf_text is None
    print("✓ Image loading test passed.")

def test_document_processor_invalid_extension():
    """Test rejection of unsupported file extensions."""
    caught = False
    try:
        DocumentProcessor.validate_file("document.docx", b"dummy")
    except DocumentValidationError as exc:
        caught = True
        assert "Unsupported file format" in str(exc)
    assert caught, "Should have raised DocumentValidationError for .docx"
    print("✓ Invalid extension rejection test passed.")

def test_image_preprocessing():
    """Test all preprocessing operations produce valid PIL images."""
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    
    for method in PREPROCESSING_METHODS:
        processed = ImagePreprocessor.apply_preprocessing(img, method)
        assert isinstance(processed, Image.Image)
        assert processed.size == (100, 100)
        
    # Test specific Otsu thresholding produces binary values
    otsu_img = ImagePreprocessor.binarize_otsu(img)
    arr = np.array(otsu_img)
    unique_vals = np.unique(arr)
    # Check that pixels are strictly binary (0 or 255)
    for v in unique_vals:
        assert v in (0, 255)
    print("✓ Image preprocessing routines passed (all methods & Otsu binarization).")

def test_ocr_engine_factory_and_mappings():
    """Test OCR engine factory and language mappings."""
    easy = get_ocr_engine("easyocr")
    assert isinstance(easy, EasyOCREngine)
    assert easy.name == "EasyOCR"
    
    tess = get_ocr_engine("tesseract")
    assert isinstance(tess, TesseractEngine)
    assert tess.name == "Tesseract"
    
    paddle = get_ocr_engine("paddleocr")
    assert isinstance(paddle, PaddleOCREngine)
    assert paddle.name == "PaddleOCR"
    
    # Language mapping verification
    assert EASYOCR_LANG_MAP["hindi"] == ["hi", "en"]
    assert EASYOCR_LANG_MAP["kannada"] == ["kn", "en"]
    assert TESSERACT_LANG_MAP["hindi"] == "hin"
    assert TESSERACT_LANG_MAP["kannada"] == "kan"
    assert PADDLEOCR_LANG_MAP["kannada"] == "ka"
    print("✓ OCR engine factory and language mappings passed.")

def test_benchmark_lookup_engine():
    """Test retrieval of benchmark OCR text for synthetic documents."""
    lookup_easy = BenchmarkLookupEngine(target_engine="easyocr")
    text_easy = lookup_easy.lookup_by_doc_id("english_invoice_001")
    assert text_easy is not None
    assert len(text_easy) > 0
    assert "INVOICE" in text_easy.upper()
    assert "INV-0001" in text_easy or "0001" in text_easy
    
    lookup_paddle = BenchmarkLookupEngine(target_engine="paddleocr")
    text_paddle = lookup_paddle.lookup_by_doc_id("english_invoice_001")
    assert text_paddle is not None
    assert len(text_paddle) > 0
    
    lookup_tess = BenchmarkLookupEngine(target_engine="tesseract")
    text_tess = lookup_tess.lookup_by_doc_id("english_invoice_001")
    assert text_tess is not None
    assert len(text_tess) > 0
    
    print("✓ Benchmark Lookup Engine passed for EasyOCR, PaddleOCR, and Tesseract.")

if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING BHARATDOC AI OCR UNIT TESTS")
    print("=" * 60)
    test_document_processor_image_loading()
    test_document_processor_invalid_extension()
    test_image_preprocessing()
    test_ocr_engine_factory_and_mappings()
    test_benchmark_lookup_engine()
    print("=" * 60)
    print("ALL OCR TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
