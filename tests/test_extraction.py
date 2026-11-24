"""
Stage 4 Extraction Unit Tests
Validates:
1. Category schema enforcement (specifically 'institution' in certificate).
2. Prompt construction for zero-shot and few-shot (9 clean training demonstrations).
3. JSON parsing: Markdown code fences, surrounding whitespace, syntax error recovery.
4. Schema validation: Missing fields, unexpected fields, type normalization to string.
5. Model backend: Adapter path verification, hardware detection, historical benchmark lookup.
6. Mocked end-to-end extraction pipeline execution.
"""

import sys
import json
from pathlib import Path

# Ensure project root is in python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CATEGORY_SCHEMAS, EXP1_ADAPTER_DIR, EXP2_ADAPTER_DIR
from src.prompt_templates import (
    EXTRACTION_SYSTEM_PROMPT,
    NINE_FEW_SHOT_DEMOS,
    get_few_shot_demonstration,
    build_user_prompt,
)
from src.extraction_engine import ExtractionEngine, ExtractionResult
from src.model_loader import get_model_backend, HardwareInfo

def test_category_schemas():
    """Verify all 3 category schemas match original research annotations."""
    assert CATEGORY_SCHEMAS["invoice"] == ["name", "date", "document_id", "address", "amount"]
    assert CATEGORY_SCHEMAS["application"] == ["name", "date", "document_id", "address", "phone"]
    assert CATEGORY_SCHEMAS["certificate"] == ["name", "date", "document_id", "institution", "grade"]
    assert "institution" in CATEGORY_SCHEMAS["certificate"]
    assert "address" not in CATEGORY_SCHEMAS["certificate"]
    print("✓ Category schemas strictly verified.")

def test_prompt_construction_and_few_shot_demos():
    """Verify system prompt and few-shot demonstration alignment."""
    assert "You are an information extraction system" in EXTRACTION_SYSTEM_PROMPT
    assert len(NINE_FEW_SHOT_DEMOS) == 9
    
    # Test few-shot demonstration retrieval
    demo_hi_inv = get_few_shot_demonstration("hindi", "invoice")
    assert demo_hi_inv["document_id"] == "hindi_invoice_031"
    assert demo_hi_inv["ground_truth"]["amount"] == "5747"
    
    demo_kn_cert = get_few_shot_demonstration("kannada", "certificate")
    assert demo_kn_cert["document_id"] == "kannada_certificate_081"
    assert demo_kn_cert["ground_truth"]["institution"] == "ರಾಷ್ಟ್ರೀಯ ತಂತ್ರಜ್ಞಾನ ಸಂಸ್ಥೆ"
    
    # Test prompt generation
    prompt_zero = build_user_prompt("SAMPLE OCR", "invoice", "english", few_shot=False)
    assert "SAMPLE OCR" in prompt_zero
    assert "Extract these fields: name, date, document_id, address, amount" in prompt_zero
    assert "Here are examples" not in prompt_zero
    
    prompt_few = build_user_prompt("SAMPLE OCR", "invoice", "english", few_shot=True)
    assert "Here are examples of correct information extraction:" in prompt_few
    assert "english_invoice_001" not in prompt_few  # Check that text is properly formatted
    assert "INV-0001" in prompt_few
    print("✓ Prompt templates and few-shot demonstrations verified.")

def test_json_parsing_and_markdown_fences():
    """Verify JSON parser handles code blocks, whitespace, and formatting defects."""
    # Standard clean JSON
    clean = '{"name": "Alice", "date": "01/01/2026", "document_id": "INV-1", "address": "Delhi", "amount": "100"}'
    res1 = ExtractionEngine.parse_model_json(clean)
    assert res1 is not None and res1["name"] == "Alice"
    
    # Surrounding markdown fences
    fenced = '```json\n{"name": "Bob", "date": "02/02/2026", "document_id": "INV-2", "address": "Pune", "amount": "200"}\n```'
    res2 = ExtractionEngine.parse_model_json(fenced)
    assert res2 is not None and res2["name"] == "Bob"
    
    # Preamble text before JSON
    preamble = 'Here is the extracted information:\n\n{"name": "Charlie", "date": "03/03/2026", "document_id": "INV-3", "address": "Goa", "amount": "300"}\nHope this helps!'
    res3 = ExtractionEngine.parse_model_json(preamble)
    assert res3 is not None and res3["name"] == "Charlie"
    
    # Invalid JSON
    invalid = 'Not a json document at all.'
    res4 = ExtractionEngine.parse_model_json(invalid)
    assert res4 is None
    print("✓ JSON parsing & markdown code fence handling verified.")

def test_schema_validation_missing_and_unexpected_fields():
    """Verify schema validator checks keys, missing fields, and type conversions."""
    # Complete valid invoice
    valid_dict = {
        "name": "David",
        "date": "04/04/2026",
        "document_id": "INV-4",
        "address": "Chennai",
        "amount": 4500  # Number should be converted to string
    }
    sanitized, valid, missing, unexpected = ExtractionEngine.validate_schema(valid_dict, "invoice")
    assert valid is True
    assert sanitized["amount"] == "4500"
    assert len(missing) == 0 and len(unexpected) == 0
    
    # Missing field
    missing_dict = {
        "name": "Eve",
        "date": "05/05/2026",
        "document_id": "INV-5",
        "address": "Kochi"
        # 'amount' is absent
    }
    sanitized, valid, missing, unexpected = ExtractionEngine.validate_schema(missing_dict, "invoice")
    assert valid is False
    assert "amount" in missing
    assert sanitized["amount"] == ""  # Safely populated with empty string
    
    # Unexpected field
    extra_dict = {
        "name": "Frank",
        "date": "06/06/2026",
        "document_id": "CERT-1",
        "institution": "IIT",
        "grade": "A+",
        "extra_info": "Should not be here"
    }
    sanitized, valid, missing, unexpected = ExtractionEngine.validate_schema(extra_dict, "certificate")
    assert valid is False
    assert "extra_info" in unexpected
    print("✓ Schema validation (missing, unexpected, type normalization) verified.")

def test_model_backend_and_adapters():
    """Verify backend adapter paths and benchmark lookup mode."""
    backend = get_model_backend()
    exp1_ok, exp2_ok = backend.verify_adapter_paths()
    assert exp1_ok is True, f"Experiment 1 adapter missing at {EXP1_ADAPTER_DIR}"
    assert exp2_ok is True, f"Experiment 2 adapter missing at {EXP2_ADAPTER_DIR}"
    
    # Test benchmark retrieval for test record
    pred = backend.get_benchmark_prediction("zero_shot", "english_invoice_009")
    assert pred is not None
    assert "ground_truth" in pred
    assert pred["prediction"]["name"] == "Ananya Joshi"
    
    # Test full extract_information pipeline on test document
    res = backend.extract_information(
        ocr_text="DUMMY",
        category="invoice",
        language="english",
        method="zero_shot",
        doc_id="english_invoice_009"
    )
    assert res.json_valid is True
    assert res.schema_valid is True
    assert res.prediction["name"] == "Ananya Joshi"
    assert res.prediction["amount"] == "2733"
    print("✓ Model backend, adapter verification, and benchmark lookup verified.")

def test_hardware_awareness():
    """Verify hardware detection communicates status without crashing."""
    desc = HardwareInfo.get_hardware_description()
    assert isinstance(desc, str) and len(desc) > 0
    print(f"✓ Hardware environment detected: {desc}")

if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING BHARATDOC AI EXTRACTION UNIT TESTS")
    print("=" * 60)
    test_category_schemas()
    test_prompt_construction_and_few_shot_demos()
    test_json_parsing_and_markdown_fences()
    test_schema_validation_missing_and_unexpected_fields()
    test_model_backend_and_adapters()
    test_hardware_awareness()
    print("=" * 60)
    print("ALL EXTRACTION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
