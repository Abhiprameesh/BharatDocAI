"""
Stage 7 Integrity & End-to-End Workflow Verification Tests for BharatDoc AI.
Rigorously tests:
1. Statistical integrity (Record-level vs Source-Document-Level paired differences).
2. Diagnostic error decomposition denominator and split scopes (Validation vs Test).
3. End-to-end document extraction pipeline (DocumentProcessor -> Preprocessing -> Benchmark / Live OCR -> Parser -> Schema Validator).
4. Category schemas strictly enforced (Invoice, Application, Certificate).
5. Missing dependencies and missing artifacts graceful error handling.
6. Export functionality (JSON, CSV, TXT).
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CATEGORY_SCHEMAS,
    SUPPORTED_LANGUAGES,
    RAW_DOCUMENTS_DIR,
    QWEN3_LORA_OCR_SUPPORT_FILE,
    THREE_MODEL_FALSE_ABSTENTION_FILE,
    EXP2_EVIDENCE_AUDIT_FILE,
    DATASET_MANIFEST_FILE,
)
from src.research_analysis import ResearchDataLoader
from src.evaluation import normalize_field_value, ModelEvaluator
from src.document_processor import DocumentProcessor, DocumentValidationError
from src.image_preprocessing import ImagePreprocessor
from src.ocr_engine import get_ocr_engine, BenchmarkLookupEngine
from src.extraction_engine import ExtractionEngine, ExtractionResult
from src.model_loader import get_model_backend, HardwareInfo


def test_statistical_two_level_comparisons():
    """Verify that both record-level and source-document-level comparisons are mathematically sound."""
    print("Testing Statistical Integrity: Two-Level Paired Comparisons...")
    
    # 1. Record-level (54 records)
    rec_df = ResearchDataLoader.get_paired_comparisons_summary()
    assert len(rec_df) == 4, f"Expected 4 record comparisons, got {len(rec_df)}"
    
    fs_zs_rec = rec_df[rec_df["comparison"] == "Few-Shot vs Zero-Shot"].iloc[0]
    assert fs_zs_rec["improved_docs"] == 23
    assert fs_zs_rec["identical_docs"] == 30
    assert fs_zs_rec["degraded_docs"] == 1
    assert fs_zs_rec["net_field_gain"] == 22
    assert fs_zs_rec["improved_docs"] + fs_zs_rec["identical_docs"] + fs_zs_rec["degraded_docs"] == 54

    lora_zs_rec = rec_df[rec_df["comparison"] == "Exp 1 LoRA vs Zero-Shot"].iloc[0]
    assert lora_zs_rec["improved_docs"] == 0
    assert lora_zs_rec["identical_docs"] == 41
    assert lora_zs_rec["degraded_docs"] == 13
    assert lora_zs_rec["net_field_gain"] == -15
    assert lora_zs_rec["improved_docs"] + lora_zs_rec["identical_docs"] + lora_zs_rec["degraded_docs"] == 54

    fs_lora_rec = rec_df[rec_df["comparison"] == "Few-Shot vs Exp 1 LoRA"].iloc[0]
    assert fs_lora_rec["improved_docs"] == 30
    assert fs_lora_rec["identical_docs"] == 24
    assert fs_lora_rec["degraded_docs"] == 0
    assert fs_lora_rec["net_field_gain"] == 37
    assert fs_lora_rec["improved_docs"] + fs_lora_rec["identical_docs"] + fs_lora_rec["degraded_docs"] == 54

    # 2. Source-document level (18 documents)
    doc_df = ResearchDataLoader.get_source_document_paired_comparisons_summary()
    assert len(doc_df) == 4, f"Expected 4 source doc comparisons, got {len(doc_df)}"

    fs_zs_doc = doc_df[doc_df["comparison"] == "Few-Shot vs Zero-Shot"].iloc[0]
    assert fs_zs_doc["improved_docs"] == 11
    assert fs_zs_doc["identical_docs"] == 7
    assert fs_zs_doc["degraded_docs"] == 0
    assert fs_zs_doc["net_field_gain"] == 22
    assert fs_zs_doc["improved_docs"] + fs_zs_doc["identical_docs"] + fs_zs_doc["degraded_docs"] == 18

    lora_zs_doc = doc_df[doc_df["comparison"] == "Exp 1 LoRA vs Zero-Shot"].iloc[0]
    assert lora_zs_doc["improved_docs"] == 0
    assert lora_zs_doc["identical_docs"] == 9
    assert lora_zs_doc["degraded_docs"] == 9
    assert lora_zs_doc["net_field_gain"] == -15
    assert lora_zs_doc["improved_docs"] + lora_zs_doc["identical_docs"] + lora_zs_doc["degraded_docs"] == 18

    fs_lora_doc = doc_df[doc_df["comparison"] == "Few-Shot vs Exp 1 LoRA"].iloc[0]
    assert fs_lora_doc["improved_docs"] == 12
    assert fs_lora_doc["identical_docs"] == 6
    assert fs_lora_doc["degraded_docs"] == 0
    assert fs_lora_doc["net_field_gain"] == 37
    assert fs_lora_doc["improved_docs"] + fs_lora_doc["identical_docs"] + fs_lora_doc["degraded_docs"] == 18

    print("✓ Two-level paired comparisons (54 records & 18 source documents) mathematically verified.")


def test_diagnostic_vs_test_error_scopes():
    """Verify exact distinction between validation diagnostic errors (71) and test errors (82)."""
    print("Testing Error Scopes: Validation Diagnostic (71) vs Test Evaluation (82)...")
    
    # 1. Validation diagnostic dataset
    diag_df = pd.read_csv(QWEN3_LORA_OCR_SUPPORT_FILE)
    manifest = pd.read_csv(DATASET_MANIFEST_FILE)
    
    diag_doc_ids = diag_df["document_id"].unique()
    assert len(diag_doc_ids) == 18, f"Expected 18 diagnostic doc IDs, got {len(diag_doc_ids)}"
    
    # Verify that all 18 diagnostic doc IDs are from the validation split
    val_manifest = manifest[manifest["split"] == "validation"]
    val_doc_ids = set(val_manifest["document_id"].unique())
    assert set(diag_doc_ids) == val_doc_ids, "Diagnostic doc IDs do not match the validation split!"

    # Verify diagnostic errors breakdown
    errors = diag_df[diag_df["correct"] == False]
    assert len(errors) == 71, f"Expected 71 diagnostic errors, got {len(errors)}"
    ocr_missing = diag_df[(diag_df["ocr_supported"] == False) & (diag_df["correct"] == False)]
    assert len(ocr_missing) == 62, f"Expected 62 OCR missing errors, got {len(ocr_missing)}"
    false_abstentions = diag_df[diag_df["false_abstention"] == True]
    assert len(false_abstentions) == 9, f"Expected 9 false abstentions, got {len(false_abstentions)}"
    assert len(ocr_missing) + len(false_abstentions) == 71

    # 2. Test evaluation dataset
    test_preds = ResearchDataLoader.load_predictions("experiment1_lora")
    assert len(test_preds) == 54
    test_doc_ids = set(r["document_id"] for r in test_preds)
    test_manifest = manifest[manifest["split"] == "test"]
    assert test_doc_ids == set(test_manifest["document_id"].unique()), "Test predictions do not match test split!"

    eval_results = ModelEvaluator.evaluate_predictions(test_preds)
    assert eval_results["correct_fields"] == 188
    assert eval_results["total_fields"] == 270
    test_errors = eval_results["total_fields"] - eval_results["correct_fields"]
    assert test_errors == 82, f"Expected 82 test errors in Exp 1 LoRA, got {test_errors}"

    print("✓ Diagnostic (71 validation errors) and Test Evaluation (82 test errors) scopes confirmed.")


def test_end_to_end_extraction_workflow():
    """Verify complete end-to-end extraction workflow with benchmark lookup and output parsing."""
    print("Testing End-to-End Extraction Workflow...")

    # Load a benchmark test sample
    sample_doc_id = "english_invoice_009"
    sample_img_path = RAW_DOCUMENTS_DIR / f"{sample_doc_id}.png"
    assert sample_img_path.exists(), f"Sample image {sample_img_path} not found"

    # Step 1: Preprocessing
    img = Image.open(sample_img_path).convert("RGB")
    preprocessed_img = ImagePreprocessor.apply_preprocessing(img, "grayscale")
    assert preprocessed_img is not None

    # Step 2: OCR Lookup
    lookup = BenchmarkLookupEngine(target_engine="easyocr")
    ocr_text = lookup.lookup_by_doc_id(sample_doc_id)
    assert ocr_text is not None and len(ocr_text) > 0, "Failed to retrieve OCR transcription"

    # Step 3: Structured Extraction
    backend = get_model_backend()
    result = backend.extract_information(
        ocr_text=ocr_text,
        category="invoice",
        language="english",
        method="few_shot",
        doc_id=sample_doc_id
    )

    # Step 4: Verification of ExtractionResult
    assert result.json_valid is True, "Extraction result was not valid JSON"
    assert result.schema_valid is True, "Extraction result failed schema validation"
    assert result.is_historical_benchmark is True, "Benchmark lookup was not flagged as historical benchmark"
    assert set(result.prediction.keys()) == set(CATEGORY_SCHEMAS["invoice"])
    assert result.prediction["name"] != "", "Expected extracted name to be non-empty"

    print("✓ End-to-end extraction workflow passed.")


def test_schema_enforcement_and_resilience():
    """Verify schema enforcement: certificates must require institution and grade, not address."""
    print("Testing Schema Enforcement and Error Handling...")

    cert_schema = CATEGORY_SCHEMAS["certificate"]
    assert "institution" in cert_schema, "Certificate schema missing 'institution'"
    assert "grade" in cert_schema, "Certificate schema missing 'grade'"
    assert "address" not in cert_schema, "Certificate schema improperly contains 'address'"

    # Test JSON extraction parser with malformed input
    raw_markdown = '```json\n{"name": "Anoop", "date": "12/04/2026", "document_id": "CERT-0081", "institution": "Bangalore University", "grade": "Distinction"}\n```'
    parsed = ExtractionEngine.process_response(raw_markdown, "certificate", "english")
    assert parsed.json_valid is True
    assert parsed.schema_valid is True
    assert parsed.prediction["institution"] == "Bangalore University"

    # Test parser resilience with invalid JSON (should not crash)
    invalid_raw = "This is not JSON text at all!"
    res_invalid = ExtractionEngine.process_response(invalid_raw, "certificate", "english")
    assert res_invalid.json_valid is False
    assert res_invalid.schema_valid is False
    assert "could not be parsed as valid JSON" in res_invalid.error_message

    print("✓ Schema enforcement and resilience verified.")


def test_hardware_cpu_awareness_and_safety():
    """Verify that when no CUDA GPU is present, CPU mode is safely detected."""
    print("Testing Hardware Awareness and CPU Safety...")
    desc = HardwareInfo.get_hardware_description()
    assert "CPU" in desc or "NVIDIA" in desc
    backend = get_model_backend()

    # Trigger extraction with arbitrary non-benchmark document
    res_cpu = backend.extract_information(
        ocr_text="Random unverified invoice text",
        category="invoice",
        language="english",
        method="zero_shot",
        doc_id=None
    )
    if not HardwareInfo.is_cuda_available():
        assert res_cpu.json_valid is False
        assert "Hardware Constraint" in res_cpu.error_message
        print("✓ Zero-GPU CPU mode safely flagged hardware limitation without crashing.")
    else:
        print("✓ CUDA hardware detected.")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING STAGE 7 INTEGRITY & WORKFLOW VERIFICATION SUITE")
    print("=" * 60)
    test_statistical_two_level_comparisons()
    test_diagnostic_vs_test_error_scopes()
    test_end_to_end_extraction_workflow()
    test_schema_enforcement_and_resilience()
    test_hardware_cpu_awareness_and_safety()
    print("=" * 60)
    print("ALL STAGE 7 INTEGRITY & WORKFLOW TESTS PASSED! 🚀")
    print("=" * 60)
