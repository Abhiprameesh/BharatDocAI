"""
Stage 5 Evaluation & Comparison Unit Tests
Validates:
1. Text normalization: NFC unicode, zero-width characters, visual separators.
2. Field-level exact match & document-level exact match calculation.
3. Master four-way evaluation recalculation on all 54 test records.
4. Granular breakdowns (Language, Category, OCR Engine, Individual Fields).
5. Paired model difference calculations.
"""

import sys
from pathlib import Path
import pandas as pd

# Ensure project root is in python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation import normalize_field_value, compute_record_metrics, ModelEvaluator
from src.research_analysis import ResearchDataLoader

def test_normalization_rules():
    """Verify research normalization rules across data types."""
    # Date formatting separators (/ \ - : . , | whitespace)
    assert normalize_field_value("01/01/2026", "date") == "01012026"
    assert normalize_field_value("01 - 01 - 2026", "date") == "01012026"
    assert normalize_field_value("01 | 01 | 2026", "date") == "01012026"
    
    # Phone number separators
    assert normalize_field_value("90000 01507", "phone") == "9000001507"
    assert normalize_field_value("90000-01507", "phone") == "9000001507"
    assert normalize_field_value("90000:01507", "phone") == "9000001507"
    
    # Amount separators
    assert normalize_field_value("1,637", "amount") == "1637"
    assert normalize_field_value(" 1637.00 ", "amount") == "163700"
    
    # Document ID
    assert normalize_field_value("INV-0001", "document_id") == "inv0001"
    assert normalize_field_value("inv 0001", "document_id") == "inv0001"
    
    # Text names (whitespace collapse & casefolding)
    assert normalize_field_value("  Kiran   Rao  ", "name") == "kiran rao"
    assert normalize_field_value("Sneha\u200cNair", "name") == "snehanair"  # ZWNJ stripped
    print("✓ Normalization rules verified across dates, phones, amounts, and text.")

def test_single_record_metrics():
    """Verify record metric computation for partial, full, and invalid predictions."""
    gt = {"name": "Alice", "date": "01/01/2026", "document_id": "INV-1", "address": "Bengaluru", "amount": "100"}
    
    # Complete match
    pred_perfect = {"name": "alice", "date": "01-01-2026", "document_id": "inv-1", "address": "bengaluru", "amount": "100"}
    c, t, doc_match, j_valid, s_valid = compute_record_metrics(gt, pred_perfect, "invoice")
    assert c == 5 and t == 5 and doc_match is True and j_valid is True and s_valid is True
    
    # 4/5 match (address mismatch)
    pred_partial = {"name": "alice", "date": "01-01-2026", "document_id": "inv-1", "address": "Delhi", "amount": "100"}
    c, t, doc_match, j_valid, s_valid = compute_record_metrics(gt, pred_partial, "invoice")
    assert c == 4 and t == 5 and doc_match is False and s_valid is True
    
    # Missing schema key
    pred_missing = {"name": "alice", "date": "01/01/2026"}
    c, t, doc_match, j_valid, s_valid = compute_record_metrics(gt, pred_missing, "invoice")
    assert s_valid is False and doc_match is False
    print("✓ Single-record metric computation verified.")

def test_master_four_way_evaluation():
    """Verify recalculation of four-way evaluation metrics matches research values."""
    eval_df = ResearchDataLoader.evaluate_four_way_comparison()
    assert len(eval_df) == 4
    
    metrics_by_model = dict(zip(eval_df["Model / Method"], eval_df.to_dict(orient="records")))
    
    # Zero-Shot Baseline
    zs = metrics_by_model["Zero-Shot Qwen3-4B"]
    assert zs["Field-Level Exact Match (%)"] == 75.19
    assert zs["Document-Level Exact Match (%)"] == 35.19
    assert zs["JSON Validity (%)"] == 100.0
    assert zs["Schema Validity (%)"] == 100.0
    assert zs["Correct Fields"] == "203/270"
    assert zs["Exact Documents"] == "19/54"
    
    # Few-Shot (Winner)
    fs = metrics_by_model["Few-Shot Qwen3-4B"]
    assert fs["Field-Level Exact Match (%)"] == 83.33
    assert fs["Document-Level Exact Match (%)"] == 61.11
    assert fs["Correct Fields"] == "225/270"
    assert fs["Exact Documents"] == "33/54"
    
    # Experiment 1 LoRA
    e1 = metrics_by_model["Experiment 1 LoRA"]
    assert e1["Field-Level Exact Match (%)"] == 69.63
    assert e1["Document-Level Exact Match (%)"] == 31.48
    assert e1["Correct Fields"] == "188/270"
    assert e1["Exact Documents"] == "17/54"
    
    # Experiment 2 LoRA (Provisional)
    e2 = metrics_by_model["Experiment 2 LoRA (Provisional)"]
    assert e2["Field-Level Exact Match (%)"] == 75.19
    assert e2["Document-Level Exact Match (%)"] == 35.19
    assert e2["Correct Fields"] == "203/270"
    assert e2["Exact Documents"] == "19/54"
    
    print("✓ Master four-way evaluation metrics 100% verified against research ground truth.")

def test_granular_breakdowns():
    """Verify Language, Category, OCR Engine, and Field breakdowns compute properly."""
    lang_df = ResearchDataLoader.get_language_breakdown()
    assert not lang_df.empty
    assert set(lang_df["Language"].str.lower()) == {"english", "hindi", "kannada"}
    
    cat_df = ResearchDataLoader.get_category_breakdown()
    assert not cat_df.empty
    assert set(cat_df["Category"].str.lower()) == {"invoice", "application", "certificate"}
    
    eng_df = ResearchDataLoader.get_engine_breakdown()
    assert not eng_df.empty
    assert set(eng_df["Ocr_Engine"].str.lower()) == {"easyocr", "paddleocr", "tesseract"}
    
    field_df = ResearchDataLoader.get_field_breakdown()
    assert not field_df.empty
    assert len(field_df["Field"].unique()) == 8  # name, date, document_id, address, amount, phone, institution, grade
    
    paired_df = ResearchDataLoader.get_paired_comparisons_summary()
    assert not paired_df.empty
    assert len(paired_df) == 4
    print("✓ Granular breakdowns (Language, Category, Engine, Field, Paired) verified.")

if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING BHARATDOC AI EVALUATION UNIT TESTS")
    print("=" * 60)
    test_normalization_rules()
    test_single_record_metrics()
    test_master_four_way_evaluation()
    test_granular_breakdowns()
    print("=" * 60)
    print("ALL EVALUATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
