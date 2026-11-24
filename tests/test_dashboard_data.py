"""
Unit tests for Stage 6 Research Dashboard data loading and analytics.
Verifies that all metrics, tables, training trajectories, and error analysis artifacts
load properly without hardcoded numbers and with exact scientific consistency.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research_analysis import ResearchDataLoader
from src.config import (
    EXP1_TRAINING_HISTORY_FILE,
    EXP2_TRAINING_HISTORY_FILE,
    BOOTSTRAP_COMPARISONS_FILE,
    DOCUMENT_CIs_FILE,
    QWEN3_LORA_OCR_SUPPORT_FILE,
    THREE_MODEL_FALSE_ABSTENTION_FILE,
    EXP2_EVIDENCE_AUDIT_FILE,
    QWEN3_LORA_FIELD_ERROR_FILE,
)


def test_dataset_summary_metrics():
    """Verify dataset summary numbers clearly distinguish 90 source docs from 1,440 images."""
    s = ResearchDataLoader.get_dataset_summary_metrics()
    assert s["total_unique_source_docs"] == 90, f"Expected 90 unique source docs, got {s['total_unique_source_docs']}"
    assert s["clean_images"] == 90, f"Expected 90 clean images, got {s['clean_images']}"
    assert s["degraded_variants"] == 1350, f"Expected 1,350 degraded variants, got {s['degraded_variants']}"
    assert s["total_image_records"] == 1440, f"Expected 1,440 total image records, got {s['total_image_records']}"
    assert s["train_docs"] == 54, f"Expected 54 train docs, got {s['train_docs']}"
    assert s["val_docs"] == 18, f"Expected 18 val docs, got {s['val_docs']}"
    assert s["test_docs"] == 18, f"Expected 18 test docs, got {s['test_docs']}"
    print("[PASS] Dataset summary metrics verified.")


def test_ocr_benchmark_tables():
    """Verify OCR overall, language, corruption, and severity tables."""
    overall = ResearchDataLoader.load_ocr_overall_metrics()
    assert not overall.empty, "OCR overall metrics table is empty"
    assert "EasyOCR" in overall["engine"].values
    assert "PaddleOCR" in overall["engine"].values
    assert "Tesseract" in overall["engine"].values

    # Check EasyOCR won overall CER
    easy_cer = overall.loc[overall["engine"] == "EasyOCR", "CER"].values[0]
    assert round(easy_cer, 4) == 0.1557, f"Expected EasyOCR CER ~0.1557, got {easy_cer}"

    lang_df = ResearchDataLoader.load_ocr_language_metrics()
    assert not lang_df.empty, "OCR language metrics table is empty"
    assert set(lang_df["language"].unique()) == {"english", "hindi", "kannada"}

    corr_df = ResearchDataLoader.load_ocr_corruption_metrics()
    assert not corr_df.empty, "OCR corruption metrics table is empty"

    sev_df = ResearchDataLoader.load_ocr_severity_metrics()
    assert not sev_df.empty, "OCR severity metrics table is empty"
    print("[PASS] OCR benchmark tables verified.")


def test_bootstrap_and_statistical_tables():
    """Verify document-level bootstrap comparisons and engine confidence intervals."""
    boot_df = ResearchDataLoader.load_bootstrap_comparisons()
    assert not boot_df.empty, "Bootstrap comparisons table is empty"
    assert len(boot_df) == 24, f"Expected 24 bootstrap comparison rows, got {len(boot_df)}"
    assert "difference" in boot_df.columns
    assert "CI_lower" in boot_df.columns
    assert "CI_upper" in boot_df.columns
    assert "interpretation" in boot_df.columns

    cis_df = ResearchDataLoader.load_document_level_cis()
    assert not cis_df.empty, "Document CIs table is empty"
    assert len(cis_df) == 24, f"Expected 24 document CIs, got {len(cis_df)}"
    print("[PASS] Bootstrap and statistical confidence intervals verified.")


def test_llm_extraction_comparison():
    """Verify four-way evaluation table matches research results."""
    eval_df = ResearchDataLoader.evaluate_four_way_comparison()
    assert len(eval_df) == 4, f"Expected 4 models in evaluation table, got {len(eval_df)}"

    few_shot_acc = eval_df.loc[eval_df["Model / Method"] == "Few-Shot Qwen3-4B", "Field-Level Exact Match (%)"].values[0]
    zero_shot_acc = eval_df.loc[eval_df["Model / Method"] == "Zero-Shot Qwen3-4B", "Field-Level Exact Match (%)"].values[0]
    lora_acc = eval_df.loc[eval_df["Model / Method"] == "Experiment 1 LoRA", "Field-Level Exact Match (%)"].values[0]

    assert few_shot_acc == 83.33, f"Expected Few-Shot 83.33%, got {few_shot_acc}"
    assert zero_shot_acc == 75.19, f"Expected Zero-Shot 75.19%, got {zero_shot_acc}"
    assert lora_acc == 69.63, f"Expected Exp 1 LoRA 69.63%, got {lora_acc}"
    print("[PASS] LLM four-way extraction comparison verified.")


def test_paired_document_differences():
    """Verify paired comparisons summary directly addresses central research hypothesis."""
    paired_df = ResearchDataLoader.get_paired_comparisons_summary()
    assert len(paired_df) == 4, f"Expected 4 paired comparisons, got {len(paired_df)}"
    assert "comparison" in paired_df.columns
    assert "improved_docs" in paired_df.columns
    assert "degraded_docs" in paired_df.columns
    assert "identical_docs" in paired_df.columns

    # Check Few-Shot vs Exp 1 LoRA (Few-Shot improved 30 documents, degraded 0, tied 24)
    row_fs_lora = paired_df[paired_df["comparison"] == "Few-Shot vs Exp 1 LoRA"].iloc[0]
    assert row_fs_lora["improved_docs"] == 30
    assert row_fs_lora["identical_docs"] == 24
    assert row_fs_lora["degraded_docs"] == 0
    assert row_fs_lora["net_field_gain"] == 37  # Few-Shot gained 37 net correct fields over LoRA

    # Check Few-Shot vs Zero-Shot (Few-Shot improved 23 documents, degraded 1, tied 30)
    row_fs_zs = paired_df[paired_df["comparison"] == "Few-Shot vs Zero-Shot"].iloc[0]
    assert row_fs_zs["improved_docs"] == 23
    assert row_fs_zs["identical_docs"] == 30
    assert row_fs_zs["degraded_docs"] == 1
    assert row_fs_zs["net_field_gain"] == 22

    # Check Exp 1 LoRA vs Zero-Shot (LoRA improved 0, degraded 13, tied 41)
    row_lora_zs = paired_df[paired_df["comparison"] == "Exp 1 LoRA vs Zero-Shot"].iloc[0]
    assert row_lora_zs["improved_docs"] == 0
    assert row_lora_zs["identical_docs"] == 41
    assert row_lora_zs["degraded_docs"] == 13
    assert row_lora_zs["net_field_gain"] == -15

    # Check Source-Document-Level Paired Comparisons (18 unique source documents)
    paired_doc_df = ResearchDataLoader.get_source_document_paired_comparisons_summary()
    assert len(paired_doc_df) == 4, f"Expected 4 source-doc paired comparisons, got {len(paired_doc_df)}"
    row_fs_zs_doc = paired_doc_df[paired_doc_df["comparison"] == "Few-Shot vs Zero-Shot"].iloc[0]
    assert row_fs_zs_doc["improved_docs"] == 11
    assert row_fs_zs_doc["identical_docs"] == 7
    assert row_fs_zs_doc["degraded_docs"] == 0
    assert row_fs_zs_doc["net_field_gain"] == 22

    row_fs_lora_doc = paired_doc_df[paired_doc_df["comparison"] == "Few-Shot vs Exp 1 LoRA"].iloc[0]
    assert row_fs_lora_doc["improved_docs"] == 12
    assert row_fs_lora_doc["identical_docs"] == 6
    assert row_fs_lora_doc["degraded_docs"] == 0
    assert row_fs_lora_doc["net_field_gain"] == 37

    row_lora_zs_doc = paired_doc_df[paired_doc_df["comparison"] == "Exp 1 LoRA vs Zero-Shot"].iloc[0]
    assert row_lora_zs_doc["improved_docs"] == 0
    assert row_lora_zs_doc["identical_docs"] == 9
    assert row_lora_zs_doc["degraded_docs"] == 9
    assert row_lora_zs_doc["net_field_gain"] == -15
    print("[PASS] Paired document differences (both record and source-doc level) verified.")


def test_error_and_recoverability_diagnostics():
    """Verify OCR error propagation vs LLM extraction error diagnostics."""
    rec = ResearchDataLoader.get_ocr_recoverability_summary()
    assert rec["total_fields"] == 270, f"Expected 270 fields, got {rec['total_fields']}"
    assert rec["supported_fields"] == 208, f"Expected 208 supported fields, got {rec['supported_fields']}"
    assert rec["unsupported_fields"] == 62, f"Expected 62 unsupported fields, got {rec['unsupported_fields']}"
    assert rec["false_abstentions"] == 9, f"Expected 9 false abstentions, got {rec['false_abstentions']}"
    assert rec["ocr_error_share_pct"] == 87.32, f"Expected 87.32% OCR error share, got {rec['ocr_error_share_pct']}"

    # Verify false abstention comparison table
    fa_df = ResearchDataLoader.load_false_abstention_comparison()
    assert len(fa_df) == 9, f"Expected 9 false abstention rows, got {len(fa_df)}"
    # All 9 were correct in zero-shot and few-shot, and incorrect in LoRA
    assert (fa_df["zero_shot_correct"] == True).all()
    assert (fa_df["few_shot_correct"] == True).all()
    assert (fa_df["lora_correct"] == False).all()
    print("[PASS] Error and recoverability diagnostics verified.")


def test_training_histories():
    """Verify training loss curves and convergence diagnostics."""
    h1, h2 = ResearchDataLoader.load_training_histories()
    assert len(h1) == 5, f"Expected 5 epochs in Exp 1, got {len(h1)}"
    assert len(h2) == 5, f"Expected 5 epochs in Exp 2, got {len(h2)}"

    # Check Exp 1 overfitting signature: min val loss at epoch 3
    val_losses_e1 = [ep["validation_loss"] for ep in h1]
    min_epoch = val_losses_e1.index(min(val_losses_e1)) + 1
    assert min_epoch == 3, f"Expected minimum validation loss at epoch 3, found at epoch {min_epoch}"
    print("[PASS] Training histories and loss curves verified.")


if __name__ == "__main__":
    print("Running Stage 6 Research Dashboard Tests...")
    test_dataset_summary_metrics()
    test_ocr_benchmark_tables()
    test_bootstrap_and_statistical_tables()
    test_llm_extraction_comparison()
    test_paired_document_differences()
    test_error_and_recoverability_diagnostics()
    test_training_histories()
    print("\nALL STAGE 6 RESEARCH DASHBOARD TESTS PASSED SUCCESSFULLY! 🚀")
