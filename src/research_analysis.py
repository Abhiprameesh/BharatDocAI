"""
BharatDoc AI - Research Data Loader & Analysis Module
Loads precomputed research artifacts and computes dynamic evaluations via ModelEvaluator.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np

from src.config import (
    ANNOTATIONS_FILE,
    DATASET_MANIFEST_FILE,
    DEGRADATION_MANIFEST_FILE,
    OCR_BENCHMARK_SUMMARY_FILE,
    THREE_ENGINE_OVERALL_FILE,
    THREE_ENGINE_LANGUAGE_FILE,
    THREE_ENGINE_CORRUPTION_FILE,
    THREE_ENGINE_SEVERITY_FILE,
    BOOTSTRAP_COMPARISONS_FILE,
    DOCUMENT_CIs_FILE,
    ZERO_SHOT_PREDICTIONS_FILE,
    FEW_SHOT_PREDICTIONS_FILE,
    EXP1_LORA_PREDICTIONS_FILE,
    EXP2_LORA_PREDICTIONS_FILE,
    EXP1_TRAINING_HISTORY_FILE,
    EXP2_TRAINING_HISTORY_FILE,
    QWEN3_LORA_OCR_SUPPORT_FILE,
    THREE_MODEL_FALSE_ABSTENTION_FILE,
    EXP2_EVIDENCE_AUDIT_FILE,
    EXP2_AUTOMATED_EVIDENCE_FILE,
    QWEN3_LORA_FIELD_ERROR_FILE,
    CATEGORY_SCHEMAS,
)
from src.evaluation import normalize_field_value, ModelEvaluator, compute_record_metrics

class ResearchDataLoader:
    """Singleton-like loader for research datasets and metrics."""

    @staticmethod
    def load_dataset_manifest() -> pd.DataFrame:
        if DATASET_MANIFEST_FILE.exists():
            return pd.read_csv(DATASET_MANIFEST_FILE)
        return pd.DataFrame()

    @staticmethod
    def load_degradation_manifest() -> pd.DataFrame:
        if DEGRADATION_MANIFEST_FILE.exists():
            return pd.read_csv(DEGRADATION_MANIFEST_FILE)
        return pd.DataFrame()

    @staticmethod
    def load_ocr_overall_metrics() -> pd.DataFrame:
        if THREE_ENGINE_OVERALL_FILE.exists():
            return pd.read_csv(THREE_ENGINE_OVERALL_FILE)
        return pd.DataFrame()

    @staticmethod
    def load_ocr_language_metrics() -> pd.DataFrame:
        if THREE_ENGINE_LANGUAGE_FILE.exists():
            return pd.read_csv(THREE_ENGINE_LANGUAGE_FILE)
        return pd.DataFrame()

    @staticmethod
    def load_ocr_corruption_metrics() -> pd.DataFrame:
        if THREE_ENGINE_CORRUPTION_FILE.exists():
            return pd.read_csv(THREE_ENGINE_CORRUPTION_FILE)
        return pd.DataFrame()

    @staticmethod
    def load_ocr_severity_metrics() -> pd.DataFrame:
        if THREE_ENGINE_SEVERITY_FILE.exists():
            return pd.read_csv(THREE_ENGINE_SEVERITY_FILE)
        return pd.DataFrame()

    @staticmethod
    def load_bootstrap_comparisons() -> pd.DataFrame:
        if BOOTSTRAP_COMPARISONS_FILE.exists():
            return pd.read_csv(BOOTSTRAP_COMPARISONS_FILE)
        return pd.DataFrame()

    @staticmethod
    def load_document_level_cis() -> pd.DataFrame:
        if DOCUMENT_CIs_FILE.exists():
            return pd.read_csv(DOCUMENT_CIs_FILE)
        return pd.DataFrame()

    @staticmethod
    def load_ocr_support_diagnostic() -> pd.DataFrame:
        if QWEN3_LORA_OCR_SUPPORT_FILE.exists():
            return pd.read_csv(QWEN3_LORA_OCR_SUPPORT_FILE)
        return pd.DataFrame()

    @staticmethod
    def load_false_abstention_comparison() -> pd.DataFrame:
        if THREE_MODEL_FALSE_ABSTENTION_FILE.exists():
            return pd.read_csv(THREE_MODEL_FALSE_ABSTENTION_FILE)
        return pd.DataFrame()

    @staticmethod
    def load_evidence_audit() -> pd.DataFrame:
        if EXP2_EVIDENCE_AUDIT_FILE.exists():
            return pd.read_csv(EXP2_EVIDENCE_AUDIT_FILE)
        return pd.DataFrame()

    @staticmethod
    def load_automated_evidence_classification() -> pd.DataFrame:
        if EXP2_AUTOMATED_EVIDENCE_FILE.exists():
            return pd.read_csv(EXP2_AUTOMATED_EVIDENCE_FILE)
        return pd.DataFrame()

    @staticmethod
    def load_field_error_analysis() -> pd.DataFrame:
        if QWEN3_LORA_FIELD_ERROR_FILE.exists():
            return pd.read_csv(QWEN3_LORA_FIELD_ERROR_FILE)
        return pd.DataFrame()

    @classmethod
    def get_dataset_summary_metrics(cls) -> Dict[str, Any]:
        """Dataset summary with clear distinction between unique sources and degraded variants."""
        dataset_df = cls.load_dataset_manifest()
        degrad_df = cls.load_degradation_manifest()

        total_unique_docs = len(dataset_df) if not dataset_df.empty else 90
        splits = dataset_df["split"].value_counts().to_dict() if not dataset_df.empty else {"train": 54, "val": 18, "test": 18}
        languages = dataset_df["language"].value_counts().to_dict() if not dataset_df.empty else {"english": 30, "hindi": 30, "kannada": 30}
        categories = dataset_df["category"].value_counts().to_dict() if not dataset_df.empty else {"invoice": 30, "application": 30, "certificate": 30}

        total_images = len(degrad_df) if not degrad_df.empty else 1440
        clean_images = len(degrad_df[degrad_df["corruption"] == "clean"]) if not degrad_df.empty else 90
        degraded_variants = len(degrad_df[degrad_df["corruption"] != "clean"]) if not degrad_df.empty else 1350

        return {
            "total_unique_source_docs": total_unique_docs,
            "train_docs": splits.get("train", 54),
            "val_docs": splits.get("val", 18),
            "test_docs": splits.get("test", 18),
            "clean_images": clean_images,
            "degraded_variants": degraded_variants,
            "total_image_records": total_images,
            "languages": languages,
            "categories": categories,
            "splits": splits,
        }

    @classmethod
    def get_ocr_recoverability_summary(cls) -> Dict[str, Any]:
        """Calculates precise breakdown of OCR error propagation vs LLM extraction failures."""
        diag_df = cls.load_ocr_support_diagnostic()
        if diag_df.empty:
            return {}

        total_eval_fields = len(diag_df)
        supported_fields = int((diag_df["ocr_supported"] == True).sum())
        unsupported_fields = int((diag_df["ocr_supported"] == False).sum())

        total_errors = int((diag_df["correct"] == False).sum())
        total_correct = int((diag_df["correct"] == True).sum())

        ocr_missing_errors = int(((diag_df["ocr_supported"] == False) & (diag_df["correct"] == False)).sum())
        llm_recoverable_errors = int(((diag_df["ocr_supported"] == True) & (diag_df["correct"] == False)).sum())

        false_abstentions = int((diag_df["false_abstention"] == True).sum())
        unsupported_nonempty = int((diag_df["unsupported_nonempty"] == True).sum())

        return {
            "total_fields": total_eval_fields,
            "supported_fields": supported_fields,
            "unsupported_fields": unsupported_fields,
            "support_rate": round(supported_fields / total_eval_fields * 100, 2) if total_eval_fields else 0.0,
            "total_errors": total_errors,
            "ocr_missing_errors": ocr_missing_errors,
            "ocr_error_share_pct": round(ocr_missing_errors / total_errors * 100, 2) if total_errors else 0.0,
            "llm_error_share_pct": round(llm_recoverable_errors / total_errors * 100, 2) if total_errors else 0.0,
            "false_abstentions": false_abstentions,
            "unsupported_hallucinations": unsupported_nonempty,
        }

    @staticmethod
    def load_training_histories() -> Tuple[List[Dict], List[Dict]]:
        exp1_hist, exp2_hist = [], []
        if EXP1_TRAINING_HISTORY_FILE.exists():
            with open(EXP1_TRAINING_HISTORY_FILE, "r", encoding="utf-8") as f:
                exp1_hist = json.load(f)
        if EXP2_TRAINING_HISTORY_FILE.exists():
            with open(EXP2_TRAINING_HISTORY_FILE, "r", encoding="utf-8") as f:
                exp2_hist = json.load(f)
        return exp1_hist, exp2_hist

    @staticmethod
    def load_predictions(model_key: str) -> List[Dict[str, Any]]:
        mapping = {
            "zero_shot": ZERO_SHOT_PREDICTIONS_FILE,
            "few_shot": FEW_SHOT_PREDICTIONS_FILE,
            "experiment1_lora": EXP1_LORA_PREDICTIONS_FILE,
            "experiment2_lora": EXP2_LORA_PREDICTIONS_FILE,
        }
        path = mapping.get(model_key)
        if path and path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        return []

    @classmethod
    def load_all_predictions_map(cls) -> Dict[str, List[Dict[str, Any]]]:
        """Load dictionary mapping friendly model labels to prediction lists."""
        return {
            "Zero-Shot Qwen3-4B": cls.load_predictions("zero_shot"),
            "Few-Shot Qwen3-4B": cls.load_predictions("few_shot"),
            "Experiment 1 LoRA": cls.load_predictions("experiment1_lora"),
            "Experiment 2 LoRA (Provisional)": cls.load_predictions("experiment2_lora"),
        }

    @classmethod
    def evaluate_four_way_comparison(cls) -> pd.DataFrame:
        """Master summary table comparing all 4 extraction approaches."""
        models = [
            ("Zero-Shot Qwen3-4B", "zero_shot", "Baseline Prompting"),
            ("Few-Shot Qwen3-4B", "few_shot", "In-Context Learning (9 Clean Demos)"),
            ("Experiment 1 LoRA", "experiment1_lora", "Standard QLoRA Adapter"),
            ("Experiment 2 LoRA (Provisional)", "experiment2_lora", "Evidence-Aware Supervision LoRA*"),
        ]

        summary_rows = []
        for display_name, key, variant in models:
            preds = cls.load_predictions(key)
            if not preds:
                continue

            metrics = ModelEvaluator.evaluate_predictions(preds)
            summary_rows.append({
                "Model / Method": display_name,
                "Approach": variant,
                "Test Records": metrics["total_docs"],
                "JSON Validity (%)": metrics["json_validity"],
                "Schema Validity (%)": metrics["schema_validity"],
                "Field-Level Exact Match (%)": metrics["field_accuracy"],
                "Document-Level Exact Match (%)": metrics["document_accuracy"],
                "Correct Fields": f"{metrics['correct_fields']}/{metrics['total_fields']}",
                "Exact Documents": f"{metrics['exact_documents']}/{metrics['total_docs']}",
            })

        return pd.DataFrame(summary_rows)

    @classmethod
    def get_language_breakdown(cls) -> pd.DataFrame:
        preds_map = cls.load_all_predictions_map()
        return ModelEvaluator.compute_grouped_metrics(preds_map, group_by="language")

    @classmethod
    def get_category_breakdown(cls) -> pd.DataFrame:
        preds_map = cls.load_all_predictions_map()
        return ModelEvaluator.compute_grouped_metrics(preds_map, group_by="category")

    @classmethod
    def get_engine_breakdown(cls) -> pd.DataFrame:
        preds_map = cls.load_all_predictions_map()
        return ModelEvaluator.compute_grouped_metrics(preds_map, group_by="ocr_engine")

    @classmethod
    def get_field_breakdown(cls) -> pd.DataFrame:
        preds_map = cls.load_all_predictions_map()
        return ModelEvaluator.compute_field_level_breakdown(preds_map)

    @classmethod
    def get_paired_comparisons_summary(cls) -> pd.DataFrame:
        """
        Record-level paired comparisons across 54 OCR test records (5 fields per record).
        """
        zs = cls.load_predictions("zero_shot")
        fs = cls.load_predictions("few_shot")
        e1 = cls.load_predictions("experiment1_lora")
        e2 = cls.load_predictions("experiment2_lora")

        comparisons = [
            ModelEvaluator.compute_paired_differences(zs, fs, "Zero-Shot", "Few-Shot"),
            ModelEvaluator.compute_paired_differences(zs, e1, "Zero-Shot", "Exp 1 LoRA"),
            ModelEvaluator.compute_paired_differences(zs, e2, "Zero-Shot", "Exp 2 LoRA*"),
            ModelEvaluator.compute_paired_differences(e1, fs, "Exp 1 LoRA", "Few-Shot"),
        ]
        df = pd.DataFrame(comparisons)
        df["analysis_unit"] = "54 OCR Records"
        return df

    @classmethod
    def get_source_document_paired_comparisons_summary(cls) -> pd.DataFrame:
        """
        Source-document-level paired comparisons aggregated across the 18 unique source documents.
        For each source document, the 3 OCR-engine records (Tesseract, EasyOCR, PaddleOCR)
        are pooled (15 fields per document, 270 fields total across all 18 documents).
        """
        zs = cls.load_predictions("zero_shot")
        fs = cls.load_predictions("few_shot")
        e1 = cls.load_predictions("experiment1_lora")
        e2 = cls.load_predictions("experiment2_lora")

        def group_by_doc(preds):
            docs = {}
            for r in preds:
                d_id = r.get("document_id")
                if d_id not in docs:
                    docs[d_id] = []
                docs[d_id].append(r)
            return docs

        zs_docs = group_by_doc(zs)
        fs_docs = group_by_doc(fs)
        e1_docs = group_by_doc(e1)
        e2_docs = group_by_doc(e2)

        pairs = [
            ("Few-Shot vs Zero-Shot", fs_docs, zs_docs, "Few-Shot", "Zero-Shot"),
            ("Exp 1 LoRA vs Zero-Shot", e1_docs, zs_docs, "Exp 1 LoRA", "Zero-Shot"),
            ("Exp 2 LoRA* vs Zero-Shot", e2_docs, zs_docs, "Exp 2 LoRA*", "Zero-Shot"),
            ("Few-Shot vs Exp 1 LoRA", fs_docs, e1_docs, "Few-Shot", "Exp 1 LoRA"),
        ]

        rows = []
        for comp_name, dict_a, dict_b, label_a, label_b in pairs:
            wins_a, wins_b, ties = 0, 0, 0
            field_diffs = []
            all_doc_ids = sorted(set(dict_a.keys()) | set(dict_b.keys()))

            for d_id in all_doc_ids:
                recs_a = dict_a.get(d_id, [])
                recs_b = dict_b.get(d_id, [])

                def count_correct(records):
                    total_c = 0
                    for r in records:
                        gt = r.get("ground_truth", {})
                        pred = r.get("prediction") or {}
                        for f, val in gt.items():
                            if normalize_field_value(pred.get(f, ""), f) == normalize_field_value(val, f):
                                total_c += 1
                    return total_c

                c_a = count_correct(recs_a)
                c_b = count_correct(recs_b)
                diff = c_a - c_b
                field_diffs.append(diff)

                if diff > 0:
                    wins_a += 1
                elif diff < 0:
                    wins_b += 1
                else:
                    ties += 1

            rows.append({
                "comparison": comp_name,
                "analysis_unit": "18 Source Docs (15 fields each)",
                "improved_docs": wins_a,
                "identical_docs": ties,
                "degraded_docs": wins_b,
                "net_field_gain": sum(field_diffs),
                "mean_field_diff_per_doc": round(float(np.mean(field_diffs)), 3) if field_diffs else 0.0,
            })

        return pd.DataFrame(rows)
