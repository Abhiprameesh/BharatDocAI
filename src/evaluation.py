"""
BharatDoc AI - Evaluation Module
Implements research-grade evaluation metrics for multilingual information extraction:
- Exact string normalization (NFC, casefold, separator stripping)
- Normalized field-level exact match accuracy
- Document-level exact match
- JSON and schema validity verification
- Granular performance breakdowns (Language, Category, OCR Engine, Field)
- Paired model comparison calculations
"""

import re
import unicodedata
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

from src.config import CATEGORY_SCHEMAS

def normalize_field_value(value: Any, field: str) -> str:
    """
    Exact research normalization from Phase 2.1 & 2.2 notebooks.
    Applies NFC unicode normalization, strips zero-width joiners/non-joiners,
    applies casefolding, and removes structural punctuation separators for
    dates, phones, amounts, and document IDs.
    """
    if value is None:
        return ""
    val_str = unicodedata.normalize("NFC", str(value))
    # Strip zero-width non-joiner and joiner characters
    val_str = val_str.replace("\u200c", "").replace("\u200d", "")
    val_str = val_str.strip().casefold()

    if field in ["date", "phone", "amount", "document_id"]:
        # Remove formatting separators, retaining alphanumeric content
        val_str = re.sub(r"[\s/\\\-:.,|]", "", val_str)
    else:
        # Collapse multiple whitespace characters
        val_str = re.sub(r"\s+", " ", val_str)

    return val_str

def compute_record_metrics(
    ground_truth: Dict[str, Any],
    prediction: Optional[Dict[str, Any]],
    category: str
) -> Tuple[int, int, bool, bool, bool]:
    """
    Computes field and document level metrics for a single prediction record.
    Returns:
        (correct_fields, total_fields, exact_document_match, json_valid, schema_valid)
    """
    expected_fields = CATEGORY_SCHEMAS.get(category.lower(), [])
    total_fields = len(expected_fields)
    
    if not prediction or not isinstance(prediction, dict):
        return 0, total_fields, False, False, False

    json_valid = True
    actual_keys = set(prediction.keys())
    expected_set = set(expected_fields)
    schema_valid = (actual_keys == expected_set)

    correct_fields = 0
    all_matched = True

    for field in expected_fields:
        target_val = ground_truth.get(field, "")
        pred_val = prediction.get(field, "")
        
        norm_target = normalize_field_value(target_val, field)
        norm_pred = normalize_field_value(pred_val, field)

        if norm_target == norm_pred:
            correct_fields += 1
        else:
            all_matched = False

    exact_doc_match = (all_matched and total_fields > 0)
    return correct_fields, total_fields, exact_doc_match, json_valid, schema_valid


class ModelEvaluator:
    """Evaluates prediction datasets and computes granular breakdowns."""

    @staticmethod
    def evaluate_predictions(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute aggregate metrics for a single model's prediction list."""
        total_docs = len(predictions)
        total_fields = 0
        correct_fields = 0
        exact_docs = 0
        valid_json_count = 0
        valid_schema_count = 0

        for r in predictions:
            gt = r.get("ground_truth", {})
            pred = r.get("prediction")
            cat = r.get("category", "invoice")
            
            c_fields, t_fields, doc_match, j_valid, s_valid = compute_record_metrics(gt, pred, cat)
            
            total_fields += t_fields
            correct_fields += c_fields
            if doc_match:
                exact_docs += 1
            if j_valid:
                valid_json_count += 1
            if s_valid:
                valid_schema_count += 1

        field_acc = (correct_fields / total_fields) * 100 if total_fields else 0.0
        doc_acc = (exact_docs / total_docs) * 100 if total_docs else 0.0
        json_rate = (valid_json_count / total_docs) * 100 if total_docs else 0.0
        schema_rate = (valid_schema_count / total_docs) * 100 if total_docs else 0.0

        return {
            "total_docs": total_docs,
            "total_fields": total_fields,
            "correct_fields": correct_fields,
            "exact_documents": exact_docs,
            "field_accuracy": round(field_acc, 2),
            "document_accuracy": round(doc_acc, 2),
            "json_validity": round(json_rate, 2),
            "schema_validity": round(schema_rate, 2),
        }

    @classmethod
    def compute_grouped_metrics(
        cls, predictions_map: Dict[str, List[Dict[str, Any]]], group_by: str
    ) -> pd.DataFrame:
        """
        Compute comparative metrics grouped by a specific metadata key:
        'language', 'category', or 'ocr_engine'.
        """
        rows = []
        for model_name, preds in predictions_map.items():
            groups = {}
            for r in preds:
                group_val = r.get(group_by, "Unknown")
                if group_val not in groups:
                    groups[group_val] = []
                groups[group_val].append(r)

            for group_val, group_preds in sorted(groups.items()):
                metrics = cls.evaluate_predictions(group_preds)
                rows.append({
                    "Model": model_name,
                    group_by.title(): group_val.title(),
                    "Field Accuracy (%)": metrics["field_accuracy"],
                    "Document Match (%)": metrics["document_accuracy"],
                    "Correct Fields": f"{metrics['correct_fields']}/{metrics['total_fields']}",
                    "Exact Docs": f"{metrics['exact_documents']}/{metrics['total_docs']}",
                })
        return pd.DataFrame(rows)

    @classmethod
    def compute_field_level_breakdown(
        cls, predictions_map: Dict[str, List[Dict[str, Any]]]
    ) -> pd.DataFrame:
        """
        Compute accuracy broken down by each individual schema field
        (name, date, document_id, address, amount, phone, institution, grade).
        """
        rows = []
        for model_name, preds in predictions_map.items():
            field_stats = {}
            for r in preds:
                gt = r.get("ground_truth", {})
                pred = r.get("prediction") or {}
                for field, expected in gt.items():
                    if field not in field_stats:
                        field_stats[field] = {"correct": 0, "total": 0}
                    field_stats[field]["total"] += 1
                    
                    norm_exp = normalize_field_value(expected, field)
                    norm_pred = normalize_field_value(pred.get(field, ""), field)
                    if norm_exp == norm_pred:
                        field_stats[field]["correct"] += 1

            for field, data in sorted(field_stats.items()):
                acc = (data["correct"] / data["total"]) * 100 if data["total"] else 0.0
                rows.append({
                    "Model": model_name,
                    "Field": field,
                    "Accuracy (%)": round(acc, 2),
                    "Correct / Total": f"{data['correct']}/{data['total']}",
                })
        return pd.DataFrame(rows)

    @classmethod
    def compute_paired_differences(
        cls,
        base_preds: List[Dict[str, Any]],
        tuned_preds: List[Dict[str, Any]],
        base_label: str = "Zero-Shot",
        tuned_label: str = "Few-Shot"
    ) -> Dict[str, Any]:
        """
        Compute paired document-by-document difference between two model predictions.
        """
        doc_wins = 0
        doc_losses = 0
        doc_ties = 0

        field_diffs = []
        for b_rec, t_rec in zip(base_preds, tuned_preds):
            gt = b_rec.get("ground_truth", {})
            b_pred = b_rec.get("prediction") or {}
            t_pred = t_rec.get("prediction") or {}

            b_correct = sum(
                1 for f, val in gt.items()
                if normalize_field_value(b_pred.get(f, ""), f) == normalize_field_value(val, f)
            )
            t_correct = sum(
                1 for f, val in gt.items()
                if normalize_field_value(t_pred.get(f, ""), f) == normalize_field_value(val, f)
            )

            if t_correct > b_correct:
                doc_wins += 1
            elif t_correct < b_correct:
                doc_losses += 1
            else:
                doc_ties += 1

            field_diffs.append(t_correct - b_correct)

        return {
            "comparison": f"{tuned_label} vs {base_label}",
            "improved_docs": doc_wins,
            "degraded_docs": doc_losses,
            "identical_docs": doc_ties,
            "net_field_gain": sum(field_diffs),
            "mean_field_diff_per_doc": round(float(np.mean(field_diffs)), 3) if field_diffs else 0.0,
        }
