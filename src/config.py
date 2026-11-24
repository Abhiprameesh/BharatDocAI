"""
BharatDoc AI - Configuration & Constants
Centralizes paths, document schemas, supported languages, and research metadata.
All paths are dynamically resolved relative to the project root.
"""

import os
from pathlib import Path

# Project Root Directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Research Archive Paths (Read-Only Reference)
RESEARCH_DIR = PROJECT_ROOT / "Research" / "BharatDocAI"
RESEARCH_DATA_DIR = RESEARCH_DIR / "data"
ANNOTATIONS_FILE = RESEARCH_DATA_DIR / "annotations" / "synthetic_documents.jsonl"
STRUCTURED_GROUND_TRUTH_FILE = RESEARCH_DATA_DIR / "annotations" / "structured_ground_truth.json"
SPLITS_DIR = RESEARCH_DATA_DIR / "splits"
DATASET_MANIFEST_FILE = SPLITS_DIR / "dataset_manifest.csv"
RAW_DOCUMENTS_DIR = RESEARCH_DATA_DIR / "raw_documents"
DEGRADED_DOCUMENTS_DIR = RESEARCH_DATA_DIR / "processed" / "degraded"
DEGRADATION_MANIFEST_FILE = RESEARCH_DATA_DIR / "processed" / "degradation_manifest.csv"

# OCR Outputs from Research Archive
OCR_OUTPUTS_DIR = RESEARCH_DATA_DIR / "ocr_outputs"
EASYOCR_RESULTS_FILE = OCR_OUTPUTS_DIR / "easyocr" / "easyocr_all_results.jsonl"
PADDLEOCR_RESULTS_FILE = OCR_OUTPUTS_DIR / "paddleocr" / "paddleocr_all_results.jsonl"
TESSERACT_CLEAN_FILE = OCR_OUTPUTS_DIR / "tesseract" / "tesseract_clean_results.jsonl"
TESSERACT_DEGRADED_FILE = OCR_OUTPUTS_DIR / "tesseract" / "tesseract_degraded_results.jsonl"

# Experiments & Predictions Paths
EXPERIMENTS_DIR = RESEARCH_DIR / "experiments"
FOUR_WAY_PREDICTIONS_DIR = EXPERIMENTS_DIR / "finetuning" / "four_way_comparison"
ZERO_SHOT_PREDICTIONS_FILE = FOUR_WAY_PREDICTIONS_DIR / "zero_shot_predictions.jsonl"
FEW_SHOT_PREDICTIONS_FILE = FOUR_WAY_PREDICTIONS_DIR / "few_shot_predictions.jsonl"
EXP1_LORA_PREDICTIONS_FILE = FOUR_WAY_PREDICTIONS_DIR / "experiment1_lora_predictions.jsonl"
EXP2_LORA_PREDICTIONS_FILE = FOUR_WAY_PREDICTIONS_DIR / "experiment2_lora_predictions.jsonl"
EXP2_TEST_PREDICTIONS_FILE = EXPERIMENTS_DIR / "finetuning" / "experiment2_test_predictions.jsonl"

# Model Checkpoints (Adapters)
MODELS_DIR = RESEARCH_DIR / "models" / "checkpoints"
EXP1_ADAPTER_DIR = MODELS_DIR / "qwen3_lora"
EXP2_ADAPTER_DIR = MODELS_DIR / "qwen3_lora_experiment2"
BASE_MODEL_NAME = "Qwen/Qwen3-4B"

# Outputs & Metrics Paths
OUTPUTS_DIR = RESEARCH_DIR / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
FINETUNING_METRICS_DIR = METRICS_DIR / "finetuning"
VISUALIZATIONS_DIR = OUTPUTS_DIR / "visualizations"
OCR_BENCHMARK_SUMMARY_FILE = METRICS_DIR / "ocr_benchmark_summary.txt"
THREE_ENGINE_OVERALL_FILE = METRICS_DIR / "three_engine_overall_comparison.csv"
THREE_ENGINE_LANGUAGE_FILE = METRICS_DIR / "three_engine_language_comparison.csv"
THREE_ENGINE_CORRUPTION_FILE = METRICS_DIR / "three_engine_corruption_comparison.csv"
THREE_ENGINE_SEVERITY_FILE = METRICS_DIR / "three_engine_severity_comparison.csv"
BOOTSTRAP_COMPARISONS_FILE = METRICS_DIR / "paired_bootstrap_comparisons.csv"
DOCUMENT_CIs_FILE = METRICS_DIR / "document_level_engine_confidence_intervals.csv"
EXP1_TRAINING_HISTORY_FILE = FINETUNING_METRICS_DIR / "qwen3_lora_training_history.json"
EXP2_TRAINING_HISTORY_FILE = FINETUNING_METRICS_DIR / "experiment2_training_history.json"
ZERO_SHOT_SUMMARY_FILE = METRICS_DIR / "llm_extraction" / "qwen3_zero_shot_baseline_summary.json"
FEW_SHOT_SUMMARY_FILE = METRICS_DIR / "llm_extraction" / "qwen3_few_shot_summary.json"

# Error & Recoverability Diagnostics
QWEN3_LORA_OCR_SUPPORT_FILE = FINETUNING_METRICS_DIR / "qwen3_lora_ocr_support_diagnostic.csv"
THREE_MODEL_FALSE_ABSTENTION_FILE = FINETUNING_METRICS_DIR / "three_model_false_abstention_comparison.csv"
EXP2_EVIDENCE_AUDIT_FILE = FINETUNING_METRICS_DIR / "experiment2_evidence_audit.csv"
EXP2_AUTOMATED_EVIDENCE_FILE = FINETUNING_METRICS_DIR / "experiment2_automated_evidence_classification.csv"
QWEN3_LORA_FIELD_ERROR_FILE = FINETUNING_METRICS_DIR / "qwen3_lora_field_error_analysis.csv"

# Static Assets
ASSETS_DIR = PROJECT_ROOT / "assets"
CSS_FILE = ASSETS_DIR / "css" / "style.css"

# Document Extraction Schemas (Verified from Research Ground Truth)
CATEGORY_SCHEMAS = {
    "invoice": ["name", "date", "document_id", "address", "amount"],
    "application": ["name", "date", "document_id", "address", "phone"],
    "certificate": ["name", "date", "document_id", "institution", "grade"],
}

# Category Display Names
CATEGORY_NAMES = {
    "invoice": "Invoice / Bill",
    "application": "Application Form",
    "certificate": "Certificate / Grade Sheet",
}

# Supported Languages
SUPPORTED_LANGUAGES = ["english", "hindi", "kannada"]
LANGUAGE_LABELS = {
    "english": "English",
    "hindi": "Hindi (हिन्दी)",
    "kannada": "Kannada (ಕನ್ನಡ)",
}

# Supported OCR Engines
OCR_ENGINES = {
    "easyocr": "EasyOCR (Synthetic Benchmark Leader)",
    "tesseract": "Tesseract OCR",
    "paddleocr": "PaddleOCR (Multilingual)",
}
DEFAULT_OCR_ENGINE = "easyocr"

# Supported Upload File Extensions
ALLOWED_EXTENSIONS = [".png", ".jpg", ".jpeg", ".pdf"]

# Standard Extraction System Prompt
EXTRACTION_SYSTEM_PROMPT = (
    "You are an information extraction system.\n"
    "Extract only information explicitly supported by the OCR text.\n"
    "Never guess or invent missing values.\n"
    "Return exactly the requested fields as valid JSON.\n"
    "Use an empty string when a field is missing or unreadable."
)

# Scientific Findings regarding Experiment 2 (Evidence-Aware Supervision)
EXP2_STATUS = "GPU-Verified (Evidence-Aware LoRA)"
EXP2_STATUS_DETAILS = (
    "Experiment 2 test inference was executed on an NVIDIA GPU using the trained "
    "qwen3_lora_experiment2 adapter. The adapter achieves 75.19% field accuracy and "
    "35.19% document exact match, demonstrating that evidence-aware supervision successfully "
    "mitigated the severe over-abstention penalty seen in Experiment 1 (69.63%), but does not "
    "outperform Few-Shot in-context learning (83.33%)."
)
