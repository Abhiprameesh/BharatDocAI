# BharatDoc AI - Research Archive Inventory

**Generated on:** 2026-11-20  
**Base Directory:** C:\ALLMLPROJ\MAINMLPROJS\BharatDoc AI\Research\BharatDocAI  
**Total File Count:** 1,563 files  
**Total Archive Size:** 835.20 MB  

This document provides an exhaustive inventory of all original research artifacts produced during the Google Colab experimental phases of BharatDoc AI. In accordance with scientific integrity guidelines, all original research files remain completely untouched and unmodified.

---

## Summary by Artifact Classification

| Classification | File Formats | File Count | Approx. Total Size | Description |
|---|---|---|---|---|
| **Research Notebooks** | .ipynb | 7 | 4.46 MB | Original Jupyter/Colab notebooks covering Phases 0 through 2.2 |
| **Model Checkpoints** | .safetensors, .json, .jinja, .md | 13 | 151.49 MB | LoRA adapter weights (Experiment 1 and 2), tokenizer, and configs |
| **Document Annotations** | .json, .jsonl | 3 | 59.88 KB | Synthetic ground truth (90 docs) and Phase 0 sample ground truth |
| **Splits & Manifests** | .csv, .jsonl | 4 | 70.45 KB | Train (54), Val (18), Test (18) split definitions and dataset manifest |
| **Raw Document Images** | .png | 93 | 3.10 MB | 90 clean synthetic document images + 3 Phase 0 invoice images |
| **Degraded Document Images** | .png, .csv | 1,351 | 660.18 MB | 1,350 degraded image variants across 5 corruptions + degradation manifest |
| **OCR Engine Outputs** | .jsonl, .json | 7 | 3.76 MB | Full-dataset OCR outputs for EasyOCR, PaddleOCR, and Tesseract (1,440 records each) |
| **Baseline & Extraction Datasets** | .jsonl, .csv, .json | 5 | 4.97 MB | Unified extraction dataset and intermediate baseline extraction logs |
| **Four-Way Comparison Predictions** | .jsonl | 4 | 212.76 KB | Exact 54-test-record predictions for Zero-Shot, Few-Shot, Exp 1 LoRA, Exp 2 LoRA |
| **Evaluation Metrics & CIs** | .csv, .json, .txt | 44 | 1.84 MB | Comprehensive OCR CER/WER benchmarks, bootstrap CIs, LLM extraction metrics, evidence audits |
| **Visualizations** | .png | 5 | 822.45 KB | Pre-rendered benchmark charts for OCR engines, corruption, language, severity |

## 1. Research Notebooks

| File Path | Type | Size | Artifact Category | Necessary for App? | Description |
|---|---|---|---|---|---|
| Research\BharatDocAI\Notebooks\BharatDoc_AI_Phase_0.ipynb | .ipynb | 368.12 KB | Research Notebook | No (Reference Only) | Phase 0 feasibility study: sample generation, font rendering (Noto Sans), basic OCR, ground truth alignment. |
| Research\BharatDocAI\Notebooks\BharatDoc_AI_Phase_0_5_LLM.ipynb | .ipynb | 1.78 MB | Research Notebook | No (Reference Only) | Phase 0.5 prototype: early LLM extraction exploration with base Qwen/Qwen3-4B. |
| Research\BharatDocAI\Notebooks\BharatDoc_AI_Phase_1_3_OCR_Benchmarking.ipynb.ipynb | .ipynb | 383.97 KB | Research Notebook | No (Reference Only) | Phase 1.3 OCR benchmark: full Tesseract evaluation on clean and degraded dataset. |
| Research\BharatDocAI\Notebooks\BharatDoc_AI_Phase_2_1_LLM_Extraction.ipynb | .ipynb | 324.09 KB | Research Notebook | No (Reference Only) | Phase 2.1 zero-shot and few-shot baseline extraction pipeline on 54 test documents. |
| Research\BharatDocAI\Notebooks\BharatDoc_AI_Phase_2_2_Experiment2_Training.ipynb | .ipynb | 312.33 KB | Research Notebook | No (Reference Only) | Phase 2.2 Experiment 2: Evidence-aware supervision training and four-way comparison inference. |
| Research\BharatDocAI\Notebooks\BharatDoc_AI_Phase_2_2_LoRA_FineTuning.ipynb | .ipynb | 598.59 KB | Research Notebook | No (Reference Only) | Phase 2.2 Experiment 1: Standard QLoRA fine-tuning of Qwen3-4B on synthetic training set. |
| Research\BharatDocAI\Notebooks\EasyOCR and PaddleOCR benchmarking.ipynb | .ipynb | 767.91 KB | Research Notebook | No (Reference Only) | Phase 1 OCR benchmarking for EasyOCR and PaddleOCR across languages, corruptions, and severities. |

---

## 2. Model Checkpoints and Adapters

| File Path | Type | Size | Artifact Category | Necessary for App? | Description |
|---|---|---|---|---|---|
| Research\BharatDocAI\models\checkpoints\qwen3_lora\adapter_config.json | .json | 1.14 KB | Model Checkpoint | Yes (Optional Live Inference Backend) | PEFT LoRA adapter configuration (r=8, alpha=16, dropout=0.05, 7 projection modules). [Experiment 1 (Standard LoRA)] |
| Research\BharatDocAI\models\checkpoints\qwen3_lora\adapter_model.safetensors | .safetensors | 63.06 MB | Model Checkpoint | Yes (Optional Live Inference Backend) | LoRA adapter trained weights for Qwen3-4B (64.58 MB safetensors). [Experiment 1 (Standard LoRA)] |
| Research\BharatDocAI\models\checkpoints\qwen3_lora\chat_template.jinja | .jinja | 4.07 KB | Model Checkpoint | No (Optional) | Jinja chat template for Qwen tokenizer formatting. [Experiment 1 (Standard LoRA)] |
| Research\BharatDocAI\models\checkpoints\qwen3_lora\README.md | .md | 5.06 KB | Model Checkpoint | No (Optional) | Model card metadata generated by Hugging Face PEFT. [Experiment 1 (Standard LoRA)] |
| Research\BharatDocAI\models\checkpoints\qwen3_lora\tokenizer.json | .json | 10.89 MB | Model Checkpoint | Yes (Optional Live Inference Backend) | Full Hugging Face BPE tokenizer vocab and merges for Qwen3-4B. [Experiment 1 (Standard LoRA)] |
| Research\BharatDocAI\models\checkpoints\qwen3_lora\tokenizer_config.json | .json | 694 B | Model Checkpoint | Yes (Optional Live Inference Backend) | Tokenizer configuration and special token definitions. [Experiment 1 (Standard LoRA)] |
| Research\BharatDocAI\models\checkpoints\qwen3_lora_experiment2\adapter_config.json | .json | 1.14 KB | Model Checkpoint | Yes (Optional Live Inference Backend) | PEFT LoRA adapter configuration (r=8, alpha=16, dropout=0.05, 7 projection modules). [Experiment 2 (Evidence-Aware LoRA)] |
| Research\BharatDocAI\models\checkpoints\qwen3_lora_experiment2\adapter_model.safetensors | .safetensors | 63.06 MB | Model Checkpoint | Yes (Optional Live Inference Backend) | LoRA adapter trained weights for Qwen3-4B (64.58 MB safetensors). [Experiment 2 (Evidence-Aware LoRA)] |
| Research\BharatDocAI\models\checkpoints\qwen3_lora_experiment2\chat_template.jinja | .jinja | 4.07 KB | Model Checkpoint | No (Optional) | Jinja chat template for Qwen tokenizer formatting. [Experiment 2 (Evidence-Aware LoRA)] |
| Research\BharatDocAI\models\checkpoints\qwen3_lora_experiment2\experiment2_config.json | .json | 399 B | Model Checkpoint | Yes (Optional Live Inference Backend) | Experiment 2 specific hyperparameter and evidence masking configuration manifest. [Experiment 2 (Evidence-Aware LoRA)] |
| Research\BharatDocAI\models\checkpoints\qwen3_lora_experiment2\README.md | .md | 5.06 KB | Model Checkpoint | No (Optional) | Model card metadata generated by Hugging Face PEFT. [Experiment 2 (Evidence-Aware LoRA)] |
| Research\BharatDocAI\models\checkpoints\qwen3_lora_experiment2\tokenizer.json | .json | 10.89 MB | Model Checkpoint | Yes (Optional Live Inference Backend) | Full Hugging Face BPE tokenizer vocab and merges for Qwen3-4B. [Experiment 2 (Evidence-Aware LoRA)] |
| Research\BharatDocAI\models\checkpoints\qwen3_lora_experiment2\tokenizer_config.json | .json | 694 B | Model Checkpoint | Yes (Optional Live Inference Backend) | Tokenizer configuration and special token definitions. [Experiment 2 (Evidence-Aware LoRA)] |

---

## 3. Document Annotations, Splits, and Manifests

| File Path | Type | Size | Artifact Category | Necessary for App? | Description |
|---|---|---|---|---|---|
| Research\BharatDocAI\data\annotations\phase0_ground_truth.json | .json | 727 B | Dataset / Annotation | No (Phase 0 Archive) | Ground truth text for the 3 Phase 0 multilingual invoices. |
| Research\BharatDocAI\data\annotations\structured_ground_truth.json | .json | 600 B | Dataset / Annotation | No (Phase 0 Archive) | Ground truth key-value annotations for the 3 Phase 0 sample documents. |
| Research\BharatDocAI\data\annotations\synthetic_documents.jsonl | .jsonl | 58.58 KB | Dataset / Annotation | Yes (Research Dashboard & Live Verification) | Ground truth annotations for all 90 synthetic documents across 3 languages and 3 categories (Invoice, Application, Certificate). |
| Research\BharatDocAI\data\splits\dataset_manifest.csv | .csv | 11.56 KB | Dataset / Annotation | Yes (Research Dashboard & Live Verification) | Master split index listing all 90 source documents with language, category, template family, and split assignment. |
| Research\BharatDocAI\data\splits\test.jsonl | .jsonl | 11.71 KB | Dataset / Annotation | Yes (Research Dashboard & Live Verification) | Test split definition containing 18 document ground truth records. |
| Research\BharatDocAI\data\splits\train.jsonl | .jsonl | 35.18 KB | Dataset / Annotation | Yes (Research Dashboard & Live Verification) | Training split definition containing 54 document ground truth records. |
| Research\BharatDocAI\data\splits\validation.jsonl | .jsonl | 11.70 KB | Dataset / Annotation | Yes (Research Dashboard & Live Verification) | Validation split definition containing 18 document ground truth records. |

---

## 4. OCR Engine Outputs

| File Path | Type | Size | Artifact Category | Necessary for App? | Description |
|---|---|---|---|---|---|
| `Research\BharatDocAI\data\ocr_outputs\easyocr\easyocr_all_results.jsonl` | `.jsonl` | 1.24 MB | Dataset (OCR Output) | Yes (OCR Benchmark & Research Dashboard) | Complete EasyOCR recognitions for all 1,440 images (90 clean + 1,350 degraded) across English, Hindi, Kannada. |
| `Research\BharatDocAI\data\ocr_outputs\easyocr_results.json` | `.json` | 616 B | Dataset (OCR Output) | No (Historical Reference) | Preliminary sample run OCR results for easyocr. |
| `Research\BharatDocAI\data\ocr_outputs\paddleocr\paddleocr_all_results.jsonl` | `.jsonl` | 1.21 MB | Dataset (OCR Output) | Yes (OCR Benchmark & Research Dashboard) | Complete PaddleOCR recognitions for all 1,440 images across English, Hindi, Kannada. |
| `Research\BharatDocAI\data\ocr_outputs\paddleocr_results.json` | `.json` | 319 B | Dataset (OCR Output) | No (Historical Reference) | Preliminary sample run OCR results for paddleocr. |
| `Research\BharatDocAI\data\ocr_outputs\phase0_ocr.json` | `.json` | 597 B | Dataset (OCR Output) | No (Historical Reference) | Phase 0 prototype OCR output for sample documents. |
| `Research\BharatDocAI\data\ocr_outputs\tesseract\tesseract_clean_results.jsonl` | `.jsonl` | 71.77 KB | Dataset (OCR Output) | Yes (OCR Benchmark & Research Dashboard) | Tesseract OCR text recognitions on 90 clean document images. |
| `Research\BharatDocAI\data\ocr_outputs\tesseract\tesseract_degraded_results.jsonl` | `.jsonl` | 1.14 MB | Dataset (OCR Output) | Yes (OCR Benchmark & Research Dashboard) | Tesseract OCR text recognitions on 1,350 degraded document images. |

---

## 5. Baseline Datasets & Extraction Training Manifests

| File Path | Type | Size | Artifact Category | Necessary for App? | Description |
|---|---|---|---|---|---|
| `Research\BharatDocAI\experiments\baselines\llm_extraction\qwen3_few_shot_validation_results.jsonl` | `.jsonl` | 36.49 KB | Dataset / Experiment Log | No (Validation checkpoint) | Validation split intermediate inference results for qwen3_few_shot_validation_results. |
| `Research\BharatDocAI\experiments\baselines\llm_extraction\qwen3_zero_shot_partial.jsonl` | `.jsonl` | 27.00 KB | Dataset / Experiment Log | No (Intermediate log) | Partial run log of zero-shot extraction during Colab development. |
| `Research\BharatDocAI\experiments\baselines\llm_extraction\qwen3_zero_shot_validation_results.jsonl` | `.jsonl` | 39.03 KB | Dataset / Experiment Log | No (Validation checkpoint) | Validation split intermediate inference results for qwen3_zero_shot_validation_results. |
| `Research\BharatDocAI\experiments\baselines\llm_extraction\unified_extraction_dataset.csv` | `.csv` | 1.97 MB | Dataset / Experiment Log | Yes (Baseline reference & live evaluation) | Consolidated dataset mapping document annotations to OCR engine texts across splits. |
| `Research\BharatDocAI\experiments\baselines\llm_extraction\unified_extraction_dataset.jsonl` | `.jsonl` | 2.75 MB | Dataset / Experiment Log | Yes (Baseline reference & live evaluation) | Consolidated dataset mapping document annotations to OCR engine texts across splits. |
| `Research\BharatDocAI\experiments\baselines\qwen3_zero_shot_results.json` | `.json` | 3.83 KB | Dataset / Experiment Log | No (Phase 0 archive) | Phase 0 zero-shot prototype results. |
| `Research\BharatDocAI\experiments\finetuning\experiment2_test_predictions.jsonl` | `.jsonl` | 48.56 KB | Dataset / Experiment Log | Yes (Research analysis & verification) | Experiment 2 initial test set predictions generated immediately post-training (54 clean test records). |
| `Research\BharatDocAI\experiments\finetuning\experiment2_tokenized_training.json` | `.json` | 636.67 KB | Dataset / Experiment Log | No (Training artifact only) | Pre-tokenized input IDs, attention masks, and masked loss labels for Experiment 2 training. |
| `Research\BharatDocAI\experiments\finetuning\experiment2_training_manifest.csv` | `.csv` | 17.72 KB | Dataset / Experiment Log | No (Training artifact only) | Manifest recording the evidence status and supported/masked field counts for each training record. |

---

## 6. Four-Way Extraction Prediction Records (Test Set: 54 Clean Documents)

| File Path | Type | Size | Artifact Category | Necessary for App? | Description |
|---|---|---|---|---|---|
| `Research\BharatDocAI\experiments\finetuning\four_way_comparison\experiment1_lora_predictions.jsonl` | `.jsonl` | 52.72 KB | Prediction | Yes (Model Comparison Page & Dashboard) | Standard QLoRA fine-tuned Qwen3-4B predictions on 54 test records. |
| `Research\BharatDocAI\experiments\finetuning\four_way_comparison\experiment2_lora_predictions.jsonl` | `.jsonl` | 53.30 KB | Prediction | Yes (Model Comparison Page & Dashboard) | Evidence-aware LoRA predictions on 54 test records (**Identical to Zero-Shot across 54/54 test records: Marked for Scientific Verification**). |
| `Research\BharatDocAI\experiments\finetuning\four_way_comparison\few_shot_predictions.jsonl` | `.jsonl` | 53.44 KB | Prediction | Yes (Model Comparison Page & Dashboard) | Few-shot Qwen3-4B predictions across identical 54 test records with in-context demonstrations. |
| `Research\BharatDocAI\experiments\finetuning\four_way_comparison\zero_shot_predictions.jsonl` | `.jsonl` | 53.30 KB | Prediction | Yes (Model Comparison Page & Dashboard) | Zero-shot Qwen3-4B predictions across 54 test records (EasyOCR, PaddleOCR, Tesseract x 18 test docs). |

---

## 7. Evaluation Metrics and Statistical Analysis Outputs

| File Path | Type | Size | Artifact Category | Necessary for App? | Description |
|---|---|---|---|---|---|
| `Research\BharatDocAI\outputs\metrics\dataset_integrity_audit.json` | `.json` | 597 B | Evaluation Result | Yes | Evaluation metrics file. |
| `Research\BharatDocAI\outputs\metrics\document_level_engine_confidence_intervals.csv` | `.csv` | 1.89 KB | Evaluation Result | Yes (Research Dashboard - Statistical Analysis Section) | Document-level paired bootstrap confidence intervals and statistical tests (CER/WER comparisons across OCR engines and languages). |
| `Research\BharatDocAI\outputs\metrics\easyocr_by_corruption.csv` | `.csv` | 461 B | Evaluation Result | Yes (Research Dashboard - Drill-down views) | Detailed engine-level evaluation metrics for easyocr_by_corruption. |
| `Research\BharatDocAI\outputs\metrics\easyocr_by_language.csv` | `.csv` | 231 B | Evaluation Result | Yes (Research Dashboard - Drill-down views) | Detailed engine-level evaluation metrics for easyocr_by_language. |
| `Research\BharatDocAI\outputs\metrics\easyocr_by_severity.csv` | `.csv` | 295 B | Evaluation Result | Yes (Research Dashboard - Drill-down views) | Detailed engine-level evaluation metrics for easyocr_by_severity. |
| `Research\BharatDocAI\outputs\metrics\easyocr_detailed_metrics.csv` | `.csv` | 171.41 KB | Evaluation Result | Yes (Research Dashboard - Drill-down views) | Detailed engine-level evaluation metrics for easyocr_detailed_metrics. |
| `Research\BharatDocAI\outputs\metrics\easyocr_language_corruption.csv` | `.csv` | 1.40 KB | Evaluation Result | Yes (Research Dashboard - Drill-down views) | Detailed engine-level evaluation metrics for easyocr_language_corruption. |
| `Research\BharatDocAI\outputs\metrics\easyocr_language_severity.csv` | `.csv` | 903 B | Evaluation Result | Yes (Research Dashboard - Drill-down views) | Detailed engine-level evaluation metrics for easyocr_language_severity. |
| `Research\BharatDocAI\outputs\metrics\easyocr_metrics.csv` | `.csv` | 187 B | Evaluation Result | Yes (Research Dashboard - Drill-down views) | Detailed engine-level evaluation metrics for easyocr_metrics. |
| `Research\BharatDocAI\outputs\metrics\easyocr_normalized_metrics.csv` | `.csv` | 182 B | Evaluation Result | Yes (Research Dashboard - Drill-down views) | Detailed engine-level evaluation metrics for easyocr_normalized_metrics. |
| `Research\BharatDocAI\outputs\metrics\finetuning\experiment2_automated_evidence_classification.csv` | `.csv` | 115.85 KB | Evaluation Result | Yes (Model Comparison & Error Analysis) | Fine-tuning validation metrics, training history, evidence classification audits, and OCR support diagnostics. |
| `Research\BharatDocAI\outputs\metrics\finetuning\experiment2_consistent_fuzzy_audit.csv` | `.csv` | 105.55 KB | Evaluation Result | Yes (Model Comparison & Error Analysis) | Fine-tuning validation metrics, training history, evidence classification audits, and OCR support diagnostics. |
| `Research\BharatDocAI\outputs\metrics\finetuning\experiment2_context_verification.csv` | `.csv` | 5.35 KB | Evaluation Result | Yes (Model Comparison & Error Analysis) | Fine-tuning validation metrics, training history, evidence classification audits, and OCR support diagnostics. |
| `Research\BharatDocAI\outputs\metrics\finetuning\experiment2_cross_ocr_fuzzy_verification.csv` | `.csv` | 105.43 KB | Evaluation Result | Yes (Model Comparison & Error Analysis) | Fine-tuning validation metrics, training history, evidence classification audits, and OCR support diagnostics. |
| `Research\BharatDocAI\outputs\metrics\finetuning\experiment2_evidence_audit.csv` | `.csv` | 268.57 KB | Evaluation Result | Yes (Model Comparison & Error Analysis) | Fine-tuning validation metrics, training history, evidence classification audits, and OCR support diagnostics. |
| `Research\BharatDocAI\outputs\metrics\finetuning\experiment2_line_level_audit.csv` | `.csv` | 25.70 KB | Evaluation Result | Yes (Model Comparison & Error Analysis) | Fine-tuning validation metrics, training history, evidence classification audits, and OCR support diagnostics. |
| `Research\BharatDocAI\outputs\metrics\finetuning\experiment2_manual_evidence_review.csv` | `.csv` | 28.02 KB | Evaluation Result | Yes (Model Comparison & Error Analysis) | Fine-tuning validation metrics, training history, evidence classification audits, and OCR support diagnostics. |
| `Research\BharatDocAI\outputs\metrics\finetuning\experiment2_proposed_evidence_categories.csv` | `.csv` | 30.30 KB | Evaluation Result | Yes (Model Comparison & Error Analysis) | Fine-tuning validation metrics, training history, evidence classification audits, and OCR support diagnostics. |
| `Research\BharatDocAI\outputs\metrics\finetuning\experiment2_strict_field_verification.csv` | `.csv` | 2.94 KB | Evaluation Result | Yes (Model Comparison & Error Analysis) | Fine-tuning validation metrics, training history, evidence classification audits, and OCR support diagnostics. |
| `Research\BharatDocAI\outputs\metrics\finetuning\experiment2_training_history.json` | `.json` | 1.01 KB | Evaluation Result | Yes (Model Comparison & Error Analysis) | Fine-tuning validation metrics, training history, evidence classification audits, and OCR support diagnostics. |
| `Research\BharatDocAI\outputs\metrics\finetuning\qwen3_lora_field_error_analysis.csv` | `.csv` | 25.53 KB | Evaluation Result | Yes (Model Comparison & Error Analysis) | Fine-tuning validation metrics, training history, evidence classification audits, and OCR support diagnostics. |
| `Research\BharatDocAI\outputs\metrics\finetuning\qwen3_lora_ocr_support_diagnostic.csv` | `.csv` | 27.28 KB | Evaluation Result | Yes (Model Comparison & Error Analysis) | Fine-tuning validation metrics, training history, evidence classification audits, and OCR support diagnostics. |
| `Research\BharatDocAI\outputs\metrics\finetuning\qwen3_lora_training_history.json` | `.json` | 691 B | Evaluation Result | Yes (Model Comparison & Error Analysis) | Fine-tuning validation metrics, training history, evidence classification audits, and OCR support diagnostics. |
| `Research\BharatDocAI\outputs\metrics\finetuning\qwen3_lora_validation_metrics.csv` | `.csv` | 4.04 KB | Evaluation Result | Yes (Model Comparison & Error Analysis) | Fine-tuning validation metrics, training history, evidence classification audits, and OCR support diagnostics. |
| `Research\BharatDocAI\outputs\metrics\finetuning\three_model_false_abstention_comparison.csv` | `.csv` | 1.07 KB | Evaluation Result | Yes (Model Comparison & Error Analysis) | Fine-tuning validation metrics, training history, evidence classification audits, and OCR support diagnostics. |
| `Research\BharatDocAI\outputs\metrics\finetuning\training_target_consistency_audit.csv` | `.csv` | 24.21 KB | Evaluation Result | Yes (Model Comparison & Error Analysis) | Fine-tuning validation metrics, training history, evidence classification audits, and OCR support diagnostics. |
| `Research\BharatDocAI\outputs\metrics\finetuning\training_target_consistency_types.csv` | `.csv` | 31.15 KB | Evaluation Result | Yes (Model Comparison & Error Analysis) | Fine-tuning validation metrics, training history, evidence classification audits, and OCR support diagnostics. |
| `Research\BharatDocAI\outputs\metrics\llm_extraction\qwen3_few_shot_document_metrics.csv` | `.csv` | 2.38 KB | Evaluation Result | Yes (Model Comparison & Research Dashboard) | LLM extraction baseline metric summaries, field accuracy tables, and error logs. |
| `Research\BharatDocAI\outputs\metrics\llm_extraction\qwen3_few_shot_field_metrics.csv` | `.csv` | 26.29 KB | Evaluation Result | Yes (Model Comparison & Research Dashboard) | LLM extraction baseline metric summaries, field accuracy tables, and error logs. |
| `Research\BharatDocAI\outputs\metrics\llm_extraction\qwen3_few_shot_summary.json` | `.json` | 349 B | Evaluation Result | Yes (Model Comparison & Research Dashboard) | LLM extraction baseline metric summaries, field accuracy tables, and error logs. |
| `Research\BharatDocAI\outputs\metrics\llm_extraction\qwen3_ocr_recoverability_analysis.csv` | `.csv` | 43.65 KB | Evaluation Result | Yes | Evaluation metrics file. |
| `Research\BharatDocAI\outputs\metrics\llm_extraction\qwen3_zero_shot_baseline_summary.json` | `.json` | 562 B | Evaluation Result | Yes (Model Comparison & Research Dashboard) | LLM extraction baseline metric summaries, field accuracy tables, and error logs. |
| `Research\BharatDocAI\outputs\metrics\llm_extraction\qwen3_zero_shot_errors.csv` | `.csv` | 29.79 KB | Evaluation Result | Yes (Model Comparison & Research Dashboard) | LLM extraction baseline metric summaries, field accuracy tables, and error logs. |
| `Research\BharatDocAI\outputs\metrics\llm_extraction\zero_shot_vs_few_shot_comparison.csv` | `.csv` | 282 B | Evaluation Result | Yes (Model Comparison & Research Dashboard) | LLM extraction baseline metric summaries, field accuracy tables, and error logs. |
| `Research\BharatDocAI\outputs\metrics\ocr_benchmark_summary.txt` | `.txt` | 2.87 KB | Evaluation Result | Yes (Research Dashboard - OCR Benchmark Section) | Aggregated benchmark comparisons across EasyOCR, PaddleOCR, and Tesseract by language, corruption, and severity. |
| `Research\BharatDocAI\outputs\metrics\ocr_engine_comparison.csv` | `.csv` | 498 B | Evaluation Result | Yes (Research Dashboard - OCR Benchmark Section) | Aggregated benchmark comparisons across EasyOCR, PaddleOCR, and Tesseract by language, corruption, and severity. |
| `Research\BharatDocAI\outputs\metrics\overall_document_level_statistical_summary.csv` | `.csv` | 521 B | Evaluation Result | Yes | Evaluation metrics file. |
| `Research\BharatDocAI\outputs\metrics\paddleocr_by_corruption.csv` | `.csv` | 458 B | Evaluation Result | Yes (Research Dashboard - Drill-down views) | Detailed engine-level evaluation metrics for paddleocr_by_corruption. |
| `Research\BharatDocAI\outputs\metrics\paddleocr_by_language.csv` | `.csv` | 228 B | Evaluation Result | Yes (Research Dashboard - Drill-down views) | Detailed engine-level evaluation metrics for paddleocr_by_language. |
| `Research\BharatDocAI\outputs\metrics\paddleocr_by_severity.csv` | `.csv` | 293 B | Evaluation Result | Yes (Research Dashboard - Drill-down views) | Detailed engine-level evaluation metrics for paddleocr_by_severity. |
| `Research\BharatDocAI\outputs\metrics\paddleocr_detailed_metrics.csv` | `.csv` | 167.60 KB | Evaluation Result | Yes (Research Dashboard - Drill-down views) | Detailed engine-level evaluation metrics for paddleocr_detailed_metrics. |
| `Research\BharatDocAI\outputs\metrics\paddleocr_language_corruption.csv` | `.csv` | 1.41 KB | Evaluation Result | Yes (Research Dashboard - Drill-down views) | Detailed engine-level evaluation metrics for paddleocr_language_corruption. |
| `Research\BharatDocAI\outputs\metrics\paddleocr_language_severity.csv` | `.csv` | 907 B | Evaluation Result | Yes (Research Dashboard - Drill-down views) | Detailed engine-level evaluation metrics for paddleocr_language_severity. |
| `Research\BharatDocAI\outputs\metrics\paddleocr_metrics.csv` | `.csv` | 121 B | Evaluation Result | Yes (Research Dashboard - Drill-down views) | Detailed engine-level evaluation metrics for paddleocr_metrics. |
| `Research\BharatDocAI\outputs\metrics\paired_bootstrap_comparisons.csv` | `.csv` | 2.76 KB | Evaluation Result | Yes (Research Dashboard - Statistical Analysis Section) | Document-level paired bootstrap confidence intervals and statistical tests (CER/WER comparisons across OCR engines and languages). |
| `Research\BharatDocAI\outputs\metrics\phase0_ocr_metrics.csv` | `.csv` | 189 B | Evaluation Result | Yes | Evaluation metrics file. |
| `Research\BharatDocAI\outputs\metrics\tesseract_clean_metrics.csv` | `.csv` | 9.65 KB | Evaluation Result | Yes (Research Dashboard - Drill-down views) | Detailed engine-level evaluation metrics for tesseract_clean_metrics. |
| `Research\BharatDocAI\outputs\metrics\tesseract_degraded_by_corruption.csv` | `.csv` | 392 B | Evaluation Result | Yes (Research Dashboard - Drill-down views) | Detailed engine-level evaluation metrics for tesseract_degraded_by_corruption. |
| `Research\BharatDocAI\outputs\metrics\tesseract_degraded_by_language.csv` | `.csv` | 227 B | Evaluation Result | Yes (Research Dashboard - Drill-down views) | Detailed engine-level evaluation metrics for tesseract_degraded_by_language. |
| `Research\BharatDocAI\outputs\metrics\tesseract_degraded_by_severity.csv` | `.csv` | 228 B | Evaluation Result | Yes (Research Dashboard - Drill-down views) | Detailed engine-level evaluation metrics for tesseract_degraded_by_severity. |
| `Research\BharatDocAI\outputs\metrics\tesseract_degraded_detailed_metrics.csv` | `.csv` | 161.90 KB | Evaluation Result | Yes (Research Dashboard - Drill-down views) | Detailed engine-level evaluation metrics for tesseract_degraded_detailed_metrics. |
| `Research\BharatDocAI\outputs\metrics\tesseract_degraded_language_corruption.csv` | `.csv` | 1.19 KB | Evaluation Result | Yes (Research Dashboard - Drill-down views) | Detailed engine-level evaluation metrics for tesseract_degraded_language_corruption. |
| `Research\BharatDocAI\outputs\metrics\tesseract_degraded_language_severity.csv` | `.csv` | 694 B | Evaluation Result | Yes (Research Dashboard - Drill-down views) | Detailed engine-level evaluation metrics for tesseract_degraded_language_severity. |
| `Research\BharatDocAI\outputs\metrics\tesseract_psm_diagnostic.csv` | `.csv` | 1 B | Evaluation Result | Yes (Research Dashboard - Drill-down views) | Detailed engine-level evaluation metrics for tesseract_psm_diagnostic. |
| `Research\BharatDocAI\outputs\metrics\three_engine_corruption_comparison.csv` | `.csv` | 1.44 KB | Evaluation Result | Yes (Research Dashboard - OCR Benchmark Section) | Aggregated benchmark comparisons across EasyOCR, PaddleOCR, and Tesseract by language, corruption, and severity. |
| `Research\BharatDocAI\outputs\metrics\three_engine_detailed_comparison.csv` | `.csv` | 545.60 KB | Evaluation Result | Yes (Research Dashboard - OCR Benchmark Section) | Aggregated benchmark comparisons across EasyOCR, PaddleOCR, and Tesseract by language, corruption, and severity. |
| `Research\BharatDocAI\outputs\metrics\three_engine_language_comparison.csv` | `.csv` | 711 B | Evaluation Result | Yes (Research Dashboard - OCR Benchmark Section) | Aggregated benchmark comparisons across EasyOCR, PaddleOCR, and Tesseract by language, corruption, and severity. |
| `Research\BharatDocAI\outputs\metrics\three_engine_language_corruption.csv` | `.csv` | 3.61 KB | Evaluation Result | Yes (Research Dashboard - OCR Benchmark Section) | Aggregated benchmark comparisons across EasyOCR, PaddleOCR, and Tesseract by language, corruption, and severity. |
| `Research\BharatDocAI\outputs\metrics\three_engine_language_severity.csv` | `.csv` | 2.22 KB | Evaluation Result | Yes (Research Dashboard - OCR Benchmark Section) | Aggregated benchmark comparisons across EasyOCR, PaddleOCR, and Tesseract by language, corruption, and severity. |
| `Research\BharatDocAI\outputs\metrics\three_engine_overall_comparison.csv` | `.csv` | 236 B | Evaluation Result | Yes (Research Dashboard - OCR Benchmark Section) | Aggregated benchmark comparisons across EasyOCR, PaddleOCR, and Tesseract by language, corruption, and severity. |
| `Research\BharatDocAI\outputs\metrics\three_engine_severity_comparison.csv` | `.csv` | 925 B | Evaluation Result | Yes (Research Dashboard - OCR Benchmark Section) | Aggregated benchmark comparisons across EasyOCR, PaddleOCR, and Tesseract by language, corruption, and severity. |
| `Research\BharatDocAI\outputs\metrics\unified_ocr_evaluation.csv` | `.csv` | 814 B | Evaluation Result | Yes | Evaluation metrics file. |
| `Research\BharatDocAI\outputs\metrics\zero_shot_document_metrics.csv` | `.csv` | 738 B | Evaluation Result | Yes (Model Comparison & Research Dashboard) | LLM extraction baseline metric summaries, field accuracy tables, and error logs. |
| `Research\BharatDocAI\outputs\metrics\zero_shot_engine_metrics.csv` | `.csv` | 412 B | Evaluation Result | Yes (Model Comparison & Research Dashboard) | LLM extraction baseline metric summaries, field accuracy tables, and error logs. |
| `Research\BharatDocAI\outputs\metrics\zero_shot_field_metrics.csv` | `.csv` | 2.24 KB | Evaluation Result | Yes (Model Comparison & Research Dashboard) | LLM extraction baseline metric summaries, field accuracy tables, and error logs. |
| `Research\BharatDocAI\outputs\metrics\zero_shot_language_metrics.csv` | `.csv` | 348 B | Evaluation Result | Yes (Model Comparison & Research Dashboard) | LLM extraction baseline metric summaries, field accuracy tables, and error logs. |
| `Research\BharatDocAI\outputs\metrics\zero_shot_summary.csv` | `.csv` | 202 B | Evaluation Result | Yes (Model Comparison & Research Dashboard) | LLM extraction baseline metric summaries, field accuracy tables, and error logs. |

---

## 8. Document Images and Visualization Assets

| Artifact Path / Directory | Type | Item Count | Total Size | Artifact Category | Necessary for App? | Description |
|---|---|---|---|---|---|---|
| `Research\BharatDocAI\data\raw_documents\*.png` | `.png` | 93 files | 3.12 MB | Dataset (Raw Images) | Yes (Sample Document Previews) | 90 synthetic clean document images (English, Hindi, Kannada across Invoices, Applications, Certificates) + 3 Phase 0 multilingual invoices. |
| `Research\BharatDocAI\data\processed\degradation_manifest.csv` | `.csv` | 1 file (1,440 records) | 288.62 KB | Dataset Manifest | Yes (Research Dashboard - Dataset Overview) | Full metadata index of all 1,440 image records mapping corruption types, severities, parameters, and paths. |
| `Research\BharatDocAI\data\processed\degraded\train\` | `.png` | 810 files (54 docs x 15 variants) | 399.75 MB | Dataset (Degraded Images) | No (Research archive) | Training split degraded images across blur, noise, rotation, low contrast, and JPEG compression. |
| `Research\BharatDocAI\data\processed\degraded\validation\` | `.png` | 270 files (18 docs x 15 variants) | 130.66 MB | Dataset (Degraded Images) | No (Research archive) | Validation split degraded images across 5 corruptions and 3 severities. |
| `Research\BharatDocAI\data\processed\degraded\test\` | `.png` | 270 files (18 docs x 15 variants) | 136.89 MB | Dataset (Degraded Images) | Optional (Testing degradation in live OCR) | Test split degraded images across 5 corruptions and 3 severities. |
| `Research\BharatDocAI\outputs\visualizations\corruption_wise_comparison.png` | `.png` | 1 file | 192.36 KB | Visualization Asset | Yes (Static Fallback Visuals) | Pre-rendered benchmark chart: Corruption Wise Comparison. |
| `Research\BharatDocAI\outputs\visualizations\language_wise_comparison.png` | `.png` | 1 file | 109.57 KB | Visualization Asset | Yes (Static Fallback Visuals) | Pre-rendered benchmark chart: Language Wise Comparison. |
| `Research\BharatDocAI\outputs\visualizations\overall_ocr_comparison.png` | `.png` | 1 file | 91.90 KB | Visualization Asset | Yes (Static Fallback Visuals) | Pre-rendered benchmark chart: Overall Ocr Comparison. |
| `Research\BharatDocAI\outputs\visualizations\severity_wise_comparison.png` | `.png` | 1 file | 168.93 KB | Visualization Asset | Yes (Static Fallback Visuals) | Pre-rendered benchmark chart: Severity Wise Comparison. |
| `Research\BharatDocAI\outputs\metrics\zero_shot_language_accuracy.png` | `.png` | 1 file | 78.15 KB | Visualization Asset | Yes (Static Fallback Visuals) | Pre-rendered chart of zero-shot accuracy by language. |

---

## 9. Potentially Problematic, Anomalous, or Incomplete Artifacts

1. **Experiment 2 Four-Way Comparison Prediction Anomaly (`experiment2_lora_predictions.jsonl`)**:
   - All 54 test set records in `experiment2_lora_predictions.jsonl` are 100% character-for-character identical in both `prediction` dictionary and `raw_response` text to `zero_shot_predictions.jsonl`.
   - However, the adapter model `models/checkpoints/qwen3_lora_experiment2/adapter_model.safetensors` contains valid, trained weights that differ from Experiment 1 (SHA-256: `74bcd93b...` vs `431efc94...`).
   - In `BharatDoc_AI_Phase_2_2_Experiment2_Training.ipynb` (Cell 27/28), `base_model = model.unload()` was invoked before attaching Experiment 2, or adapter loading in PEFT may have failed silently or evaluated on the unloaded base model.
   - In addition, an earlier file `experiments/finetuning/experiment2_test_predictions.jsonl` contains 53/54 identical outputs to zero-shot, differing only on 1 record (kannada date format).
   - **Conclusion:** Experiment 2 results CANNOT be claimed as showing evidence-aware fine-tuning efficacy without marking them as provisional and requiring verification.

2. **Empty Diagnostic File (`outputs/metrics/tesseract_psm_diagnostic.csv`)**:
   - Size is 1 byte (a single newline character `\n`). It contains no PSM diagnostic data.

3. **Partial Phase 0 OCR File (`data/ocr_outputs/paddleocr_results.json`)**:
   - Contains only `english` and `hindi` keys; `kannada` is omitted in this early prototype file (not an issue for main benchmark as full dataset has complete 1,440 Kannada records).

4. **Empty Source Code and Reports Stubs in Research Directory (`Research/BharatDocAI/src/*`, `reports/`)**:
   - The directories `src/data_generation`, `src/evaluation`, `src/extraction`, `src/ocr`, and `reports/` exist as empty folders because the research pipeline was executed inside the Colab `.ipynb` notebooks rather than external `.py` scripts.

