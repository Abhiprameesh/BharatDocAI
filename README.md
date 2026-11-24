# BharatDoc AI: Multilingual AI-Powered Document Intelligence System

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![PEFT / LoRA](https://img.shields.io/badge/tuning-QLoRA-green.svg)](https://github.com/huggingface/peft)
[![License: Academic](https://img.shields.io/badge/license-Academic%20Research-lightgrey.svg)]()
[![Zero-GPU Ready](https://img.shields.io/badge/deployment-Zero--GPU%20Ready-success.svg)]()

**BharatDoc AI** is a research-backed multilingual document information extraction system designed for Indian administrative and commercial documents. It investigates whether parameter-efficient fine-tuning (QLoRA) of a compact open-source Large Language Model (`Qwen/Qwen3-4B`) enhances structured field extraction from noisy, degraded OCR transcriptions compared with zero-shot and few-shot in-context learning.

---

## 1. Project Overview

BharatDoc AI addresses the real-world challenge of automated information extraction from noisy, degraded document scans in linguistically diverse Indian administrative settings. Indian documents present complex typography, non-Latin scripts, variable layouts, and pervasive physical degradations (blur, sensor noise, rotation, low contrast, and compression artifacts).

The project is structured as an interactive academic research platform providing:
1. **Interactive Document Extraction Pipeline:** Document upload, image enhancement, multi-engine OCR, human-in-the-loop transcription editing, and structured JSON parsing.
2. **Model Comparison Suite:** Four-way comparative evaluation across Zero-Shot, Few-Shot, and LoRA adapters with granular drill-downs by language, category, field type, and OCR engine.
3. **Comprehensive Research Dashboard:** Publication-grade visualizations covering dataset architecture, OCR engine benchmarks, paired document-level bootstrap confidence intervals, error recoverability diagnostics, and training trajectories.
4. **Research Transparency & Reproducibility:** Rigorous separation between verified historical research artifacts, live OCR execution, and live LLM inference.

---

## 2. Core Research Objective

> **Central Research Question:**  
> *"Does LoRA fine-tuning of Qwen3-4B improve structured field extraction from noisy OCR text in English, Hindi, and Kannada compared with zero-shot and few-shot prompting?"*

**Supported Languages & Scripts:**
- **English:** Latin script
- **Hindi (`हिन्दी`):** Devanagari script
- **Kannada (`ಕನ್ನಡ`):** Kannada script

**Document Categories & Schemas:**
- **`invoice` (Invoices / Bills):** `name`, `date`, `document_id`, `address`, `amount`
- **`application` (Application Forms):** `name`, `date`, `document_id`, `address`, `phone`
- **`certificate` (Certificates / Marksheets):** `name`, `date`, `document_id`, `institution`, `grade`

---

## 3. System Architecture

The application is decoupled into modular layers:

```
┌────────────────────────────────────────────────────────┐
│                   Streamlit Web UI                     │
│  (app.py, 1_Extraction, 2_Comparison, 3_Dashboard, 4_About)  │
└───────────────────────────┬────────────────────────────┘
                            │
       ┌────────────────────┼────────────────────┐
       ▼                    ▼                    ▼
┌──────────────┐   ┌─────────────────┐   ┌─────────────────┐
│  Document    │   │      Image      │   │   OCR Engine    │
│  Processor   │──▶│  Preprocessor   │──▶│     Factory     │
│  (PDF/PNG)   │   │(Otsu/Denoise/raw│   │(Easy/Paddle/Tess│
└──────────────┘   └─────────────────┘   └────────┬────────┘
                                                  │
                                                  ▼
                                         ┌─────────────────┐
                                         │ Human-in-the-   │
                                         │ Loop Text Editor│
                                         └────────┬────────┘
                                                  │
                                                  ▼
                                         ┌─────────────────┐
                                         │Extraction Engine│
                                         │  (Qwen3-4B /    │
                                         │Benchmark Lookup)│
                                         └────────┬────────┘
                                                  │
                                                  ▼
                                         ┌─────────────────┐
                                         │Schema Validator │
                                         │& String Normalizer
                                         └─────────────────┘
```

- **Zero-GPU Startup:** Research dashboards and statistical evaluations run instantaneously on CPU without model weights or GPU allocation.
- **Lazy Loading:** Heavy ML frameworks (`torch`, `transformers`, `peft`) are only imported when live inference is triggered.
- **Historical Benchmark Mode:** For research documents, verified OCR and LLM outputs are retrieved from the research archive, guaranteeing 100% scientific reproducibility at zero compute cost.

---

## 4. Dataset Description

The research benchmark consists of:
- **90 Unique Ground-Truth Source Documents:**
  - 3 languages $\times$ 3 categories $\times$ 10 documents per combination.
  - **Train split:** 54 documents (60%)
  - **Validation split:** 18 documents (20%)
  - **Test split:** 18 documents (20%)
- **Controlled Image Degradations:**
  - 5 corruption types: Gaussian Blur, Gaussian Noise, Low Contrast, Rotation, JPEG Compression.
  - 3 severity levels: Mild, Moderate, Severe.
  - 15 degraded variants + 1 clean image = 16 image records per source document.
- **Total Image Records:** **1,440 records** (90 clean + 1,350 degraded).
- **Critical Methodological Distinction:** The 1,440 image records originate from 90 source documents. They are **not** 1,440 independent documents.

---

## 5. OCR Benchmarking Pipeline

Three OCR engines were benchmarked across the 1,440 synthetic document images:

| Rank | Engine | Overall CER | Overall WER | Character Accuracy | Strengths & Observations |
|---|---|---:|---:|---:|---|
| 🥇 **1** | **EasyOCR** | **15.57%** | **45.03%** | **84.43%** | Benchmark leader overall; highest accuracy on Kannada and English. |
| 🥈 **2** | **PaddleOCR** | **23.75%** | **56.34%** | **76.25%** | Lowest CER on Hindi (15.99%); severe degradation on Kannada (43.36% CER). |
| 🥉 **3** | **Tesseract** | **29.22%** | **55.43%** | **70.78%** | Struggled with complex Indic conjunct consonants and character ligatures. |

---

## 6. Information Extraction Methods

Structured field extraction evaluates four configurations of `Qwen/Qwen3-4B`:

1. **Zero-Shot Prompting:** Standard baseline prompt with extraction guidelines and strict JSON instructions.
2. **Few-Shot In-Context Prompting:** Baseline prompt augmented with 9 clean training demonstrations (1 exemplar per language $\times$ category pair).
3. **Experiment 1 (Standard QLoRA):** Parameter-efficient fine-tuning ($r=16, \alpha=32$, dropout $0.05$) supervised on full ground-truth JSON targets across 54 training documents.
4. **Experiment 2 (Evidence-Aware LoRA):** Supervised on evidence-filtered targets (predicting empty strings when OCR drops field evidence) to mitigate hallucination. *(Status: Marked Provisional / Requiring Verification)*.

---

## 7. Evaluation Protocol & Metrics

- **String Normalization Protocol (`normalize_field_value`):**
  - Unicode NFC canonical decomposition/composition.
  - Zero-width non-joiner (`\u200c`) and zero-width joiner (`\u200d`) removal.
  - Whitespace collapse and casefolding.
  - Punctuation and separator stripping for numeric dates, phone numbers, amounts, and document IDs.
- **Field-Level Exact Match (%):** Percentage of schema fields whose normalized prediction exactly matches ground truth.
- **Document-Level Exact Match (%):** Percentage of documents where **all 5 fields** match ground truth simultaneously.
- **Master Test Set Denominator:** 18 unique source test documents processed through all 3 OCR engines = **54 test records** (270 total field instances).

---

## 8. Main Experimental Findings

### Master Test Set Results (54 Records / 270 Fields)

| Method | Approach | JSON Validity | Schema Validity | Field Accuracy (%) | Document Match (%) |
|---|---|---:|---:|---:|---:|
| **Zero-Shot Qwen3-4B** | Baseline Prompting | 100.0% (54/54) | 100.0% (54/54) | 75.19% (203/270) | 35.19% (19/54) |
| **Few-Shot Qwen3-4B (🏆 Winner)** | In-Context Learning | 100.0% (54/54) | 100.0% (54/54) | **83.33%** (225/270) | **61.11%** (33/54) |
| **Experiment 1 LoRA** | Standard QLoRA | 100.0% (54/54) | 100.0% (54/54) | 69.63% (188/270) | 31.48% (17/54) |
| **Experiment 2 LoRA\*** | Evidence-Aware LoRA | 100.0% (54/54) | 100.0% (54/54) | 75.19% (203/270) | 35.19% (19/54) |

\* *Provisional status: Test predictions in Colab matched zero-shot outputs.*

### Paired Comparison Findings

1. **Record-Level (54 OCR Test Records):**
   - **Few-Shot vs Zero-Shot:** Few-Shot scored higher on 23 records, identical on 30, lower on 1 (+22 net fields).
   - **Exp 1 LoRA vs Zero-Shot:** LoRA scored higher on 0 records, identical on 41, lower on 13 (-15 net fields).
   - **Few-Shot vs Exp 1 LoRA:** Few-Shot scored higher on 30 records, identical on 24, lower on 0 (+37 net fields).
2. **Source-Document Level (18 Unique Source Documents, 3 OCR Engines Aggregated):**
   - **Few-Shot vs Zero-Shot:** Few-Shot won on 11 source docs, tied on 7, lost on 0 (+1.222 mean fields/doc).
   - **Exp 1 LoRA vs Zero-Shot:** LoRA won on 0 source docs, tied on 9, lost on 9 (-0.833 mean fields/doc).
   - **Few-Shot vs Exp 1 LoRA:** Few-Shot won on 12 source docs, tied on 6, lost on 0 (+2.056 mean fields/doc).

### Error Analysis & Recoverability Insights
- **Root Cause:** In the validation diagnostic benchmark (71 total errors), **87.32% (62/71)** of failures were caused by OCR error propagation (ground truth dropped from OCR text).
- **False Abstentions:** **12.68% (9/71)** were false abstentions (OCR text contained valid evidence, but LoRA output empty strings). In all 9 cases, Zero-Shot and Few-Shot extracted the correct value.
- **Over-Abstention Mechanism:** LoRA fine-tuning on noisy OCR text caused the model to develop an excessive bias toward empty-string abstention and disrupted numeric tokenization.

---

## 9. Research Limitations

1. **Synthetic Document Domain:** Documents were synthetically generated from 6 template families. Real-world documents feature complex handwriting, stamps, folds, creases, low-resolution phone cameras, and diverse typographic variations.
2. **Sample Size:** 18 unique source test documents (54 evaluation records across 3 OCR engines). Findings reflect pilot-scale controlled benchmarking and should not be extrapolated to unconstrained industrial production without in-the-wild testing.
3. **Experiment 2 Status:** Maintained as provisional pending further verification of the adapter checkpoint application.

---

## 10. Installation Instructions (Windows PowerShell)

Ensure Python 3.12 is installed on your Windows system.

### Step 1: Clone or Navigate to Repository

```powershell
cd "c:\ALLMLPROJ\MAINMLPROJS\BharatDoc AI"
```

### Step 2: Create and Activate Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Core Dependencies

```powershell
pip install -r requirements.txt
```

---

## 11. Application Launch Instructions

To launch the BharatDoc AI Streamlit application:

```powershell
.\.venv\Scripts\streamlit run app.py
```

The application will start immediately on `http://localhost:8501`. No GPU or model weight downloads are required to view the overview, model comparisons, research dashboards, or test sample extractions.

---

## 12. Optional OCR Dependencies

The application includes `BenchmarkLookupEngine` which provides verified OCR transcriptions for all synthetic documents without installing heavy OCR engines. To enable live OCR on novel uploaded images:

- **EasyOCR:**
  ```powershell
  pip install easyocr torch torchvision
  ```
- **Tesseract OCR:**
  Install Tesseract for Windows from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki), install language packs (`eng`, `hin`, `kan`), and ensure `tesseract.exe` is in your `PATH`.
- **PaddleOCR:**
  ```powershell
  pip install paddlepaddle paddleocr
  ```

---

## 13. CPU-Only Environment Limitations

- **Dashboard & Evaluation:** 100% operational on CPU.
- **Sample Document Extraction:** 100% operational on CPU using verified benchmark records.
- **Live LLM Inference on Novel Uploads:** `Qwen3-4B` in 4-bit NormalFloat (NF4) quantization requires a CUDA-compatible NVIDIA GPU. On CPU-only environments (such as the development Intel Core i5-1340P), live extraction on arbitrary custom uploads is safely blocked with an informative hardware notice rather than crashing the system.

---

## 14. Project Folder Structure

```
BharatDoc AI/
│
├── app.py                             # Main Streamlit application entry point
├── requirements.txt                   # Application Python dependencies
├── README.md                          # Full research documentation
├── .gitignore                         # Git exclusion rules
├── research_archive_inventory.md      # Inventory of 1,563 research files
│
├── src/
│   ├── __init__.py
│   ├── config.py                      # Relative paths, category schemas, language mappings
│   ├── utils.py                       # Header, alert banners, export helpers, CSS injector
│   ├── document_processor.py          # PNG/JPG/PDF validation, PDF page rasterization
│   ├── image_preprocessing.py         # Pillow/NumPy grayscale, Otsu binarization, denoise
│   ├── ocr_engine.py                  # EasyOCR, PaddleOCR, Tesseract & BenchmarkLookup
│   ├── prompt_templates.py            # System prompts & 9 verified clean training demos
│   ├── extraction_engine.py           # Robust JSON parsing, Markdown fences, schema validator
│   ├── model_loader.py                # Hardware detection, Zero-GPU safety, benchmark cache
│   ├── evaluation.py                  # normalize_field_value, ModelEvaluator, 4-way metrics
│   └── research_analysis.py           # Metrics loader, two-level paired comparisons
│
├── pages/
│   ├── 1_Document_Extraction.py       # End-to-end extraction pipeline with human-in-the-loop editor
│   ├── 2_Model_Comparison.py          # 4-way model comparison, two-level paired difference matrix
│   ├── 3_Research_Dashboard.py        # 7-tab comprehensive academic research dashboard
│   └── 4_About_Project.py             # Methodology, schemas, and scientific integrity disclosures
│
├── assets/
│   └── css/style.css                  # Custom typography, badge, and metric card styling
│
├── tests/
│   ├── test_startup.py                # Configuration and data loader startup tests
│   ├── test_ocr.py                    # Preprocessing, PDF extraction, and OCR factory tests
│   ├── test_extraction.py             # JSON parser, schema validation, and prompt tests
│   ├── test_evaluation.py             # Normalization, single-record, and 4-way evaluation tests
│   ├── test_dashboard_data.py         # Research dashboard loader, metrics, and loss curve tests
│   ├── test_statistical_and_workflow_integrity.py # Two-level paired comparisons, error scopes, end-to-end
│   └── test_pages_smoke.py            # Full headless smoke tests for all 5 Streamlit pages
│
└── Research/                          # Original research archive (Preserved & Untouched)
    └── BharatDocAI/
        ├── Notebooks/                 # Colab research notebooks (Phases 0 - 2.2)
        ├── data/                      # 90 clean documents, 1,350 degraded images, OCR outputs
        ├── models/                    # LoRA adapters (Experiment 1 and 2)
        ├── experiments/               # Four-way comparison predictions & manifests
        └── outputs/                   # Benchmark CSVs, bootstrap CIs, and visualizations
```

---

## 📜 Citation & Academic Use

If referencing this research, please cite:
```bibtex
@misc{bharatdoc_ai_2026,
  title={BharatDoc AI: Investigating LoRA Fine-Tuning of Small Language Models for Multilingual Information Extraction from Degraded Indian Documents},
  author={BharatDoc AI Research Team},
  year={2026},
  note={Experimental study on Qwen3-4B across English, Hindi, and Kannada administrative documents}
}
```
