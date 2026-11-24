"""
Page 4: About BharatDoc AI
Project overview, scientific methodology, experimental design, and academic integrity disclosures.
"""

import streamlit as st
import pandas as pd
from src.config import CSS_FILE, CATEGORY_SCHEMAS, EXP2_STATUS_DETAILS
from src.utils import load_css, render_header

st.set_page_config(page_title="About Project - BharatDoc AI", page_icon="ℹ️", layout="wide")
load_css(CSS_FILE)

render_header(
    title="About BharatDoc AI",
    subtitle="Academic Research & Multilingual Document Intelligence System",
    badge="Documentation"
)

st.markdown(
    """
    ### 🔬 Project Objective & Core Research Question
    **BharatDoc AI** investigates whether parameter-efficient fine-tuning (QLoRA) of a small open-source 
    Large Language Model improves structured information extraction from noisy, degraded OCR text in Indian languages.

    > **Core Research Question:**  
    > *"Does LoRA fine-tuning of Qwen3-4B improve structured field extraction from noisy OCR text in English, Hindi, and Kannada compared with zero-shot and few-shot prompting?"*

    **Key Experimental Dimensions:**
    - **3 Indian Languages:** English (Latin script), Hindi (Devanagari script), and Kannada (Kannada script).
    - **3 Document Categories:** Invoices / Bills, Application Forms, and Certificates / Marksheets.
    - **3 OCR Engines:** EasyOCR (Benchmark Leader), PaddleOCR (Multilingual), and Tesseract OCR.
    - **4 LLM Extraction Methods:**
      1. Zero-Shot Prompting (Qwen3-4B)
      2. Few-Shot In-Context Prompting (9 Clean Demos, 1 per language $\times$ category pair)
      3. Experiment 1: Standard QLoRA Fine-Tuning ($r=16, \alpha=32$, full target supervision)
      4. Experiment 2: Evidence-Aware Loss Supervision (*Provisional*)

    ---

    ### 🏆 Core Research Findings & Empirical Answer
    Across all evaluated statistical dimensions, the empirical answer to the research question is **No**:
    1. **Few-Shot In-Context Learning is the Clear Benchmark Leader:**
       - **Field Accuracy:** **83.33%** (225/270 correct fields)
       - **Exact Document Match:** **61.11%** (33/54 exact documents)
       - Outperformed Zero-Shot by **+8.15%** field accuracy and **+25.93%** exact document match.
       - Outperformed Experiment 1 LoRA by **+13.70%** field accuracy and **+29.63%** exact document match.
    2. **Why Did LoRA Underperform In-Context Learning?**
       - **Over-Abstention Bias:** Fine-tuning on noisy negative examples caused the model to develop an excessive propensity to return empty strings (false abstentions) even when valid text was present in the OCR transcription.
       - **Numeric Tokenization Disruption:** LoRA disrupted tokenization of Indian numeric dates, amounts, and alphanumeric IDs.
    3. **OCR Error Propagation vs LLM Extraction Failures:**
       - **87.32%** of extraction errors stem directly from OCR error propagation (the ground truth was dropped or corrupted by the OCR engine).
       - Only **12.68%** of errors were false abstentions by the LLM where valid OCR evidence existed.

    ---

    ### 📜 Verified Document Extraction Schemas
    All schemas strictly follow the ground truth annotations preserved in the research archive:
    """
)

for cat, fields in CATEGORY_SCHEMAS.items():
    st.markdown(f"- **{cat.capitalize()}:** `{'`, `'.join(fields)}`")

st.markdown(
    f"""
    ---

    ### 📐 System Architecture & Module Pipeline
    BharatDoc AI is built on a clean, decoupled modular architecture:
    ```
    Input Document (PNG/JPG/PDF)
            │
            ▼
    Document Processor (src/document_processor.py)
            │ Validation, Multi-page PDF splitting, RGB conversion
            ▼
    Image Preprocessor (src/image_preprocessing.py)
            │ Grayscale, Contrast boost, Otsu binarization, Median denoise
            ▼
    OCR Engine (src/ocr_engine.py)
            │ EasyOCR / PaddleOCR / Tesseract / Benchmark Lookup Engine
            ▼
    Human-in-the-Loop Editor (Streamlit UI)
            │ User inspects & rectifies transcription errors prior to extraction
            ▼
    Extraction Engine (src/extraction_engine.py & src/prompt_templates.py)
            │ Qwen3-4B backend, Prompt builder, Output parser & Schema validator
            ▼
    Structured JSON Output & Evaluation (src/evaluation.py & src/research_analysis.py)
            │ Field normalization, Paired bootstrap analysis, Research Dashboard
    ```

    ---

    ### ⚠️ Scientific Integrity & Experimental Transparency
    1. **Synthetic Data Disclosure:** All evaluation benchmarks are computed on synthetically rendered documents subjected to controlled digital degradations (Blur, Gaussian Noise, Low Contrast, Rotation, JPEG Compression across Mild, Moderate, and Severe levels). Findings should not be assumed to generalize directly to real-world scanned documents without extensive in-the-wild testing.
    2. **Document-Level Independence:** The 1,440 image records originate from **90 unique ground-truth source documents**. Multiple degraded variants share ground truth and are not treated as independent documents when evaluating generalization.
    3. **Experiment 2 Status:** {EXP2_STATUS_DETAILS}
    4. **Metric Integrity:** We report exact string-normalized field accuracy and document-level exact match. We do not mislabel field accuracy as token-level F1.

    ---

    ### 💻 System Deployment & Zero-GPU Architecture
    - **Zero-GPU Startup:** The Streamlit dashboard and research comparisons launch in standard CPU environments without requiring local GPU hardware.
    - **Lazy Loading & Adapter Isolation:** Heavy PyTorch and PEFT dependencies are only imported when live inference is triggered.
    - **Benchmark Lookup Mode:** For the synthetic research document set, verified historical OCR and LLM outputs are retrieved from the research archive, ensuring 100% scientific reproducibility at zero computational cost.
    """
)
