"""
BharatDoc AI - Main Application Entrypoint
Multilingual AI-Powered Document Intelligence System
"""

import streamlit as st
from pathlib import Path
import pandas as pd

from src.config import (
    CSS_FILE,
    CATEGORY_SCHEMAS,
    CATEGORY_NAMES,
    SUPPORTED_LANGUAGES,
    LANGUAGE_LABELS,
    EXP2_STATUS,
    EXP2_STATUS_DETAILS,
)
from src.utils import load_css, render_header
from src.research_analysis import ResearchDataLoader

# Streamlit Page Config
st.set_page_config(
    page_title="BharatDoc AI - Multilingual Document Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS
load_css(CSS_FILE)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/documents.png", width=64)
    st.title("BharatDoc AI")
    st.caption("Multilingual Document Intelligence System")
    st.markdown("---")
    st.markdown("**Core Research Focus:**")
    st.write(
        "Investigating whether LoRA fine-tuning of open-source Qwen3-4B "
        "improves structured information extraction from noisy OCR text in Indian languages."
    )
    st.markdown("---")
    st.markdown("**Supported Languages:**")
    for lang in SUPPORTED_LANGUAGES:
        st.markdown(f"- {LANGUAGE_LABELS[lang]}")
    st.markdown("---")
    st.info("💡 **Zero-GPU Startup:** Full dashboard & comparison run locally on CPU.")

# Main Page Header
render_header(
    title="BharatDoc AI",
    subtitle="Multilingual AI-Powered Document Intelligence System",
    badge="Research v1.0"
)

# Research Question Callout
st.markdown(
    """
    > **Core Research Question:**  
    > *"Does LoRA fine-tuning of Qwen3-4B improve structured field extraction from noisy OCR text in English, Hindi, and Kannada compared with zero-shot and few-shot prompting?"*
    """
)

# High-Level Metrics (Dataset & Research Scope)
manifest_df = ResearchDataLoader.load_dataset_manifest()
degrad_df = ResearchDataLoader.load_degradation_manifest()
ocr_df = ResearchDataLoader.load_ocr_overall_metrics()
comp_df = ResearchDataLoader.evaluate_four_way_comparison()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-value">90</div>
            <div class="metric-label">Source Documents</div>
            <div class="metric-subtitle">3 Languages · 3 Categories</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-value">1,440</div>
            <div class="metric-label">Total Image Records</div>
            <div class="metric-subtitle">90 Clean · 1,350 Degraded Variants</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-value">3</div>
            <div class="metric-label">OCR Engines</div>
            <div class="metric-subtitle">EasyOCR · PaddleOCR · Tesseract</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-value">4</div>
            <div class="metric-label">Extraction Methods</div>
            <div class="metric-subtitle">Zero-Shot · Few-Shot · LoRA Exp1 & 2</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("### 📋 Executive Research Findings")

col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.markdown("#### Four-Way Extraction Method Summary")
    if not comp_df.empty:
        st.dataframe(
            comp_df[["Model / Method", "Approach", "Field-Level Exact Match (%)", "Document-Level Exact Match (%)", "JSON Validity (%)"]],
            use_container_width=True,
            hide_index=True
        )
        st.caption("Evaluated on the exact same 54 clean test records across 18 unique source documents.")
    else:
        st.info("Loading 4-way comparison predictions...")

with col_right:
    st.markdown("#### OCR Engine Benchmark Leader")
    if not ocr_df.empty:
        st.dataframe(
            ocr_df[["engine", "CER", "WER", "character_accuracy"]],
            use_container_width=True,
            hide_index=True
        )
        st.caption("EasyOCR achieved the lowest overall CER and WER on the synthetic document benchmark.")
    else:
        st.info("Loading OCR benchmark metrics...")

st.markdown("---")

# Navigation Guidance
st.markdown("### 🧭 Application Navigation")
nav1, nav2, nav3, nav4 = st.columns(4)

with nav1:
    st.subheader("1. Extraction")
    st.write("Upload an Indian document (Invoice, Application, Certificate), run OCR, edit text, and extract structured JSON.")

with nav2:
    st.subheader("2. Model Comparison")
    st.write("Detailed 4-way evaluation breakdown by field, language, document category, and OCR engine.")

with nav3:
    st.subheader("3. Research Dashboard")
    st.write("In-depth exploration of OCR degradation curves, bootstrap confidence intervals, and error recoverability.")

with nav4:
    st.subheader("4. About Project")
    st.write("Research methodology, model specifications, scientific integrity disclosures, and project reproducibility.")
