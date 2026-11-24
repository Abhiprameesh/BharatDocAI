"""
Page 2: Model Comparison
Head-to-head evaluation of Zero-Shot, Few-Shot, Experiment 1 LoRA, and Experiment 2 LoRA.
Recalculates exact research metrics dynamically from 54 clean test records.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import CSS_FILE
from src.utils import load_css, render_header, render_provisional_badge, export_csv_button
from src.research_analysis import ResearchDataLoader
from src.evaluation import normalize_field_value

st.set_page_config(page_title="Model Comparison - BharatDoc AI", page_icon="⚖️", layout="wide")
load_css(CSS_FILE)

render_header(
    title="Model Comparison: Extraction Methods",
    subtitle="Rigorous comparative evaluation of 4 extraction methods on 54 identical clean test records.",
    badge="Phase 5 Results"
)

# Research Question
st.markdown(
    """
    > **Core Research Question Addressed:**  
    > *"Does LoRA fine-tuning of Qwen3-4B improve structured field extraction from noisy OCR text in English, Hindi, and Kannada compared with zero-shot and few-shot prompting?"*
    """
)

# Render Provisional Warning for Experiment 2
render_provisional_badge()

# Load Dynamic Evaluation Data
summary_df = ResearchDataLoader.evaluate_four_way_comparison()

if not summary_df.empty:
    # Top KPI Metrics Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">83.33%</div>
                <div class="metric-label">Few-Shot Accuracy</div>
                <div class="metric-subtitle">Highest Overall Field Accuracy</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with kpi2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">75.19%</div>
                <div class="metric-label">Zero-Shot Baseline</div>
                <div class="metric-subtitle">Qwen3-4B Baseline Performance</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with kpi3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">69.63%</div>
                <div class="metric-label">Exp 1 Standard LoRA</div>
                <div class="metric-subtitle">-5.56 pp vs Zero-Shot Baseline</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with kpi4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">75.19%*</div>
                <div class="metric-label">Exp 2 LoRA (Provisional)</div>
                <div class="metric-subtitle">*Identical to Zero-Shot (Unverified)</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### 📋 Four-Way Overall Extraction Performance")
    
    col_tbl, col_export = st.columns([4, 1])
    with col_tbl:
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    with col_export:
        st.write("")
        st.write("")
        export_csv_button(summary_df, filename="four_way_extraction_summary.csv", label="📥 Export Table (CSV)")
        
    st.caption(
        "Evaluated on 54 clean test records (18 source documents × 3 OCR engines). "
        "Field-Level Exact Match requires exact normalized string equality. "
        "Document-Level Exact Match requires every required schema field in the document to match."
    )

    st.markdown("---")

    # Plotly Visual Comparison
    st.subheader("Visual Model Comparison")
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        fig_bar = px.bar(
            summary_df,
            x="Model / Method",
            y=["Field-Level Exact Match (%)", "Document-Level Exact Match (%)"],
            barmode="group",
            title="Field-Level Accuracy vs Document Exact Match (%)",
            color_discrete_sequence=["#1E3A8A", "#0D9488"],
            labels={"value": "Accuracy (%)", "variable": "Metric"}
        )
        fig_bar.update_layout(yaxis_range=[0, 100], legend_title_text="Evaluation Metric")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_c2:
        # Paired differences summary
        st.markdown("#### Paired Comparisons: Record vs Source-Doc Level")
        sub_pair_rec, sub_pair_doc = st.tabs(["54 OCR Records", "18 Source Documents"])
        with sub_pair_rec:
            paired_rec_df = ResearchDataLoader.get_paired_comparisons_summary()
            st.dataframe(paired_rec_df, use_container_width=True, hide_index=True)
            st.caption(
                "Evaluated across all 54 OCR-engine records (5 fields each). "
                "Few-Shot scored higher on 23 records, identical on 30, and lower on only 1 record relative to Zero-Shot."
            )
        with sub_pair_doc:
            paired_doc_df = ResearchDataLoader.get_source_document_paired_comparisons_summary()
            st.dataframe(paired_doc_df, use_container_width=True, hide_index=True)
            st.caption(
                "Aggregated across 18 unique source documents (15 pooled fields across 3 OCR engines per document). "
                "Few-Shot won on 11 source documents, tied 7, and lost 0 relative to Zero-Shot."
            )

    st.markdown("---")

    # Detailed Breakdown Tabs
    st.subheader("Detailed Performance Breakdowns")
    tab_lang, tab_cat, tab_engine, tab_field, tab_inspector = st.tabs([
        "🌐 By Language",
        "📄 By Category",
        "🔍 By OCR Engine",
        "🏷️ By Individual Field",
        "🔎 Record-by-Record Inspector"
    ])

    # Tab 1: Language Breakdown
    with tab_lang:
        lang_df = ResearchDataLoader.get_language_breakdown()
        col_l1, col_l2 = st.columns([1.2, 1])
        with col_l1:
            fig_l = px.bar(
                lang_df,
                x="Language",
                y="Field Accuracy (%)",
                color="Model",
                barmode="group",
                title="Field Accuracy Across Languages (English, Hindi, Kannada)",
                color_discrete_sequence=["#2563EB", "#0D9488", "#E11D48", "#F59E0B"]
            )
            fig_l.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(fig_l, use_container_width=True)
        with col_l2:
            st.markdown("#### Language Performance Table")
            st.dataframe(lang_df, use_container_width=True, hide_index=True)
            export_csv_button(lang_df, filename="language_breakdown.csv", label="Export Language Data")

    # Tab 2: Category Breakdown
    with tab_cat:
        cat_df = ResearchDataLoader.get_category_breakdown()
        col_cat1, col_cat2 = st.columns([1.2, 1])
        with col_cat1:
            fig_cat = px.bar(
                cat_df,
                x="Category",
                y="Field Accuracy (%)",
                color="Model",
                barmode="group",
                title="Field Accuracy Across Document Categories",
                color_discrete_sequence=["#2563EB", "#0D9488", "#E11D48", "#F59E0B"]
            )
            fig_cat.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(fig_cat, use_container_width=True)
        with col_cat2:
            st.markdown("#### Category Performance Table")
            st.dataframe(cat_df, use_container_width=True, hide_index=True)
            export_csv_button(cat_df, filename="category_breakdown.csv", label="Export Category Data")

    # Tab 3: OCR Engine Breakdown
    with tab_engine:
        eng_df = ResearchDataLoader.get_engine_breakdown()
        col_eng1, col_eng2 = st.columns([1.2, 1])
        with col_eng1:
            fig_eng = px.bar(
                eng_df,
                x="Ocr_Engine",
                y="Field Accuracy (%)",
                color="Model",
                barmode="group",
                title="Extraction Accuracy Conditioned on Upstream OCR Engine",
                color_discrete_sequence=["#2563EB", "#0D9488", "#E11D48", "#F59E0B"]
            )
            fig_eng.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(fig_eng, use_container_width=True)
        with col_eng2:
            st.markdown("#### OCR Engine Conditioned Table")
            st.dataframe(eng_df, use_container_width=True, hide_index=True)
            st.caption("Shows extraction robustness against EasyOCR, PaddleOCR, and Tesseract transcriptions.")

    # Tab 4: Field-by-Field Breakdown
    with tab_field:
        field_df = ResearchDataLoader.get_field_breakdown()
        fig_f = px.bar(
            field_df,
            x="Field",
            y="Accuracy (%)",
            color="Model",
            barmode="group",
            title="Field-Level Accuracy Across All 8 Schema Fields",
            color_discrete_sequence=["#2563EB", "#0D9488", "#E11D48", "#F59E0B"]
        )
        fig_f.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(fig_f, use_container_width=True)
        st.dataframe(field_df, use_container_width=True, hide_index=True)

    # Tab 5: Record-by-Record Inspector
    with tab_inspector:
        st.subheader("Record-by-Record Side-by-Side Inspector")
        zs_all = ResearchDataLoader.load_predictions("zero_shot")
        fs_all = ResearchDataLoader.load_predictions("few_shot")
        e1_all = ResearchDataLoader.load_predictions("experiment1_lora")
        e2_all = ResearchDataLoader.load_predictions("experiment2_lora")
        
        record_labels = [
            f"{r['document_id']} ({r['language'].title()} {r['category'].title()} · {r['ocr_engine'].title()})"
            for r in zs_all
        ]
        
        sel_idx = st.selectbox(
            "Select Test Record to Inspect",
            options=range(len(record_labels)),
            format_func=lambda i: record_labels[i]
        )
        
        r_zs = zs_all[sel_idx]
        r_fs = fs_all[sel_idx]
        r_e1 = e1_all[sel_idx]
        r_e2 = e2_all[sel_idx]
        gt = r_zs["ground_truth"]
        
        st.markdown(f"**Document ID:** `{r_zs['document_id']}` | **Language:** `{r_zs['language']}` | **Category:** `{r_zs['category']}` | **OCR Engine:** `{r_zs['ocr_engine']}`")
        
        with st.expander("📄 View OCR Transcription Text", expanded=False):
            st.text(r_zs["ocr_text"])
            
        # Side-by-side comparison table for this record
        row_fields = list(gt.keys())
        comp_records = []
        for f in row_fields:
            exp_val = str(gt.get(f, ""))
            p_zs = str(r_zs.get("prediction", {}).get(f, ""))
            p_fs = str(r_fs.get("prediction", {}).get(f, ""))
            p_e1 = str(r_e1.get("prediction", {}).get(f, ""))
            p_e2 = str(r_e2.get("prediction", {}).get(f, ""))
            
            norm_exp = normalize_field_value(exp_val, f)
            match_zs = "✅" if normalize_field_value(p_zs, f) == norm_exp else "❌"
            match_fs = "✅" if normalize_field_value(p_fs, f) == norm_exp else "❌"
            match_e1 = "✅" if normalize_field_value(p_e1, f) == norm_exp else "❌"
            match_e2 = "✅" if normalize_field_value(p_e2, f) == norm_exp else "❌"
            
            comp_records.append({
                "Field": f,
                "Ground Truth": exp_val,
                "Zero-Shot": f"{match_zs} {p_zs}",
                "Few-Shot": f"{match_fs} {p_fs}",
                "Exp 1 LoRA": f"{match_e1} {p_e1}",
                "Exp 2 LoRA*": f"{match_e2} {p_e2}",
            })
            
        st.dataframe(pd.DataFrame(comp_records), use_container_width=True, hide_index=True)
else:
    st.warning("Prediction files not found. Ensure Research/BharatDocAI directory is present.")
