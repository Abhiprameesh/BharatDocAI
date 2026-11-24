"""
Page 3: Research Dashboard
Comprehensive, academic research dashboard presenting experimental findings of BharatDoc AI:
1. Synthetic dataset statistics & architecture.
2. OCR benchmarking across EasyOCR, PaddleOCR, and Tesseract.
3. LLM extraction experiments (Zero-Shot, Few-Shot, Exp 1 LoRA, Exp 2 LoRA [Provisional]).
4. Statistical comparisons & paired bootstrap confidence intervals.
5. Error analysis & OCR recoverability diagnostics.
6. LoRA training trajectories and loss curves.
7. Research methodology, limitations, and reproducibility.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.config import (
    CSS_FILE,
    EXP2_STATUS,
    EXP2_STATUS_DETAILS,
    OCR_ENGINES,
    CATEGORY_NAMES,
    LANGUAGE_LABELS,
)
from src.utils import load_css, render_header, render_provisional_badge, export_csv_button
from src.research_analysis import ResearchDataLoader

# Page Configuration
st.set_page_config(
    page_title="Research Dashboard - BharatDoc AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
load_css(CSS_FILE)

# Header
render_header(
    title="Comprehensive Research Dashboard",
    subtitle="Interactive experimental findings, 3-engine OCR benchmarks, bootstrap confidence intervals, and loss curves.",
    badge="Academic Research"
)

# Seven Comprehensive Academic Tabs
(
    tab_dataset,
    tab_ocr,
    tab_llm,
    tab_stats,
    tab_error,
    tab_training,
    tab_methodology,
) = st.tabs([
    "📁 Dataset Overview",
    "🔍 OCR Benchmarking",
    "🤖 LLM Extraction Benchmark",
    "📈 Statistical Analysis",
    "🔬 Error & Recoverability",
    "📉 LoRA Trajectories",
    "📜 Methodology & Limitations"
])

# ==============================================================================
# TAB 1: DATASET OVERVIEW
# ==============================================================================
with tab_dataset:
    st.subheader("1. Synthetic Dataset Architecture & Multi-Degradation Design")
    
    st.markdown(
        """
        The BharatDoc AI research dataset investigates document information extraction across **3 Indian languages** 
        (*English, Hindi, and Kannada*) and **3 administrative categories** (*Invoices, Applications, and Certificates*).
        
        > **Critical Methodological Distinction:**  
        > The benchmark comprises **90 unique source documents** (54 train, 18 validation, 18 test).  
        > Each document was degraded using **5 synthetic corruption types** at **3 severity levels** (15 degraded variants + 1 clean image = 16 images per source document).  
        > This produces **1,440 total image records**, which must **not** be mistaken for 1,440 independent documents.
        """
    )
    
    # Top KPI Cards
    summary_stats = ResearchDataLoader.get_dataset_summary_metrics()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Unique Source Docs", summary_stats.get("total_unique_source_docs", 90), help="Unique ground truth documents")
    c2.metric("Clean Images", summary_stats.get("clean_images", 90), help="1 clean scan per source doc")
    c3.metric("Degraded Variants", summary_stats.get("degraded_variants", 1350), help="15 corruptions per source doc")
    c4.metric("Total Image Records", summary_stats.get("total_image_records", 1440), help="90 clean + 1350 degraded")
    c5.metric("Train / Val / Test", "54 / 18 / 18", help="Document split (60% / 20% / 20%)")
    
    st.markdown("---")
    
    # Visualizations Row 1: Splits & Corruptions
    col_vis1, col_vis2 = st.columns(2)
    degrad_df = ResearchDataLoader.load_degradation_manifest()
    
    with col_vis1:
        st.markdown("##### Document Split Distribution")
        split_data = pd.DataFrame([
            {"Split": "Train (54 source docs)", "Source Docs": 54, "Image Records": 54 * 16, "Percentage": "60%"},
            {"Split": "Validation (18 source docs)", "Source Docs": 18, "Image Records": 18 * 16, "Percentage": "20%"},
            {"Split": "Test (18 source docs)", "Source Docs": 18, "Image Records": 18 * 16, "Percentage": "20%"},
        ])
        fig_split = px.pie(
            split_data,
            names="Split",
            values="Image Records",
            hole=0.45,
            color_discrete_sequence=["#2563EB", "#10B981", "#F59E0B"],
            title="Image Records by Document Split (Total: 1,440)"
        )
        fig_split.update_traces(textinfo="percent+label")
        st.plotly_chart(fig_split, use_container_width=True)
        
    with col_vis2:
        st.markdown("##### Degradation Conditions per Document")
        corr_data = pd.DataFrame([
            {"Corruption": "Clean Baseline", "Images": 90, "Severity": "None"},
            {"Corruption": "Gaussian Blur", "Images": 270, "Severity": "Mild, Mod, Sev"},
            {"Corruption": "Gaussian Noise", "Images": 270, "Severity": "Mild, Mod, Sev"},
            {"Corruption": "Low Contrast", "Images": 270, "Severity": "Mild, Mod, Sev"},
            {"Corruption": "Rotation", "Images": 270, "Severity": "Mild, Mod, Sev"},
            {"Corruption": "JPEG Compression", "Images": 270, "Severity": "Mild, Mod, Sev"},
        ])
        fig_corr = px.bar(
            corr_data,
            x="Corruption",
            y="Images",
            color="Corruption",
            text="Images",
            title="Image Records per Degradation Type (16 variants per doc)",
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig_corr.update_layout(showlegend=False)
        st.plotly_chart(fig_corr, use_container_width=True)
        
    # Visualizations Row 2: Languages and Categories
    col_vis3, col_vis4 = st.columns(2)
    with col_vis3:
        st.markdown("##### Language Breakdown (Equal 30 Docs / 480 Images each)")
        lang_data = pd.DataFrame([
            {"Language": "English", "Source Docs": 30, "Images": 480},
            {"Language": "Hindi (हिन्दी)", "Source Docs": 30, "Images": 480},
            {"Language": "Kannada (ಕನ್ನಡ)", "Source Docs": 30, "Images": 480},
        ])
        fig_lang = px.bar(
            lang_data,
            x="Language",
            y="Images",
            color="Language",
            text="Images",
            title="Records by Script / Language",
            color_discrete_sequence=["#3B82F6", "#EC4899", "#8B5CF6"]
        )
        fig_lang.update_layout(showlegend=False)
        st.plotly_chart(fig_lang, use_container_width=True)
        
    with col_vis4:
        st.markdown("##### Document Categories (Equal 30 Docs / 480 Images each)")
        cat_data = pd.DataFrame([
            {"Category": "Invoice / Bill", "Fields": "name, date, id, address, amount", "Images": 480},
            {"Category": "Application Form", "Fields": "name, date, id, address, phone", "Images": 480},
            {"Category": "Certificate / Marksheet", "Fields": "name, date, id, institution, grade", "Images": 480},
        ])
        fig_cat = px.bar(
            cat_data,
            x="Category",
            y="Images",
            color="Category",
            text="Images",
            title="Records by Document Category",
            color_discrete_sequence=["#06B6D4", "#10B981", "#F97316"]
        )
        fig_cat.update_layout(showlegend=False)
        st.plotly_chart(fig_cat, use_container_width=True)
        
    # Degradation Manifest Explorer
    st.markdown("---")
    st.markdown("##### Degradation Manifest Explorer (`degradation_manifest.csv`)")
    if not degrad_df.empty:
        c_filt1, c_filt2, c_filt3 = st.columns(3)
        with c_filt1:
            sel_split = st.multiselect("Filter Split", options=sorted(degrad_df["split"].unique()), default=sorted(degrad_df["split"].unique()))
        with c_filt2:
            sel_lang = st.multiselect("Filter Language", options=sorted(degrad_df["language"].unique()), default=sorted(degrad_df["language"].unique()))
        with c_filt3:
            sel_corr = st.multiselect("Filter Corruption", options=sorted(degrad_df["corruption"].unique()), default=sorted(degrad_df["corruption"].unique()))
            
        filtered_df = degrad_df[
            degrad_df["split"].isin(sel_split) &
            degrad_df["language"].isin(sel_lang) &
            degrad_df["corruption"].isin(sel_corr)
        ]
        st.dataframe(filtered_df.head(100), use_container_width=True, hide_index=True)
        st.caption(f"Displaying {len(filtered_df)} of {len(degrad_df)} records. Showing first 100 rows.")
        export_csv_button(filtered_df, "filtered_degradation_manifest.csv", "Download Filtered Manifest (CSV)")


# ==============================================================================
# TAB 2: OCR BENCHMARKING
# ==============================================================================
with tab_ocr:
    st.subheader("2. OCR Engine Benchmark (EasyOCR vs PaddleOCR vs Tesseract)")
    st.markdown(
        """
        Three Optical Character Recognition (OCR) engines were benchmarked across the 1,440 synthetic document images.
        Evaluation metrics include **Character Error Rate (CER)**, **Word Error Rate (WER)**, and **Character Accuracy**.
        """
    )
    
    ocr_overall = ResearchDataLoader.load_ocr_overall_metrics()
    ocr_lang = ResearchDataLoader.load_ocr_language_metrics()
    ocr_corr = ResearchDataLoader.load_ocr_corruption_metrics()
    ocr_sev = ResearchDataLoader.load_ocr_severity_metrics()
    
    # Engine Leaderboard Cards
    c_ocr_kpi1, c_ocr_kpi2, c_ocr_kpi3 = st.columns(3)
    c_ocr_kpi1.success("🏆 **Rank 1: EasyOCR**\n\n- **CER:** 15.57%\n- **WER:** 45.03%\n- **Char Accuracy:** 84.43%\n- **Best in:** Kannada & English")
    c_ocr_kpi2.info("🥈 **Rank 2: PaddleOCR**\n\n- **CER:** 23.75%\n- **WER:** 56.34%\n- **Char Accuracy:** 76.25%\n- **Best in:** Hindi (15.99% CER)")
    c_ocr_kpi3.warning("🥉 **Rank 3: Tesseract**\n\n- **CER:** 29.22%\n- **WER:** 55.43%\n- **Char Accuracy:** 70.78%\n- **Struggled:** Indic character ligatures")
    
    st.markdown("---")
    
    # Overall Table
    st.markdown("##### Overall Performance Comparison across 1,440 Images")
    if not ocr_overall.empty:
        display_ocr_overall = ocr_overall.copy()
        if "CER" in display_ocr_overall.columns:
            display_ocr_overall["CER (%)"] = (display_ocr_overall["CER"] * 100).round(2)
        if "WER" in display_ocr_overall.columns:
            display_ocr_overall["WER (%)"] = (display_ocr_overall["WER"] * 100).round(2)
        if "character_accuracy" in display_ocr_overall.columns:
            display_ocr_overall["Char Accuracy (%)"] = (display_ocr_overall["character_accuracy"] * 100).round(2)
        st.dataframe(display_ocr_overall[["engine", "CER (%)", "WER (%)", "Char Accuracy (%)"]], use_container_width=True, hide_index=True)
    
    # Charts Row 1: Language-wise and Corruption-wise
    c_ocr_c1, c_ocr_c2 = st.columns(2)
    with c_ocr_c1:
        st.markdown("##### Character Error Rate (CER) by Language")
        if not ocr_lang.empty:
            df_plot_lang = ocr_lang.copy()
            df_plot_lang["CER (%)"] = df_plot_lang["CER"] * 100
            fig_ocr_lang = px.bar(
                df_plot_lang,
                x="language",
                y="CER (%)",
                color="engine",
                barmode="group",
                title="CER by Language (Lower is Better)",
                color_discrete_map={"EasyOCR": "#10B981", "PaddleOCR": "#3B82F6", "Tesseract": "#F59E0B"},
                text_auto=".1f"
            )
            fig_ocr_lang.update_layout(yaxis_title="CER (%)")
            st.plotly_chart(fig_ocr_lang, use_container_width=True)
            
    with c_ocr_c2:
        st.markdown("##### Word Error Rate (WER) by Language")
        if not ocr_lang.empty:
            df_plot_lang_wer = ocr_lang.copy()
            df_plot_lang_wer["WER (%)"] = df_plot_lang_wer["WER"] * 100
            fig_ocr_lang_wer = px.bar(
                df_plot_lang_wer,
                x="language",
                y="WER (%)",
                color="engine",
                barmode="group",
                title="WER by Language (Lower is Better)",
                color_discrete_map={"EasyOCR": "#10B981", "PaddleOCR": "#3B82F6", "Tesseract": "#F59E0B"},
                text_auto=".1f"
            )
            fig_ocr_lang_wer.update_layout(yaxis_title="WER (%)")
            st.plotly_chart(fig_ocr_lang_wer, use_container_width=True)
            
    # Charts Row 2: Corruption and Severity
    c_ocr_c3, c_ocr_c4 = st.columns(2)
    with c_ocr_c3:
        st.markdown("##### CER Across Corruption Types")
        if not ocr_corr.empty:
            df_plot_corr = ocr_corr.copy()
            df_plot_corr["CER (%)"] = df_plot_corr["CER"] * 100
            fig_ocr_corr = px.bar(
                df_plot_corr,
                x="corruption",
                y="CER (%)",
                color="engine",
                barmode="group",
                title="CER Across Degradation Types",
                color_discrete_map={"EasyOCR": "#10B981", "PaddleOCR": "#3B82F6", "Tesseract": "#F59E0B"},
                text_auto=".1f"
            )
            fig_ocr_corr.update_layout(yaxis_title="CER (%)")
            st.plotly_chart(fig_ocr_corr, use_container_width=True)
            
    with c_ocr_c4:
        st.markdown("##### CER Across Degradation Severity Levels")
        if not ocr_sev.empty:
            df_plot_sev = ocr_sev.copy()
            df_plot_sev["CER (%)"] = df_plot_sev["CER"] * 100
            fig_ocr_sev = px.bar(
                df_plot_sev,
                x="severity",
                y="CER (%)",
                color="engine",
                barmode="group",
                title="CER Degradation from Mild to Severe",
                color_discrete_map={"EasyOCR": "#10B981", "PaddleOCR": "#3B82F6", "Tesseract": "#F59E0B"},
                text_auto=".1f"
            )
            fig_ocr_sev.update_layout(yaxis_title="CER (%)")
            st.plotly_chart(fig_ocr_sev, use_container_width=True)


# ==============================================================================
# TAB 3: LLM EXTRACTION BENCHMARK
# ==============================================================================
with tab_llm:
    st.subheader("3. LLM Information Extraction Benchmark (Qwen3-4B)")
    st.markdown(
        """
        Structured information extraction was evaluated across 4 model setups on the **54 clean test document evaluations**
        (18 unique test documents processed through all 3 OCR engines = 54 evaluation records, 270 total field instances).
        """
    )
    
    render_provisional_badge()
    
    # Master Table
    master_df = ResearchDataLoader.evaluate_four_way_comparison()
    st.markdown("##### Master Comparison Table (54 Test Records / 270 Fields)")
    st.dataframe(master_df, use_container_width=True, hide_index=True)
    export_csv_button(master_df, "qwen3_extraction_master_comparison.csv", "Download Master Table (CSV)")
    
    st.markdown("---")
    
    # Comparison Chart
    c_llm_chart1, c_llm_chart2 = st.columns(2)
    with c_llm_chart1:
        st.markdown("##### Field-Level vs Document-Level Exact Match (%)")
        fig_eval_bar = px.bar(
            master_df,
            x="Model / Method",
            y=["Field-Level Exact Match (%)", "Document-Level Exact Match (%)"],
            barmode="group",
            title="Extraction Accuracy Comparison (Higher is Better)",
            color_discrete_sequence=["#2563EB", "#10B981"],
            text_auto=".1f"
        )
        fig_eval_bar.update_layout(yaxis_title="Accuracy (%)", legend_title="Metric")
        st.plotly_chart(fig_eval_bar, use_container_width=True)
        
    with c_llm_chart2:
        st.markdown("##### Key Takeaway & Ranking")
        st.info(
            """
            🏆 **Few-Shot Qwen3-4B is the clear benchmark winner:**
            - **Field-Level Exact Match:** **83.33%** (225/270 correct fields)
            - **Document-Level Exact Match:** **61.11%** (33/54 exact documents)
            - Outperforms Zero-Shot by **+8.15%** field accuracy and **+25.93%** document match.
            - Outperforms Experiment 1 LoRA by **+13.70%** field accuracy and **+29.63%** document match.
            
            ⚠️ **Why did LoRA underperform?**  
            Fine-tuning on noisy synthetic training records caused the model to develop an **over-abstention bias** (returning empty strings when valid text was present) and disrupted the tokenization of numeric dates, amounts, and document IDs.
            """
        )

    # Sub-tabs for Granular Breakdowns
    st.markdown("---")
    st.markdown("##### Detailed Performance Breakdowns")
    subtab_lang, subtab_cat, subtab_field, subtab_engine = st.tabs([
        "Language Breakdown",
        "Category Breakdown",
        "Field-Level Breakdown",
        "OCR Engine Dependency"
    ])
    
    with subtab_lang:
        df_lang = ResearchDataLoader.get_language_breakdown()
        st.dataframe(df_lang, use_container_width=True, hide_index=True)
        fig_lang_break = px.bar(
            df_lang,
            x="Language",
            y="Field Accuracy (%)",
            color="Model",
            barmode="group",
            title="Field Accuracy by Language across Models",
            color_discrete_sequence=px.colors.qualitative.Safe,
            text_auto=".1f"
        )
        st.plotly_chart(fig_lang_break, use_container_width=True)
        
    with subtab_cat:
        df_cat = ResearchDataLoader.get_category_breakdown()
        st.dataframe(df_cat, use_container_width=True, hide_index=True)
        fig_cat_break = px.bar(
            df_cat,
            x="Category",
            y="Field Accuracy (%)",
            color="Model",
            barmode="group",
            title="Field Accuracy by Document Category",
            color_discrete_sequence=px.colors.qualitative.Bold,
            text_auto=".1f"
        )
        st.plotly_chart(fig_cat_break, use_container_width=True)
        
    with subtab_field:
        df_field = ResearchDataLoader.get_field_breakdown()
        st.dataframe(df_field, use_container_width=True, hide_index=True)
        fig_field_break = px.bar(
            df_field,
            x="Field",
            y="Accuracy (%)",
            color="Model",
            barmode="group",
            title="Accuracy by Field Type (Notice drop in document_id)",
            color_discrete_sequence=px.colors.qualitative.Vivid,
            text_auto=".1f"
        )
        st.plotly_chart(fig_field_break, use_container_width=True)
        
    with subtab_engine:
        df_engine = ResearchDataLoader.get_engine_breakdown()
        if "Ocr_Engine" in df_engine.columns:
            df_engine = df_engine.rename(columns={"Ocr_Engine": "OCR Engine"})
        st.dataframe(df_engine, use_container_width=True, hide_index=True)
        fig_eng_break = px.bar(
            df_engine,
            x="OCR Engine",
            y="Field Accuracy (%)",
            color="Model",
            barmode="group",
            title="Field Accuracy Grouped by OCR Input Engine",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            text_auto=".1f"
        )
        st.plotly_chart(fig_eng_break, use_container_width=True)


# ==============================================================================
# TAB 4: STATISTICAL ANALYSIS & HYPOTHESIS TESTING
# ==============================================================================
with tab_stats:
    st.subheader("4. Statistical Comparisons & Research Hypothesis Testing")
    
    st.markdown(
        """
        ### Central Research Question:
        > *"Does LoRA fine-tuning of Qwen3-4B improve structured field extraction from noisy OCR text in English, Hindi and Kannada compared with zero-shot and few-shot prompting?"*
        
        ### Empirical Answer:
        **No.** Across all evaluated dimensions, **Few-Shot Prompting (In-Context Learning)** significantly outperforms LoRA fine-tuning.
        """
    )
    
    # Paired Differences Matrix
    st.markdown("##### LLM Paired Differences: Record-Level vs Source-Document Level")
    
    tab_pair_rec, tab_pair_doc = st.tabs([
        "Record-Level (54 OCR Test Records)",
        "Source-Document-Level (18 Unique Source Documents)"
    ])
    
    with tab_pair_rec:
        paired_rec_df = ResearchDataLoader.get_paired_comparisons_summary()
        st.dataframe(paired_rec_df, use_container_width=True, hide_index=True)
        st.markdown(
            """
            *Unit of Analysis: 54 OCR-specific records (5 fields per record).*
            - **Few-Shot vs Zero-Shot:** Few-Shot scored higher on 23 records, identical on 30, and lower on only 1 record (+22 net fields gained, +0.407 mean fields/record).
            - **Exp 1 LoRA vs Zero-Shot:** LoRA scored higher on 0 records, identical on 41, and lower on 13 records (-15 net fields lost, -0.278 mean fields/record).
            - **Few-Shot vs Exp 1 LoRA:** Few-Shot scored higher on 30 records, identical on 24, and lower on 0 records (+37 net fields gained, +0.685 mean fields/record).
            """
        )
        export_csv_button(paired_rec_df, "paired_record_differences.csv", "Download Record-Level Comparisons (CSV)")
        
    with tab_pair_doc:
        paired_doc_df = ResearchDataLoader.get_source_document_paired_comparisons_summary()
        st.dataframe(paired_doc_df, use_container_width=True, hide_index=True)
        st.markdown(
            """
            *Unit of Analysis: 18 unique source documents (pooling the 3 OCR engine records = 15 fields per document).*
            - **Few-Shot vs Zero-Shot:** Few-Shot won on 11 source documents, tied on 7, and lost on 0 documents (+22 net fields gained, +1.222 mean fields/doc).
            - **Exp 1 LoRA vs Zero-Shot:** LoRA won on 0 source documents, tied on 9, and lost on 9 documents (-15 net fields lost, -0.833 mean fields/doc).
            - **Few-Shot vs Exp 1 LoRA:** Few-Shot won on 12 source documents, tied on 6, and lost on 0 documents (+37 net fields gained, +2.056 mean fields/doc).
            """
        )
        export_csv_button(paired_doc_df, "paired_source_document_differences.csv", "Download Source-Doc Comparisons (CSV)")
    
    st.markdown("---")
    
    # Document-Level Bootstrap Confidence Intervals
    st.markdown("##### Document-Level Bootstrap 95% Confidence Intervals (1,440 OCR Evaluations)")
    st.markdown(
        "Bootstrap confidence intervals (1,000 resamples) computed at the document level. "
        "A negative difference indicates the first engine achieved lower error."
    )
    
    boot_df = ResearchDataLoader.load_bootstrap_comparisons()
    if not boot_df.empty:
        # Forest plot of bootstrap differences
        fig_forest = go.Figure()
        
        cer_boot = boot_df[boot_df["metric"] == "CER"].copy()
        for idx, row in cer_boot.iterrows():
            label = f"{row['language']} ({row['comparison']})"
            fig_forest.add_trace(go.Scatter(
                x=[row["difference"]],
                y=[label],
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[row["CI_upper"] - row["difference"]],
                    arrayminus=[row["difference"] - row["CI_lower"]],
                    color="#2563EB",
                    thickness=2,
                    width=6
                ),
                mode="markers",
                name=label,
                marker=dict(size=10, color="#1D4ED8")
            ))
            
        fig_forest.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="No Difference (Zero)")
        fig_forest.update_layout(
            title="CER Paired Bootstrap Differences & 95% Confidence Intervals",
            xaxis_title="Mean Paired CER Difference (Negative = Engine 1 Better)",
            yaxis_title="Comparison Pair",
            showlegend=False,
            height=450
        )
        st.plotly_chart(fig_forest, use_container_width=True)
        
        st.dataframe(boot_df, use_container_width=True, hide_index=True)
        export_csv_button(boot_df, "paired_bootstrap_comparisons.csv", "Download Bootstrap Comparisons (CSV)")
        
    st.markdown("---")
    
    # Document-Level Engine CIs
    st.markdown("##### Engine Mean Error Rates with 95% Confidence Intervals")
    cis_df = ResearchDataLoader.load_document_level_cis()
    if not cis_df.empty:
        st.dataframe(cis_df, use_container_width=True, hide_index=True)


# ==============================================================================
# TAB 5: ERROR & RECOVERABILITY ANALYSIS
# ==============================================================================
with tab_error:
    st.subheader("5. Error & OCR Recoverability Analysis")
    st.markdown(
        """
        A key research inquiry is whether extraction errors stem from **OCR error propagation** 
        (the ground truth was missing from OCR text) or **LLM extraction failure** 
        (the OCR text contained valid evidence, but the model failed to extract it).
        
        > ℹ️ **Methodological Denominator & Split Scope Notice:**  
        > The recoverability and false abstention diagnostics below are loaded directly from the **Validation Diagnostic Benchmark** 
        > (`qwen3_lora_ocr_support_diagnostic.csv`, evaluated on the 18 validation documents, 54 records, 270 field instances).  
        > In this diagnostic validation split, LoRA encountered **71 total extraction errors** (199 correct, 73.70% accuracy).  
        > By contrast, the **Test Set Master Benchmark** evaluates the 18 held-out test documents where LoRA encountered **82 extraction errors** (188 correct, 69.63% accuracy).  
        > Both benchmarks use distinct, non-overlapping document splits.
        """
    )
    
    rec_summary = ResearchDataLoader.get_ocr_recoverability_summary()
    
    if rec_summary:
        c_rec1, c_rec2, c_rec3, c_rec4 = st.columns(4)
        c_rec1.metric(
            "Total Field Instances",
            rec_summary.get("total_fields", 270),
            help="54 test documents x 5 fields"
        )
        c_rec2.metric(
            "OCR-Supported Fields",
            f"{rec_summary.get('supported_fields', 208)} ({rec_summary.get('support_rate', 77.04)}%)",
            help="OCR text contained exact/fuzzy ground truth"
        )
        c_rec3.metric(
            "OCR Missing Errors",
            f"{rec_summary.get('ocr_missing_errors', 62)} ({rec_summary.get('ocr_error_share_pct', 87.32)}%)",
            delta="-87.3% of total errors",
            delta_color="inverse",
            help="Failures caused by OCR dropping the ground truth string"
        )
        c_rec4.metric(
            "False Abstentions",
            f"{rec_summary.get('false_abstentions', 9)} ({rec_summary.get('llm_error_share_pct', 12.68)}%)",
            delta="-12.7% of total errors",
            delta_color="inverse",
            help="Cases where valid OCR evidence existed but LoRA output empty string"
        )
        
    st.markdown("---")
    
    # Visualizations: Error Decomposition & Field Vulnerability
    c_err1, c_err2 = st.columns(2)
    with c_err1:
        st.markdown("##### Root Cause Decomposition of Extraction Failures")
        err_decomp = pd.DataFrame([
            {"Root Cause": "OCR Error Propagation (Ground truth missing in OCR text)", "Count": 62, "Percentage": "87.3%"},
            {"Root Cause": "LLM False Abstention (OCR text valid, LoRA output empty)", "Count": 9, "Percentage": "12.7%"}
        ])
        fig_err_pie = px.pie(
            err_decomp,
            names="Root Cause",
            values="Count",
            hole=0.4,
            color_discrete_sequence=["#EF4444", "#F59E0B"],
            title="Decomposition of 71 LoRA Extraction Failures"
        )
        st.plotly_chart(fig_err_pie, use_container_width=True)
        
    with c_err2:
        st.markdown("##### Evidence Audit in OCR Text (`experiment2_evidence_audit.csv`)")
        audit_df = ResearchDataLoader.load_evidence_audit()
        if not audit_df.empty:
            audit_counts = audit_df["evidence_category"].value_counts().reset_index()
            audit_counts.columns = ["Evidence Category", "Fields"]
            fig_audit = px.bar(
                audit_counts,
                x="Evidence Category",
                y="Fields",
                color="Evidence Category",
                text="Fields",
                title="Evidence Availability Across 810 Fields",
                color_discrete_sequence=["#10B981", "#EF4444", "#F59E0B"]
            )
            fig_audit.update_layout(showlegend=False)
            st.plotly_chart(fig_audit, use_container_width=True)
            
    st.markdown("---")
    
    # False Abstention Detailed Table
    st.markdown("##### False Abstention Comparison (All 9 Cases)")
    st.markdown(
        """
        These 9 cases demonstrate the exact failure mode of Experiment 1 LoRA:
        The OCR text contained the correct information, and **both Zero-Shot and Few-Shot extracted it successfully**,
        yet **LoRA output an empty string / NaN**.
        """
    )
    fa_df = ResearchDataLoader.load_false_abstention_comparison()
    if not fa_df.empty:
        st.dataframe(fa_df, use_container_width=True, hide_index=True)
        export_csv_button(fa_df, "false_abstention_comparison.csv", "Download False Abstention Cases (CSV)")
        
    st.markdown("---")
    
    # Field Error Analysis Table
    st.markdown("##### Field-Level Error Distribution (`qwen3_lora_field_error_analysis.csv`)")
    fea_df = ResearchDataLoader.load_field_error_analysis()
    if not fea_df.empty:
        c_fea1, c_fea2 = st.columns(2)
        with c_fea1:
            err_by_field = fea_df.groupby("field")["correct"].agg(
                Total="count",
                Correct=lambda x: (x == True).sum(),
                Errors=lambda x: (x == False).sum()
            ).reset_index()
            err_by_field["Accuracy (%)"] = (err_by_field["Correct"] / err_by_field["Total"] * 100).round(1)
            err_by_field = err_by_field.sort_values(by="Errors", ascending=False)
            fig_field_err = px.bar(
                err_by_field,
                x="field",
                y="Errors",
                text="Errors",
                title="Errors by Field (document_id is most vulnerable)",
                color="Errors",
                color_continuous_scale="Reds"
            )
            st.plotly_chart(fig_field_err, use_container_width=True)
            
        with c_fea2:
            err_by_lang = fea_df.groupby("language")["correct"].agg(
                Total="count",
                Correct=lambda x: (x == True).sum(),
                Errors=lambda x: (x == False).sum()
            ).reset_index()
            err_by_lang["Accuracy (%)"] = (err_by_lang["Correct"] / err_by_lang["Total"] * 100).round(1)
            fig_lang_err = px.bar(
                err_by_lang,
                x="language",
                y="Errors",
                text="Errors",
                title="Errors by Language (English 3 vs Hindi 31 vs Kannada 37)",
                color="Errors",
                color_continuous_scale="OrRd"
            )
            st.plotly_chart(fig_lang_err, use_container_width=True)


# ==============================================================================
# TAB 6: LORA TRAINING TRAJECTORIES
# ==============================================================================
with tab_training:
    st.subheader("6. LoRA Fine-Tuning Loss Curves & Training Dynamics")
    
    exp1_hist, exp2_hist = ResearchDataLoader.load_training_histories()
    
    c_train1, c_train2 = st.columns(2)
    
    with c_train1:
        st.markdown("##### Experiment 1: Standard QLoRA (5 Epochs)")
        st.markdown(
            """
            - **Supervision Target:** Full ground truth JSON target.
            - **Penalty:** Penalized on all ground truth fields even when OCR dropped the field.
            - **Convergence:** Training loss dropped from 0.0630 to 0.0015.
            - **Overfitting Indicator:** Validation loss reached its minimum at **Epoch 3 (0.01028)**, then rebounded to 0.01602 at Epoch 5.
            """
        )
        if exp1_hist:
            df_hist1 = pd.DataFrame(exp1_hist)
            fig_loss1 = go.Figure()
            fig_loss1.add_trace(go.Scatter(x=df_hist1["epoch"], y=df_hist1["train_loss"], mode="lines+markers", name="Train Loss", line=dict(color="#2563EB", width=2)))
            fig_loss1.add_trace(go.Scatter(x=df_hist1["epoch"], y=df_hist1["validation_loss"], mode="lines+markers", name="Validation Loss", line=dict(color="#EF4444", width=2, dash="dash")))
            fig_loss1.update_layout(title="Experiment 1 Loss Curves", xaxis_title="Epoch", yaxis_title="Loss")
            st.plotly_chart(fig_loss1, use_container_width=True)
            st.dataframe(df_hist1, use_container_width=True, hide_index=True)
            
    with c_train2:
        st.markdown("##### Experiment 2: Evidence-Aware LoRA (5 Epochs)")
        st.markdown(
            """
            - **Supervision Target:** Evidence-aware targets (empty string when unsupported in OCR).
            - **Goal:** Teach model principled abstention without guessing.
            - **Convergence:** Training loss dropped from 0.1020 to 0.00019; validation loss dropped to 0.000275.
            - **Status Note:** *Provisional pending verification of Colab test inference.*
            """
        )
        if exp2_hist:
            df_hist2 = pd.DataFrame(exp2_hist)
            fig_loss2 = go.Figure()
            fig_loss2.add_trace(go.Scatter(x=df_hist2["epoch"], y=df_hist2["training_loss"], mode="lines+markers", name="Train Loss", line=dict(color="#10B981", width=2)))
            fig_loss2.add_trace(go.Scatter(x=df_hist2["epoch"], y=df_hist2["validation_loss"], mode="lines+markers", name="Validation Loss", line=dict(color="#F59E0B", width=2, dash="dash")))
            fig_loss2.update_layout(title="Experiment 2 Loss Curves", xaxis_title="Epoch", yaxis_title="Loss")
            st.plotly_chart(fig_loss2, use_container_width=True)
            st.dataframe(df_hist2, use_container_width=True, hide_index=True)
            
    st.markdown("---")
    
    # Learning Rate Schedule
    st.markdown("##### Learning Rate Decay Schedule")
    if exp1_hist:
        df_lr = pd.DataFrame(exp1_hist)
        fig_lr = px.line(df_lr, x="epoch", y="learning_rate", markers=True, title="Linear Learning Rate Decay (Peak: 1.68e-4 to 0.0)")
        fig_lr.update_layout(yaxis_title="Learning Rate")
        st.plotly_chart(fig_lr, use_container_width=True)


# ==============================================================================
# TAB 7: METHODOLOGY & LIMITATIONS
# ==============================================================================
with tab_methodology:
    st.subheader("7. Research Methodology, Limitations & Reproducibility")
    
    st.markdown(
        """
        This research was conducted under academic principles of transparency, scientific reproducibility,
        and honest reporting of null or negative results.
        """
    )
    
    c_m1, c_m2 = st.columns(2)
    
    with c_m1:
        st.markdown("##### Experimental Methodology")
        st.markdown(
            """
            1. **Base Foundation Model:** Qwen3-4B (4 billion parameters), chosen for strong multilingual representation in Indic scripts.
            2. **Quantization & PEFT:** 4-bit NormalFloat (NF4) quantization via bitsandbytes; LoRA rank $r=16$, $\\alpha=32$, dropout $0.05$ on all linear projection modules.
            3. **Prompt Preservation:** Exact zero-shot system prompt and 9 clean training demonstrations (1 per language x category pair) preserved across all comparisons.
            4. **Evaluation Normalization:** Exact research string normalization protocol (`normalize_field_value`): Unicode NFC normalization, zero-width joiner removal, casefolding, and separator normalization for dates, phone numbers, amounts, and IDs.
            """
        )
        
    with c_m2:
        st.markdown("##### Scientific Limitations")
        st.markdown(
            """
            1. **Synthetic Document Bounds:** Documents were programmatically generated from 6 template families. Real-world administrative documents feature handwriting, non-standard layouts, physical tears, stamps, and low-grade camera capture.
            2. **Test Sample Size:** 18 unique source test documents (evaluated across 3 OCR engines = 54 records). While adequate for pilot statistical bootstrapping, larger multi-center test sets are required for production claims.
            3. **Non-Generalizability:** EasyOCR's benchmark lead on synthetic noise may not transfer directly to complex real-world document scans with varied font typography.
            4. **Zero-GPU CPU Inference:** The web interface operates in Zero-GPU mode with CPU-safe inference and benchmark lookup caching to ensure zero cloud cost and instant startup.
            """
        )
        
    st.markdown("---")
    render_provisional_badge()
    st.caption(f"Status Details: {EXP2_STATUS_DETAILS}")
