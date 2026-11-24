"""
Page 1: Document Information Extraction Pipeline
End-to-end interface for document upload, multi-engine OCR, human-in-the-loop text editing,
and structured JSON information extraction using Qwen3-4B methods.
"""

import streamlit as st
from PIL import Image
from pathlib import Path
import json

from src.config import (
    CSS_FILE,
    CATEGORY_SCHEMAS,
    CATEGORY_NAMES,
    SUPPORTED_LANGUAGES,
    LANGUAGE_LABELS,
    OCR_ENGINES,
    DEFAULT_OCR_ENGINE,
    ALLOWED_EXTENSIONS,
    RAW_DOCUMENTS_DIR,
    EXP2_STATUS,
    EXP2_STATUS_DETAILS,
)
from src.utils import load_css, render_header, render_provisional_badge, export_json_button
from src.document_processor import DocumentProcessor, DocumentValidationError
from src.image_preprocessing import ImagePreprocessor, PREPROCESSING_METHODS
from src.ocr_engine import get_ocr_engine, BenchmarkLookupEngine
from src.model_loader import get_model_backend, HardwareInfo
from src.prompt_templates import build_user_prompt

st.set_page_config(page_title="Document Extraction - BharatDoc AI", page_icon="📝", layout="wide")
load_css(CSS_FILE)

render_header(
    title="Document Information Extraction",
    subtitle="Upload an Indian document, run OCR, inspect text, and extract structured JSON fields.",
    badge="Live Pipeline"
)

# Initialize Session State
if "raw_ocr_text" not in st.session_state:
    st.session_state["raw_ocr_text"] = ""
if "edited_ocr_text" not in st.session_state:
    st.session_state["edited_ocr_text"] = ""
if "current_image" not in st.session_state:
    st.session_state["current_image"] = None
if "selected_doc_id" not in st.session_state:
    st.session_state["selected_doc_id"] = None
if "extraction_result" not in st.session_state:
    st.session_state["extraction_result"] = None

# Hardware Status Banner
hw_desc = HardwareInfo.get_hardware_description()
st.caption(f"💻 **Execution Environment:** `{hw_desc}` | Lazy Loading: **Active**")

# Section 1: Source Document & Extraction Settings
col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.subheader("1. Source Document")
    
    source_mode = st.radio(
        "Choose Document Source",
        options=["Upload Custom Document", "Select Sample Synthetic Benchmark Document"],
        index=0,
        horizontal=True
    )
    
    current_image = None
    doc_id = None
    
    if source_mode == "Upload Custom Document":
        uploaded_file = st.file_uploader(
            "Upload Image or PDF",
            type=[ext.lstrip(".") for ext in ALLOWED_EXTENSIONS],
            help="Supported formats: PNG, JPG, JPEG, PDF (up to 20 MB)"
        )
        if uploaded_file is not None:
            try:
                images, pdf_text = DocumentProcessor.load_document(
                    uploaded_file.name, uploaded_file.getvalue()
                )
                if images:
                    if len(images) > 1:
                        page_num = st.slider("Select PDF Page", 1, len(images), 1)
                        current_image = images[page_num - 1]
                    else:
                        current_image = images[0]
                    st.session_state["current_image"] = current_image
                    st.session_state["selected_doc_id"] = None
            except DocumentValidationError as e:
                st.error(f"Validation Error: {e}")
                
    else:
        sample_options = [
            ("english_invoice_009", "English Invoice (INV-0009 - Test Benchmark)"),
            ("english_application_019", "English Application (APP-0019 - Test Benchmark)"),
            ("english_certificate_029", "English Certificate (CERT-0029 - Test Benchmark)"),
            ("hindi_invoice_039", "Hindi Invoice (चालान INV-0039 - Test Benchmark)"),
            ("hindi_application_049", "Hindi Application (आवेदन APP-0049 - Test Benchmark)"),
            ("hindi_certificate_059", "Hindi Certificate (प्रमाण पत्र CERT-0059 - Test Benchmark)"),
            ("kannada_invoice_069", "Kannada Invoice (ಸರಕುಪಟ್ಟಿ INV-0069 - Test Benchmark)"),
            ("kannada_application_079", "Kannada Application (ಅರ್ಜಿ APP-0079 - Test Benchmark)"),
            ("kannada_certificate_089", "Kannada Certificate (ಪ್ರಮಾಣಪತ್ರ CERT-0089 - Test Benchmark)"),
            ("english_invoice_001", "English Invoice (INV-0001 - Training Demo)"),
            ("hindi_invoice_031", "Hindi Invoice (चालान INV-0031 - Training Demo)"),
            ("kannada_invoice_061", "Kannada Invoice (ಸರಕುಪಟ್ಟಿ INV-0061 - Training Demo)"),
        ]
        
        sample_choice = st.selectbox(
            "Choose a Benchmark Sample",
            options=[k for k, v in sample_options],
            format_func=lambda x: dict(sample_options).get(x, x),
            index=0
        )
        
        doc_id = sample_choice
        sample_file = RAW_DOCUMENTS_DIR / f"{doc_id}.png"
        if sample_file.exists():
            current_image = Image.open(sample_file).convert("RGB")
            st.session_state["current_image"] = current_image
            st.session_state["selected_doc_id"] = doc_id
            
    st.markdown("---")
    st.subheader("2. Extraction & Engine Settings")
    
    # Auto-infer defaults for benchmark samples
    default_cat_idx = 0
    default_lang_idx = 0
    if doc_id:
        if "invoice" in doc_id:
            default_cat_idx = 0
        elif "application" in doc_id:
            default_cat_idx = 1
        elif "certificate" in doc_id:
            default_cat_idx = 2
            
        if "english" in doc_id:
            default_lang_idx = 0
        elif "hindi" in doc_id:
            default_lang_idx = 1
        elif "kannada" in doc_id:
            default_lang_idx = 2

    category = st.selectbox(
        "Document Category",
        options=list(CATEGORY_SCHEMAS.keys()),
        index=default_cat_idx,
        format_func=lambda x: CATEGORY_NAMES.get(x, x),
        help="Determines the target structured extraction schema."
    )

    language = st.selectbox(
        "Document Language",
        options=SUPPORTED_LANGUAGES,
        index=default_lang_idx,
        format_func=lambda x: LANGUAGE_LABELS.get(x, x),
        help="Target language for OCR and extraction."
    )

    ocr_engine_name = st.selectbox(
        "OCR Engine",
        options=list(OCR_ENGINES.keys()),
        index=0,
        format_func=lambda x: OCR_ENGINES.get(x, x),
        help="EasyOCR achieved best overall performance on the synthetic benchmark."
    )

    preprocessing_choice = st.selectbox(
        "Image Preprocessing (Optional)",
        options=list(PREPROCESSING_METHODS.keys()),
        format_func=lambda x: PREPROCESSING_METHODS.get(x, x),
        index=0,
        help="Raw image passed directly to OCR by default. Use preprocessing only if image has contrast or noise defects."
    )

with col_right:
    st.subheader("3. Document Preview")
    
    if st.session_state.get("current_image"):
        active_img = st.session_state["current_image"]
        
        if preprocessing_choice != "none":
            preprocessed_img = ImagePreprocessor.apply_preprocessing(active_img, preprocessing_choice)
            tab_orig, tab_prep = st.tabs(["Preprocessed View", "Original View"])
            with tab_orig:
                st.image(preprocessed_img, caption=f"Preprocessed: {PREPROCESSING_METHODS[preprocessing_choice]}", use_container_width=True)
            with tab_prep:
                st.image(active_img, caption="Original Raw Image", use_container_width=True)
            ocr_input_image = preprocessed_img
        else:
            st.image(active_img, caption="Document Preview (Raw Image)", use_container_width=True)
            ocr_input_image = active_img
            
        st.markdown("---")
        
        run_ocr_button = st.button("🚀 Step 1: Run OCR Extraction", type="primary", use_container_width=True)
        
        if run_ocr_button:
            with st.spinner(f"Running {OCR_ENGINES[ocr_engine_name]} on document..."):
                engine = get_ocr_engine(ocr_engine_name)
                extracted_text = ""
                
                if engine.is_available():
                    try:
                        extracted_text = engine.extract_text(ocr_input_image, language)
                        st.success(f"✅ **Live OCR Transcription:** Text extracted dynamically using local {engine.name} engine.")
                    except Exception as e:
                        st.error(f"Live OCR execution failed: {e}")
                else:
                    current_doc_id = st.session_state.get("selected_doc_id")
                    if current_doc_id:
                        lookup_engine = BenchmarkLookupEngine(target_engine=ocr_engine_name)
                        extracted_text = lookup_engine.lookup_by_doc_id(current_doc_id) or ""
                        st.info(
                            f"ℹ️ **Historical Research Benchmark Mode:** Loaded verified {engine.name} "
                            f"transcription from the research archive for `{current_doc_id}`. "
                            f"*(Historical benchmark data, not live OCR).*"
                        )
                    else:
                        st.warning(
                            f"⚠️ **{engine.name}** is not installed locally. "
                            f"To enable live OCR on novel uploads: `pip install easyocr torch`"
                        )
                
                st.session_state["raw_ocr_text"] = extracted_text
                st.session_state["edited_ocr_text"] = extracted_text
                if extracted_text:
                    st.caption("OCR transcription loaded into the editor below.")
    else:
        st.info("👈 Please upload a document or select a benchmark sample to preview.")

st.markdown("---")

# Section 2: Human-in-the-Loop Text Editing & Target Schema
col_schema, col_editor = st.columns([1, 1.2])

with col_schema:
    st.subheader("Target Extraction Schema")
    fields = CATEGORY_SCHEMAS.get(category, [])
    st.markdown(
        f"""
        <div class="schema-box">
            <b>{CATEGORY_NAMES.get(category, category)} Fields:</b><br/>
            <code>{fields}</code>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.write("**Human-in-the-Loop Principle:**")
    st.caption(
        "You can inspect and correct OCR transcription mistakes in the text editor on the right "
        "before proceeding to structured extraction. This prevents OCR transcription errors from "
        "propagating into LLM field extraction."
    )

with col_editor:
    st.subheader("Extracted OCR Text (Editable)")
    
    edited_text = st.text_area(
        "OCR Transcription",
        value=st.session_state.get("edited_ocr_text", ""),
        height=240,
        help="Review and edit transcription errors here prior to running structured extraction."
    )
    st.session_state["edited_ocr_text"] = edited_text
    
    char_count = len(edited_text)
    word_count = len(edited_text.split()) if edited_text.strip() else 0
    line_count = len(edited_text.splitlines()) if edited_text.strip() else 0
    
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    stat_col1.metric("Characters", char_count)
    stat_col2.metric("Words", word_count)
    stat_col3.metric("Lines", line_count)
    with stat_col4:
        st.download_button(
            label="💾 Download OCR Text",
            data=edited_text,
            file_name=f"{st.session_state.get('selected_doc_id', 'ocr_output')}.txt",
            mime="text/plain",
            disabled=(char_count == 0)
        )

st.markdown("---")

# Section 3: Structured Information Extraction
st.subheader("4. Structured Information Extraction (LLM Backend)")

col_model_cfg, col_extract_action = st.columns([1, 1.2])

with col_model_cfg:
    extraction_method = st.selectbox(
        "Extraction Model / Method",
        options=["zero_shot", "few_shot", "experiment1_lora", "experiment2_lora"],
        format_func=lambda x: {
            "zero_shot": "Zero-Shot Prompting (Qwen3-4B)",
            "few_shot": "Few-Shot In-Context Learning (9 Clean Demos)",
            "experiment1_lora": "Experiment 1: Standard QLoRA Adapter",
            "experiment2_lora": "Experiment 2: Evidence-Aware LoRA (Provisional)",
        }.get(x, x),
        index=0,
        help="Select the extraction approach. Experiment 2 is marked provisional pending live GPU verification."
    )
    
    if extraction_method == "experiment2_lora":
        render_provisional_badge()

with col_extract_action:
    st.write("")
    st.write("")
    extract_button = st.button("✨ Step 2: Extract Structured Information", type="primary", use_container_width=True)

if extract_button:
    current_ocr = st.session_state.get("edited_ocr_text", "").strip()
    if not current_ocr:
        st.error("Please run OCR or provide document text before triggering extraction.")
    else:
        with st.spinner(f"Extracting structured fields using {extraction_method}..."):
            backend = get_model_backend()
            selected_doc = st.session_state.get("selected_doc_id")
            
            result = backend.extract_information(
                ocr_text=current_ocr,
                category=category,
                language=language,
                method=extraction_method,
                doc_id=selected_doc,
            )
            st.session_state["extraction_result"] = result
            st.success("Structured extraction completed!")

# Display Extraction Results
if st.session_state.get("extraction_result"):
    res = st.session_state["extraction_result"]
    st.markdown("### 📊 Extraction Results")

    if res.is_historical_benchmark:
        st.info(
            f"ℹ️ **Historical Research Benchmark Record:** Displaying verified research prediction "
            f"from the `Research/BharatDocAI` archive for benchmark sample `{st.session_state.get('selected_doc_id')}`. "
            f"*(This is historical evaluation data, not live model inference).* "
        )
    elif HardwareInfo.is_cuda_available():
        st.success("✅ **Live Model Inference:** Output dynamically generated by the local Qwen3-4B backend.")

    if res.error_message:
        if not res.json_valid:
            st.warning(res.error_message)
        else:
            st.info(res.error_message)
            
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        if res.json_valid:
            st.success("✅ **JSON Validity:** Valid JSON output", icon="✔")
        else:
            st.error("❌ **JSON Validity:** Invalid JSON format", icon="✖")
    with status_col2:
        if res.schema_valid:
            st.success("✅ **Schema Validity:** All 5 required keys matched", icon="✔")
        else:
            st.warning(f"⚠️ **Schema Validity:** Missing/unexpected keys ({res.missing_fields})", icon="⚠")
            
    st.markdown("#### Extracted Key-Value Fields:")
    
    # Display fields in a clean responsive grid
    field_cols = st.columns(len(fields))
    for idx, f in enumerate(fields):
        val = res.prediction.get(f, "")
        with field_cols[idx]:
            st.markdown(
                f"""
                <div class="metric-card" style="text-align: left; padding: 0.8rem;">
                    <div style="font-size: 0.75rem; color: #64748B; font-weight: 700; text-transform: uppercase;">{f}</div>
                    <div style="font-size: 1.1rem; color: #1E293B; font-weight: 600; margin-top: 0.2rem; word-break: break-all;">
                        {val if val else '<span style=\"color:#94A3B8; font-style:italic;\">[Missing/Empty]</span>'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
    # Raw JSON and Export
    col_json, col_raw = st.columns(2)
    with col_json:
        st.subheader("Structured JSON Object")
        st.json(res.prediction)
        export_json_button(
            res.prediction,
            filename=f"extracted_{category}_{language}.json",
            label="📥 Download Extracted JSON"
        )
        
    with col_raw:
        st.subheader("Raw Model Response")
        with st.expander("Inspect Raw Text Response", expanded=False):
            st.code(res.raw_response if res.raw_response else "[No raw response generated]", language="json")
