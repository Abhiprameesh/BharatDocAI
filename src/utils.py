"""
BharatDoc AI - Utility Helpers
Provides formatting, alert banners, JSON/CSV exports, and styling helpers.
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd

def load_css(css_path: Path):
    """Inject custom CSS into Streamlit page."""
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def render_header(title: str, subtitle: str = None, badge: str = None):
    """Render a consistent, clean page header."""
    badge_html = f'<span class="badge">{badge}</span>' if badge else ""
    st.markdown(
        f"""
        <div class="custom-header">
            <h1 style="margin-bottom: 0.2rem; font-weight: 700; color: #1E293B;">
                {title} {badge_html}
            </h1>
            {f'<p style="color: #64748B; font-size: 1.1rem; margin-top: 0.2rem;">{subtitle}</p>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")

def render_provisional_badge():
    """Render scientific note for Experiment 2 status."""
    st.info(
        "🔬 **Experiment 2 Status (GPU-Verified):** "
        "Experiment 2 test inference was executed with the trained `qwen3_lora_experiment2` adapter. "
        "It achieves **75.19% field accuracy** and **35.19% document exact match**, successfully avoiding "
        "the over-abstention penalty of Experiment 1 (69.63%), while Few-Shot In-Context Learning (83.33%) "
        "remains the overall benchmark leader.",
        icon="📊"
    )

def export_json_button(data: dict, filename: str = "extracted_data.json", label: str = "Download Extracted JSON"):
    """Render a download button for JSON data."""
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    st.download_button(
        label=label,
        data=json_str,
        file_name=filename,
        mime="application/json"
    )

def export_csv_button(df: pd.DataFrame, filename: str = "data.csv", label: str = "Download CSV"):
    """Render a download button for Pandas DataFrame."""
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=filename,
        mime="text/csv"
    )
