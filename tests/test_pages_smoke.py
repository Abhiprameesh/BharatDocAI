"""
Smoke Test for BharatDoc AI Pages
Mocks Streamlit UI functions to test that all data operations, Plotly charts,
and research loader pipelines in each page execute cleanly without any runtime errors.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

def mock_streamlit():
    """Mock streamlit display methods so pages can run in headless test without browser."""
    st.set_page_config = MagicMock()
    st.markdown = MagicMock()
    st.write = MagicMock()
    st.caption = MagicMock()
    st.title = MagicMock()
    st.subheader = MagicMock()
    st.header = MagicMock()
    st.info = MagicMock()
    st.warning = MagicMock()
    st.error = MagicMock()
    st.success = MagicMock()
    st.dataframe = MagicMock()
    st.plotly_chart = MagicMock()
    st.image = MagicMock()
    st.json = MagicMock()
    st.download_button = MagicMock()
    st.file_uploader = MagicMock(return_value=None)
    st.selectbox = MagicMock(side_effect=lambda label, options, **kwargs: options[0] if options else None)
    st.multiselect = MagicMock(side_effect=lambda label, options, default=None, **kwargs: default if default is not None else options)
    st.text_area = MagicMock(side_effect=lambda label, value="", **kwargs: value)
    st.metric = MagicMock()
    
    # Mock container / context managers
    col_mock = MagicMock()
    col_mock.__enter__ = MagicMock(return_value=col_mock)
    col_mock.__exit__ = MagicMock(return_value=None)
    col_mock.metric = MagicMock()
    col_mock.info = MagicMock()
    col_mock.success = MagicMock()
    col_mock.warning = MagicMock()
    col_mock.error = MagicMock()
    st.columns = MagicMock(side_effect=lambda n, **kwargs: [col_mock] * (len(n) if isinstance(n, list) else n))
    
    tab_mock = MagicMock()
    tab_mock.__enter__ = MagicMock(return_value=tab_mock)
    tab_mock.__exit__ = MagicMock(return_value=None)
    st.tabs = MagicMock(side_effect=lambda titles: [tab_mock] * len(titles))
    
    sidebar_mock = MagicMock()
    sidebar_mock.__enter__ = MagicMock(return_value=sidebar_mock)
    sidebar_mock.__exit__ = MagicMock(return_value=None)
    st.sidebar = sidebar_mock

def test_page(path: Path):
    print(f"Executing {path.name}...")
    mock_streamlit()
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()
    globs = {"__file__": str(path), "__name__": "__main__"}
    exec(code, globs)
    print(f"✓ {path.name} executed cleanly with 0 errors!")

if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING STREAMLIT PAGES SMOKE TEST")
    print("=" * 60)
    
    pages = [
        ROOT / "app.py",
        ROOT / "pages" / "1_Document_Extraction.py",
        ROOT / "pages" / "2_Model_Comparison.py",
        ROOT / "pages" / "3_Research_Dashboard.py",
        ROOT / "pages" / "4_About_Project.py",
    ]
    
    for page in pages:
        test_page(page)
        
    print("=" * 60)
    print("ALL STREAMLIT PAGES PASSED SMOKE TEST SUCCESSFULLY!")
    print("=" * 60)
