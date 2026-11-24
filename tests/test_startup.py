"""
Stage 2 Startup & Configuration Verification Test
Validates:
1. Module imports and relative path resolution
2. Ground truth schemas (specifically 'institution' in certificate)
3. Precomputed dataset and metric loading
4. Dynamic 4-way evaluation calculation (CPU-only, Zero-GPU)
"""

import sys
from pathlib import Path

# Ensure project root is in python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def test_config_paths():
    from src.config import (
        PROJECT_ROOT,
        RESEARCH_DIR,
        ANNOTATIONS_FILE,
        DATASET_MANIFEST_FILE,
        DEGRADATION_MANIFEST_FILE,
        THREE_ENGINE_OVERALL_FILE,
        FOUR_WAY_PREDICTIONS_DIR,
        CATEGORY_SCHEMAS,
    )
    
    print(f"Project Root: {PROJECT_ROOT}")
    assert RESEARCH_DIR.exists(), f"Research directory missing: {RESEARCH_DIR}"
    assert ANNOTATIONS_FILE.exists(), f"Annotations missing: {ANNOTATIONS_FILE}"
    assert DATASET_MANIFEST_FILE.exists(), f"Dataset manifest missing: {DATASET_MANIFEST_FILE}"
    assert DEGRADATION_MANIFEST_FILE.exists(), f"Degradation manifest missing: {DEGRADATION_MANIFEST_FILE}"
    assert THREE_ENGINE_OVERALL_FILE.exists(), f"OCR metrics missing: {THREE_ENGINE_OVERALL_FILE}"
    assert FOUR_WAY_PREDICTIONS_DIR.exists(), f"Four-way predictions missing: {FOUR_WAY_PREDICTIONS_DIR}"
    
    # Schema check
    assert "institution" in CATEGORY_SCHEMAS["certificate"], "Certificate schema must include 'institution'!"
    assert "amount" in CATEGORY_SCHEMAS["invoice"], "Invoice schema must include 'amount'!"
    assert "phone" in CATEGORY_SCHEMAS["application"], "Application schema must include 'phone'!"
    print("✓ All configuration paths and schemas verified.")

def test_data_loader():
    from src.research_analysis import ResearchDataLoader
    
    manifest = ResearchDataLoader.load_dataset_manifest()
    assert len(manifest) == 90, f"Expected 90 source docs, got {len(manifest)}"
    
    degrad = ResearchDataLoader.load_degradation_manifest()
    assert len(degrad) == 1440, f"Expected 1440 image records, got {len(degrad)}"
    
    ocr = ResearchDataLoader.load_ocr_overall_metrics()
    assert len(ocr) == 3, f"Expected 3 OCR engines, got {len(ocr)}"
    
    boot = ResearchDataLoader.load_bootstrap_comparisons()
    assert len(boot) > 0, "Bootstrap comparisons should not be empty"
    
    print(f"✓ Research data loader verified: 90 source docs, 1,440 images, 3 OCR engines.")

def test_four_way_evaluation():
    from src.research_analysis import ResearchDataLoader
    
    eval_df = ResearchDataLoader.evaluate_four_way_comparison()
    assert len(eval_df) == 4, f"Expected 4 models evaluated, got {len(eval_df)}"
    
    # Verify that field accuracy is calculated accurately
    accuracies = dict(zip(eval_df["Model / Method"], eval_df["Field-Level Exact Match (%)"]))
    print(f"Field Accuracies: {accuracies}")
    
    assert accuracies["Zero-Shot Qwen3-4B"] == 75.19, f"Unexpected ZS accuracy: {accuracies['Zero-Shot Qwen3-4B']}"
    assert accuracies["Few-Shot Qwen3-4B"] == 83.33, f"Unexpected FS accuracy: {accuracies['Few-Shot Qwen3-4B']}"
    assert accuracies["Experiment 1 LoRA"] == 69.63, f"Unexpected Exp1 accuracy: {accuracies['Experiment 1 LoRA']}"
    assert accuracies["Experiment 2 LoRA (Provisional)"] == 75.19, f"Unexpected Exp2 accuracy: {accuracies['Experiment 2 LoRA (Provisional)']}"
    
    print("✓ 4-Way extraction evaluation verified with exact matching metrics.")

if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING BHARATDOC AI STARTUP & CONFIGURATION TESTS")
    print("=" * 60)
    test_config_paths()
    test_data_loader()
    test_four_way_evaluation()
    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
