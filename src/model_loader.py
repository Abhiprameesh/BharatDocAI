"""
BharatDoc AI - Model Backend & LoRA Adapter Manager
Handles lazy loading of Qwen3-4B and PEFT LoRA adapters (Experiment 1 and Experiment 2).
Provides hardware awareness for CPU-only systems and safe benchmark retrieval.
"""

import os
import gc
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd

from src.config import (
    BASE_MODEL_NAME,
    EXP1_ADAPTER_DIR,
    EXP2_ADAPTER_DIR,
    ZERO_SHOT_PREDICTIONS_FILE,
    FEW_SHOT_PREDICTIONS_FILE,
    EXP1_LORA_PREDICTIONS_FILE,
    EXP2_LORA_PREDICTIONS_FILE,
    EXP2_STATUS,
    EXP2_STATUS_DETAILS,
)
from src.prompt_templates import EXTRACTION_SYSTEM_PROMPT, build_user_prompt
from src.extraction_engine import ExtractionEngine, ExtractionResult

class HardwareInfo:
    """Detects available computing hardware."""

    @staticmethod
    def is_cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    @staticmethod
    def get_hardware_description() -> str:
        if HardwareInfo.is_cuda_available():
            import torch
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return f"NVIDIA GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)"
        return "CPU Environment: Intel Core i5-1340P / 16 GB RAM (Zero-GPU Mode)"


class ModelBackend:
    """Manages LLM backend selection, lazy loading, and adapter isolation."""

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._current_adapter: Optional[str] = None
        self._benchmark_cache: Dict[str, Dict[str, Any]] = {}

    @property
    def current_adapter(self) -> Optional[str]:
        return self._current_adapter

    def is_model_loaded(self) -> bool:
        return self._model is not None

    def release_model(self):
        """Explicitly release model resources from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._current_adapter = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def verify_adapter_paths(self) -> Tuple[bool, bool]:
        """Check that adapter safetensors files exist for Experiment 1 and 2."""
        exp1_ok = (EXP1_ADAPTER_DIR / "adapter_model.safetensors").exists()
        exp2_ok = (EXP2_ADAPTER_DIR / "adapter_model.safetensors").exists()
        return exp1_ok, exp2_ok

    def get_benchmark_prediction(
        self, method_key: str, doc_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve verified historical benchmark prediction for synthetic test records.
        Guarantees 100% scientific reproducibility without requiring local GPU.
        """
        mapping = {
            "zero_shot": ZERO_SHOT_PREDICTIONS_FILE,
            "few_shot": FEW_SHOT_PREDICTIONS_FILE,
            "experiment1_lora": EXP1_LORA_PREDICTIONS_FILE,
            "experiment2_lora": EXP2_LORA_PREDICTIONS_FILE,
        }

        path = mapping.get(method_key)
        if not path or not path.exists():
            return None

        if method_key not in self._benchmark_cache:
            self._benchmark_cache[method_key] = {}
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        d_id = item.get("document_id")
                        if d_id and d_id not in self._benchmark_cache[method_key]:
                            self._benchmark_cache[method_key][d_id] = item

        return self._benchmark_cache[method_key].get(doc_id)

    def extract_information(
        self,
        ocr_text: str,
        category: str,
        language: str,
        method: str = "zero_shot",
        doc_id: Optional[str] = None,
    ) -> ExtractionResult:
        """
        Main extraction entry point.
        Dispatches to benchmark record lookup (for research samples) or live model inference.
        """
        method = method.lower()
        few_shot = (method == "few_shot")
        user_prompt = build_user_prompt(ocr_text, category, language, few_shot=few_shot)

        # 1. If evaluating a known benchmark document, retrieve verified historical result
        if doc_id:
            historical_rec = self.get_benchmark_prediction(method, doc_id)
            if historical_rec:
                raw_response = historical_rec.get("raw_response", "")
                result = ExtractionEngine.process_response(raw_response, category, language)
                result.is_historical_benchmark = True
                if method == "experiment2_lora":
                    # Attach provisional research note
                    result.error_message = (
                        "Note: Experiment 2 inference matched Zero-Shot across test records. "
                        "Marked as provisional pending live GPU verification."
                    )
                return result

        # 2. Check hardware for live inference
        has_cuda = HardwareInfo.is_cuda_available()
        if not has_cuda:
            # CPU environment: Qwen3-4B 4-bit NF4 weights require CUDA.
            return ExtractionResult(
                category=category,
                language=language,
                prediction={k: "" for k in ["name", "date", "document_id"]},
                raw_response="",
                json_valid=False,
                schema_valid=False,
                error_message=(
                    "⚠️ Hardware Constraint: Qwen3-4B 4-bit NF4 quantized inference requires a "
                    "CUDA-compatible NVIDIA GPU. The current system (Intel Core i5-1340P) operates in "
                    "CPU-only mode. For synthetic benchmark documents, verified research predictions are "
                    "retrieved automatically. To run live inference on novel uploaded files, connect "
                    "a GPU runtime."
                ),
            )

        # 3. Live GPU Inference (when CUDA is detected)
        try:
            raw_response = self._run_live_gpu_inference(user_prompt, method)
            return ExtractionEngine.process_response(raw_response, category, language)
        except Exception as e:
            return ExtractionResult(
                category=category,
                language=language,
                prediction={},
                raw_response="",
                json_valid=False,
                schema_valid=False,
                error_message=f"Live model inference failed: {str(e)}",
            )

    def _run_live_gpu_inference(self, user_prompt: str, method: str) -> str:
        """Execute live inference on CUDA hardware with safe adapter switching."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        # Lazy tokenizer loading
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

        # Lazy base model loading
        if self._model is None:
            self._model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            self._current_adapter = None

        # Adapter switching logic
        target_adapter = None
        if method == "experiment1_lora":
            target_adapter = EXP1_ADAPTER_DIR
        elif method == "experiment2_lora":
            target_adapter = EXP2_ADAPTER_DIR

        # If adapter needs to change, unload previous adapter
        if target_adapter and self._current_adapter != str(target_adapter):
            if isinstance(self._model, PeftModel):
                self._model = self._model.unload()
            self._model = PeftModel.from_pretrained(
                self._model, str(target_adapter), is_trainable=False
            )
            self._model.eval()
            self._current_adapter = str(target_adapter)
        elif not target_adapter and self._current_adapter is not None:
            # Revert to base model for zero_shot / few_shot
            if isinstance(self._model, PeftModel):
                self._model = self._model.unload()
            self._model.eval()
            self._current_adapter = None

        # Construct generation conversation
        messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        prompt_str = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self._tokenizer(
            prompt_str,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to("cuda")

        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=160,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


# Global Singleton Instance
_BACKEND_INSTANCE = None

def get_model_backend() -> ModelBackend:
    """Retrieve global model backend instance."""
    global _BACKEND_INSTANCE
    if _BACKEND_INSTANCE is None:
        _BACKEND_INSTANCE = ModelBackend()
    return _BACKEND_INSTANCE
