# Copyright 2026 Chansung Park
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Register the typed-decision architecture with vLLM.

The model directory is read from JEV_MODEL_DIR because the pooler needs the
pointer head, which does not live in the checkpoint vLLM loads.
"""
import os


def register():
    from vllm.model_executor.models.registry import ModelRegistry

    if "Gemma3ForDecision" in ModelRegistry.get_supported_archs():
        return
    ModelRegistry.register_model("Gemma3ForDecision",
                                 "jev_vllm.model:Gemma3ForDecision")
    d = os.environ.get("JEV_MODEL_DIR")
    print(f"[jev] registered Gemma3ForDecision (model dir: {d})", flush=True)
