# Copyright 2026 Chansung Park
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The typed-decision readout as a vLLM pooling model.

The backbone is an ordinary Gemma 3 with the adapter already folded in
(`scripts/merge_lora.py`). What is not ordinary is where the answer comes
from: a dot product between the hidden state at `Decision:` and the hidden
state at each option's marker. vLLM's released endpoints read the next token,
which is the one channel this model has nothing in, so the readout lives in a
`Pooler` instead -- the same place the Laya PR puts its own.

Loaded through the `vllm.general_plugins` entry point so that the engine core
subprocess registers it too; registering from the parent process alone leaves
the engine unable to find the architecture.
"""

import os

import torch

MAX_OPTIONS = 8


def _model_dir() -> str:
    d = os.environ.get("JEV_MODEL_DIR")
    if not d:
        raise RuntimeError("set JEV_MODEL_DIR to the merged checkpoint directory")
    return d


def render(state: str, instructions: str, options) -> str:
    """The renderer training used. The marker has to trail its option: with
    causal attention a leading marker summarises everything except the option
    it stands for."""
    lines = [f"State:\n{state}\n", f"Question: {instructions}", "Options:"]
    for i, o in enumerate(options):
        lines.append(f"{o} <{i}>")
    lines.append("Decision:")
    return "\n".join(lines)


def marker_token_ids(tok, n: int = MAX_OPTIONS):
    """The id triple each `<i>` becomes *in context*.

    Encoding "<" on its own gives an id that never appears in a rendered
    prompt: the tokenizer folds the preceding space into that token. Capturing
    the triple from a realistic snippet avoids the mismatch.
    """
    return {i: tuple(tok.encode(f"x <{i}>", add_special_tokens=False)[-3:])
            for i in range(n)}


def build_classes(model_dir: str):
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from vllm.model_executor.layers.pooler import PoolingParamsUpdate
    from vllm.model_executor.layers.pooler.abstract import Pooler
    from vllm.model_executor.models.adapters import _create_pooling_model_cls
    from vllm.model_executor.models.gemma3 import Gemma3ForCausalLM

    triples = marker_token_ids(AutoTokenizer.from_pretrained(model_dir))

    class PointerPooler(Pooler):
        """Scores every option marker against the decision token."""

        def __init__(self, directory: str):
            super().__init__()
            head = load_file(
                os.path.join(directory, "extras", "pointer_head.safetensors"))
            for k, v in head.items():
                self.register_buffer(k, v.float(), persistent=False)
            self.dim = head["Wq"].shape[0]

        def get_supported_tasks(self):
            # A set, not a tuple: the runner intersects this with its own.
            # "encode" is rejected by the V2 runner; "token_classify" is the
            # shape a per-option score fits.
            return {"token_classify"}

        def get_pooling_updates(self, task):
            # Without the prompt the pooler cannot tell which positions are
            # markers, and this is the only way to be handed it.
            return PoolingParamsUpdate(requires_token_ids=True)

        def _norm(self, x, eps: float = 1e-5):
            mu = x.mean(-1, keepdim=True)
            sd = x.std(-1, keepdim=True)
            return (1.0 + self.ln_w) * ((x - mu) / (sd + eps)) + self.ln_b

        def forward(self, hidden_states, pooling_metadata):
            # Buffers load on CPU; hidden states arrive on the device.
            if self.Wq.device != hidden_states.device:
                self.to(hidden_states.device)

            lens = pooling_metadata.prompt_lens.tolist()
            toks = pooling_metadata.prompt_token_ids_cpu
            if toks is None:
                toks = pooling_metadata.prompt_token_ids

            # With chunked prefill the engine can hand over part of a sequence,
            # so the rows do not always add up to what hidden_states holds.
            # Trust the cursor when it is there and fall back to the prompt
            # lengths when it is not.
            cursor = getattr(pooling_metadata, "pooling_cursor", None)
            chunks = None
            if cursor is not None:
                n_sched = getattr(cursor, "num_scheduled_tokens_cpu", None)
                if n_sched is not None:
                    chunks = n_sched.tolist()
            if chunks is None or sum(chunks) != hidden_states.shape[0]:
                chunks = lens

            out, start = [], 0
            for row, n in enumerate(lens):
                take = chunks[row] if row < len(chunks) else n
                h = hidden_states[start:start + take]
                ids = toks[row][:take].tolist()
                start += take
                if h.shape[0] < n:
                    # Only a prefix of this sequence arrived; the decision
                    # token is not in it yet.
                    out.append(torch.zeros(0, device=hidden_states.device))
                    continue
                n = h.shape[0]

                marks = {}
                for opt, (a, b, c) in triples.items():
                    for p in range(n - 2):
                        if ids[p] == a and ids[p + 1] == b and ids[p + 2] == c:
                            marks[opt] = p + 2
                            break
                if not marks:
                    # A prompt with no markers (vLLM's profiling dummies, for
                    # one) must not raise inside the engine.
                    out.append(torch.zeros(0, device=hidden_states.device))
                    continue

                order = sorted(marks)
                q = self._norm(h[-1].float()) @ self.Wq.T
                k = self._norm(h[[marks[i] for i in order]].float()) @ self.Wk.T
                out.append((k @ q) / (self.dim ** 0.5))
            return out

    Base = _create_pooling_model_cls(Gemma3ForCausalLM)

    class Gemma3ForDecision(Base):
        def _init_pooler(self, vllm_config, prefix: str = ""):
            return PointerPooler(model_dir)

    return Gemma3ForDecision


# The cache deliberately does NOT share the public name: a module-level
# `Gemma3ForDecision = None` would satisfy the attribute lookup and module
# __getattr__ would never run, handing vLLM a None class.
_CLS = None


def __getattr__(name):
    # Built on first access so importing this module does not require vLLM.
    global _CLS
    if name == "Gemma3ForDecision":
        if _CLS is None:
            _CLS = build_classes(_model_dir())
        return _CLS
    raise AttributeError(name)
