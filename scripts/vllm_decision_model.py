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

"""
Run the typed-decision readout inside the vLLM engine.

vLLM 0.31.0 serves text. The merged PRs that add `/v1/systemone` read the next
token, which is the channel this model has nothing in. What it does have is a
dot product between the hidden state at `Decision:` and the hidden state at
each option marker -- a pooling operation, which is how the Laya PR serves the
same shape of model.

So the backbone is registered out of tree as a pooling architecture and the
readout lives in a `Pooler`. The engine keeps batching and scheduling; only
the last step differs.

Two details decide whether this works at all:

- `PoolingParamsUpdate(requires_token_ids=True)` is what gets the prompt tokens
  into the pooler. Without them it cannot tell which positions are markers.
- Markers are matched as a token-id triple captured at startup, not by encoding
  "<" on its own: the tokenizer folds the preceding space into that token, so
  the standalone id never appears in a rendered prompt.

    python3 scripts/vllm_decision_model.py --model /path/to/merged
"""

import argparse
import os
import sys

import torch
import torch.nn as nn

MAX_OPTIONS = 8


def render(state, instructions, options):
    lines = [f"State:\n{state}\n", f"Question: {instructions}", "Options:"]
    for i, o in enumerate(options):
        lines.append(f"{o} <{i}>")
    lines.append("Decision:")
    return "\n".join(lines)


def marker_token_ids(tok, n=MAX_OPTIONS):
    """The id triple each `<i>` becomes *in context*, with the leading space."""
    out = {}
    for i in range(n):
        ids = tok.encode(f"x <{i}>", add_special_tokens=False)
        out[i] = tuple(ids[-3:])
    return out


def build(model_dir):
    from vllm.model_executor.layers.pooler import PoolingParamsUpdate
    from vllm.model_executor.layers.pooler.abstract import Pooler
    from vllm.model_executor.models.adapters import _create_pooling_model_cls
    from vllm.model_executor.models.gemma3 import Gemma3ForCausalLM
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    TRIPLES = marker_token_ids(tok)

    class PointerPooler(Pooler):
        """Scores each option marker against the decision token."""

        def __init__(self, directory):
            super().__init__()
            head = load_file(os.path.join(directory, "extras", "pointer_head.safetensors"))
            for k, v in head.items():
                self.register_buffer(k, v.float(), persistent=False)
            self.dim = head["Wq"].shape[0]

        def get_supported_tasks(self):
            # Not "encode": the V2 model runner only accepts
            # classify / embed / token_classify / token_embed, and this is a
            # per-option score, which is the shape token_classify describes.
            return {"token_classify"}

        def get_pooling_updates(self, task):
            # Without this the pooler never sees the prompt and cannot locate
            # the markers.
            return PoolingParamsUpdate(requires_token_ids=True)

        def _norm(self, x, eps=1e-5):
            mu = x.mean(-1, keepdim=True)
            sd = x.std(-1, keepdim=True)
            return (1.0 + self.ln_w) * ((x - mu) / (sd + eps)) + self.ln_b

        def forward(self, hidden_states, pooling_metadata):
            # Buffers are built on CPU at load time; the engine hands back GPU
            # hidden states, so move once on first use.
            if self.Wq.device != hidden_states.device:
                self.to(hidden_states.device)
            lens = pooling_metadata.prompt_lens.tolist()
            toks = pooling_metadata.prompt_token_ids_cpu
            if toks is None:
                toks = pooling_metadata.prompt_token_ids
            out, start = [], 0
            for row, n in enumerate(lens):
                h = hidden_states[start:start + n]
                ids = toks[row][:n].tolist()
                start += n

                marks = {}
                for opt, (a, b, c) in TRIPLES.items():
                    for p in range(n - 2):
                        if ids[p] == a and ids[p + 1] == b and ids[p + 2] == c:
                            marks[opt] = p + 2
                            break
                if not marks:
                    out.append(torch.zeros(0, device=hidden_states.device))
                    continue

                order = sorted(marks)
                q = self._norm(h[n - 1].float()) @ self.Wq.T
                k = self._norm(h[[marks[i] for i in order]].float()) @ self.Wk.T
                out.append((k @ q) / (self.dim ** 0.5))
            return out

    Base = _create_pooling_model_cls(Gemma3ForCausalLM)

    class Gemma3ForDecision(Base):
        def _init_pooler(self, vllm_config, prefix=""):
            return PointerPooler(model_dir)

    return Gemma3ForDecision, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    args = ap.parse_args()

    from vllm import LLM
    from vllm.model_executor.models.registry import ModelRegistry

    cls, tok = build(args.model)
    ModelRegistry.register_model("Gemma3ForDecision", cls)
    print("registered Gemma3ForDecision")

    llm = LLM(model=args.model, runner="pooling",
              hf_overrides={"architectures": ["Gemma3ForDecision"]},
              max_model_len=1024, enforce_eager=True,
              gpu_memory_utilization=args.gpu_memory_utilization)

    cases = [
        ("Shoes arrived two weeks late and in the wrong size.",
         "Which team should handle this?",
         ["Returns and refunds", "Shipping and delivery", "Billing"]),
        ("I was charged twice for the same order and nobody replies.",
         "Which team should handle this?",
         ["Returns and refunds", "Shipping and delivery", "Billing"]),
        ("The lander touches down near the lunar south pole.",
         "This is about space or astronomy.\n\nDoes this match?",
         ["Yes", "No"]),
        ("A history of sourdough baking in northern Europe.",
         "This is about space or astronomy.\n\nDoes this match?",
         ["Yes", "No"]),
    ]
    prompts = [render(*c) for c in cases]
    outs = llm.encode(prompts, pooling_task="token_classify")

    print()
    for (state, q, options), o in zip(cases, outs):
        scores = o.outputs.data.float()
        p = torch.softmax(scores, 0)
        print(f"{state[:58]}")
        for name, v in zip(options, p.tolist()):
            print(f"    {v:.3f}  {name}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
