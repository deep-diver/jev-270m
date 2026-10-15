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
Serve the typed-decision model on vLLM as a pooling model.

vLLM's released endpoints answer with generated text, and this model answers
with a dot product between the hidden state at `Decision:` and the hidden state
at each option's marker. That is not a next-token read, so neither
`/v1/completions` nor the merged `/v1/systemone` next-token strategy can
produce it.

What does fit is the pooling path, which is how the Laya PR serves the same
shape of model: register an architecture out of tree, return hidden states
instead of logits, and put the readout in a pooler.

The adapter is expected to be merged into the weights already
(`scripts/merge_lora.py`), so nothing here needs LoRA at runtime -- only the
pointer head, which is a second readout rather than a correction to the
backbone.

Marker positions are recovered from token ids rather than from text: the
renderer writes `<0>`, `<1>`, ... after each option, and training read the
hidden state at the closing `>`. The same three-token pattern is searched for
here, so serving scores exactly the positions training did.

    python3 scripts/vllm_jev_plugin.py --model /path/to/merged --selftest
"""

import argparse
import json
import os
import sys

import numpy as np
import torch


def render(state, instructions, options):
    """Identical to the training renderer; the marker must trail its option."""
    lines = [f"State:\n{state}\n", f"Question: {instructions}", "Options:"]
    for i, o in enumerate(options):
        lines.append(f"{o} <{i}>")
    lines.append("Decision:")
    return "\n".join(lines)


def marker_positions(ids, tok, n_options):
    """Index of the closing `>` of each `<i>` marker.

    Matching raw ids does not work: the tokenizer folds the preceding space
    into `<`, so the id differs from the one `tok.encode("<")` returns. The
    decoded pieces are matched instead, which is exact and costs one decode
    over a prompt-length list.
    """
    pieces = [tok.decode([i]) for i in ids]
    found = {}
    for p in range(2, len(pieces)):
        if pieces[p].strip() != ">":
            continue
        digit, bracket = pieces[p - 1].strip(), pieces[p - 2].strip()
        if bracket.endswith("<") and digit.isdigit():
            idx = int(digit)
            if idx < n_options:
                found[idx] = p
    return [found.get(i) for i in range(n_options)]


class PointerReadout:
    """The trained readout, applied to hidden states vLLM hands back."""

    def __init__(self, directory, device="cpu"):
        from safetensors.torch import load_file
        head = load_file(os.path.join(directory, "extras", "pointer_head.safetensors"))
        self.Wq = head["Wq"].to(device).float()
        self.Wk = head["Wk"].to(device).float()
        self.ln_w = head["ln_w"].to(device).float()
        self.ln_b = head["ln_b"].to(device).float()
        self.dim = self.Wq.shape[0]

    def _norm(self, x, eps=1e-5):
        mu = x.mean(-1, keepdim=True)
        sd = x.std(-1, keepdim=True)
        return (1.0 + self.ln_w) * ((x - mu) / (sd + eps)) + self.ln_b

    def scores(self, hidden, decision_pos, marker_pos):
        """hidden: (seq, hidden). Returns one logit per option."""
        q = self._norm(hidden[decision_pos].float()) @ self.Wq.T
        k = self._norm(hidden[list(marker_pos)].float()) @ self.Wk.T
        return (k @ q) / (self.dim ** 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="merged checkpoint directory")
    ap.add_argument("--selftest", action="store_true",
                    help="run the readout against HuggingFace hidden states")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    readout = PointerReadout(args.model)
    print(f"pointer head: dim {readout.dim}, hidden {readout.Wq.shape[1]}")

    text = render("Shoes arrived two weeks late and in the wrong size.",
                  "Which team should handle this?",
                  ["Returns and refunds", "Shipping and delivery", "Billing"])
    ids = tok.encode(text, add_special_tokens=True)
    pos = marker_positions(ids, tok, 3)
    print(f"{len(ids)} tokens; markers at {pos}; decision at {len(ids) - 1}")
    assert all(p is not None for p in pos), "a marker was not found by id"
    for i, p in enumerate(pos):
        print(f"  option {i}: ...{tok.decode(ids[max(0, p - 5):p + 1])!r}")

    if args.selftest:
        # Hidden states from the merged checkpoint, read exactly as training did.
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float32).eval()
        with torch.no_grad():
            out = model(torch.tensor([ids]), output_hidden_states=True)
        hidden = out.hidden_states[-1][0]        # post-final-norm, as trained
        logits = readout.scores(hidden, len(ids) - 1, pos)
        p = torch.softmax(logits, 0)
        print("\nreadout over the merged checkpoint:")
        for i, (o, v) in enumerate(zip(["Returns and refunds",
                                        "Shipping and delivery", "Billing"],
                                       p.tolist())):
            print(f"  {v:.3f}  {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
