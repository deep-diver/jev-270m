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

"""Typed decisions in JAX, on whatever device JAX finds.

One class, one forward pass per batch of questions, no generation and no
torch. The same code runs on CPU, GPU and TPU because JAX picks the backend.

    from jev_infer import Decider
    d = Decider("google/gemma-3-270m-it", "lora.npz")
    d.ask("The payment page returns a 500 error.", {
        "department": {"type": "choice",
                       "criteria": ["billing", "technical", "something else"]},
        "escalate":   {"type": "noul"},
        "urgency":    {"type": "score",
                       "criteria": ["can wait", "this week", "today", "right now"]},
    })
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MAX_OPTIONS = 8
DEFAULTS = {"noul": ["Yes", "No"]}


class Decider:
    """Holds the backbone, the adapter and the pointer head."""

    def __init__(self, model, ckpt, max_len=320, batch=16):
        import jax
        import jax.numpy as jnp
        from transformers import AutoTokenizer

        from train_lora_tpu import (build_batchable, load_weights, make_model,
                                    stack_layers)

        self.jax, self.jnp = jax, jnp
        self._build_batchable = build_batchable
        self.max_len, self.batch = max_len, batch

        self.tok = AutoTokenizer.from_pretrained(model)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"

        ck = np.load(ckpt)
        self.scale = float(ck["alpha"]) / int(ck["rank"])
        self.params = {
            "lora": {k[5:]: jnp.asarray(ck[k])
                     for k in ck.files if k.startswith("lora.")},
            "head": {k[5:]: jnp.asarray(ck[k])
                     for k in ck.files if k.startswith("head.")},
        }
        self.dev_acc = float(ck["dev_acc"]) if "dev_acc" in ck.files else float("nan")

        weights = load_weights(model, jnp)
        self.const = {"stacked": stack_layers(weights, jnp),
                      "embed": weights["model.embed_tokens.weight"],
                      "final_norm": weights["model.norm.weight"]}
        self.logits_fn, _ = make_model(jnp, jax)
        self.device = jax.devices()[0].platform

    # -- scoring ---------------------------------------------------------
    def _score(self, records):
        jnp = self.jnp
        batch, kept = self._build_batchable(self.tok, records,
                                            self.max_len, MAX_OPTIONS)
        if len(kept) != len(records):
            raise ValueError("a question did not fit in max_len")
        out = []
        n = len(batch["gold"])
        for i in range(0, n, self.batch):
            j = np.arange(i, min(i + self.batch, n))
            b = {k: jnp.asarray(v[j]) for k, v in batch.items()}
            lg = self.logits_fn(self.params, self.const, b,
                                self.max_len, self.scale)
            out.append(np.asarray(self.jax.nn.softmax(lg, -1)))
        return np.concatenate(out)

    # -- public ----------------------------------------------------------
    def ask(self, state, questions):
        """state plus {name: {type, instructions?, criteria?}} -> typed answers."""
        from train_lora_tpu import render  # noqa: F401  (kept for parity)

        names, records, option_lists = [], [], []
        for name, q in questions.items():
            kind = q["type"]
            options = q.get("criteria") or DEFAULTS.get(kind)
            if isinstance(options, dict):
                options = list(options)
            if not options or not 2 <= len(options) <= MAX_OPTIONS:
                raise ValueError(f"{name}: needs 2..{MAX_OPTIONS} options")
            names.append(name)
            option_lists.append(list(options))
            records.append({"state": state,
                            "instructions": q.get("instructions", ""),
                            "options": list(options), "gold": 0})

        probs = self._score(records)
        answers = {}
        for name, options, row in zip(names, option_lists, probs):
            p = row[:len(options)]
            p = p / p.sum()
            kind = questions[name]["type"]
            table = {o: round(float(v), 4) for o, v in zip(options, p)}
            if kind == "noul":
                answers[name] = {"type": "noul", "noul": round(float(p[0]), 4)}
            elif kind == "choice":
                top = int(np.argmax(p))
                answers[name] = {"type": "choice", "choice": options[top],
                                 "confidence": round(float(p[top]), 4),
                                 "probabilities": table}
            else:
                value = float(sum(i * v for i, v in enumerate(p)))
                answers[name] = {"type": "score", "score": round(value, 4),
                                 "legend": {str(i): o for i, o in enumerate(options)},
                                 "probabilities": {str(i): round(float(v), 4)
                                                   for i, v in enumerate(p)}}
        return answers


DEMO = [
    ("The payment page returns a 500 error and customers cannot check out.",),
    ("I was charged twice for the same order and nobody replies.",),
    ("My new card still has not arrived after three weeks.",),
    ("What time does the sun set in Lisbon tomorrow?",),
]

TRIAGE = {
    "department": {"type": "choice",
                   "instructions": "Which team should handle this message?",
                   "criteria": ["billing", "shipping", "returns", "account",
                                "technical", "something else"]},
    "escalate": {"type": "noul",
                 "instructions": "Does this need urgent human attention right now?"},
    "frustration": {"type": "score",
                    "instructions": "How frustrated does the sender sound?",
                    "criteria": ["calm", "annoyed", "frustrated", "furious"]},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-270m-it")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--state", help="score one message instead of the demo set")
    args = ap.parse_args()

    t0 = time.time()
    d = Decider(args.model, args.ckpt)
    print(f"loaded on {d.device} in {time.time() - t0:.1f}s "
          f"(dev acc {d.dev_acc:.3f})\n")

    cases = [(args.state,)] if args.state else DEMO
    for (state,) in cases:
        t = time.time()
        a = d.ask(state, TRIAGE)
        ms = (time.time() - t) * 1000
        print(state)
        dep = a["department"]
        print(f"   department  {dep['choice']:16s} {dep['confidence']:.3f}")
        print(f"   escalate    {a['escalate']['noul']:.3f}")
        f = a["frustration"]
        print(f"   frustration {f['score']:.2f} / 3   "
              f"({f['legend'][str(int(round(f['score'])))]})")
        print(f"   {ms:.0f} ms\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
