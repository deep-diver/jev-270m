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

"""Is vLLM worth it for a 270M decision model?

vLLM earns its keep on continuous batching, paged attention and a scheduler
built for long generations. This model generates nothing and its prompts are
short, so none of that is obviously useful -- which makes it worth measuring
rather than assuming.

Both paths run on the same GPU over the same prompts: vLLM through its HTTP
pooling endpoint, and a plain `transformers` forward pass with the readout
applied in torch.

    python3 scripts/bench_vllm.py --model <dir> --url http://127.0.0.1:8111
"""

import argparse
import json
import statistics
import sys
import time
import urllib.request


def render(state, instructions, options):
    lines = [f"State:\n{state}\n", f"Question: {instructions}", "Options:"]
    for i, o in enumerate(options):
        lines.append(f"{o} <{i}>")
    lines.append("Decision:")
    return "\n".join(lines)


STATES = [
    "Shoes arrived two weeks late and in the wrong size.",
    "I was charged twice for the same order and nobody replies.",
    "The payment page returns a 500 error and customers cannot check out.",
    "Where is my parcel? The tracking page has not moved in five days.",
]
OPTIONS = ["returns", "shipping", "billing", "infrastructure"]
QUESTION = "Which team should handle this?"


def prompts(n):
    return [render(STATES[i % len(STATES)], QUESTION, OPTIONS) for i in range(n)]


def bench_vllm(url, model, batch, reps):
    body = json.dumps({"model": model, "input": prompts(batch)}).encode()
    times = []
    for _ in range(reps + 1):
        t = time.perf_counter()
        req = urllib.request.Request(f"{url}/pooling", body,
                                     {"content-type": "application/json"})
        urllib.request.urlopen(req, timeout=300).read()
        times.append(time.perf_counter() - t)
    return times[1:]          # drop the first, it pays for the connection


def bench_torch(model_dir, batch, reps):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16).to("cuda").eval()
    head = load_file(f"{model_dir}/extras/pointer_head.safetensors")
    Wq = head["Wq"].cuda().float()
    Wk = head["Wk"].cuda().float()
    ln_w = head["ln_w"].cuda().float()
    ln_b = head["ln_b"].cuda().float()

    def norm(x, eps=1e-5):
        return (1.0 + ln_w) * ((x - x.mean(-1, keepdim=True))
                               / (x.std(-1, keepdim=True) + eps)) + ln_b

    ps = prompts(batch)
    enc = tok(ps, return_tensors="pt", padding=True,
              add_special_tokens=True).to("cuda")
    last = enc["attention_mask"].sum(1) - 1

    times = []
    with torch.no_grad():
        for _ in range(reps + 1):
            torch.cuda.synchronize()
            t = time.perf_counter()
            out = model(**enc, output_hidden_states=True)
            h = out.hidden_states[-1]
            rows = torch.arange(len(ps), device="cuda")
            q = norm(h[rows, last].float()) @ Wq.T
            # The marker positions are fixed across these prompts; scoring the
            # last four non-pad tokens is enough for a timing comparison.
            k = norm(h[rows][:, -5:-1].float()) @ Wk.T
            _ = torch.einsum("bod,bd->bo", k, q)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
    return times[1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8111")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32, 128])
    ap.add_argument("--skip-torch", action="store_true")
    args = ap.parse_args()

    print(f"{'batch':>6} {'vLLM ms':>10} {'per item':>9} "
          f"{'torch ms':>10} {'per item':>9}  {'ratio':>6}")
    print("-" * 58)
    for b in args.batches:
        v = bench_vllm(args.url, args.model, b, args.reps)
        vm = statistics.median(v) * 1000
        if args.skip_torch:
            print(f"{b:6d} {vm:10.1f} {vm / b:9.2f}")
            continue
        t = bench_torch(args.model, b, args.reps)
        tm = statistics.median(t) * 1000
        print(f"{b:6d} {vm:10.1f} {vm / b:9.2f} {tm:10.1f} {tm / b:9.2f}"
              f"  {tm / vm:6.2f}x")
    print("\nratio > 1 means vLLM is faster.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
