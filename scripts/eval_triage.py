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

"""Does the triage model actually work?

Four questions a demo cannot answer:

1. accuracy on the held-out split, broken out by question type and source --
   an average hides which half is carrying it
2. whether it declines when it should: messages written here, not drawn from
   any corpus, that belong to none of the departments
3. whether it declines when it should *not*: in-scope messages must not fall
   through to the escape option
4. whether the answer survives shuffling the options, which is the cheapest
   test for a model that has learned positions rather than content

    python3 scripts/eval_triage.py --model <dir> --ckpt triage.npz \\
        --data data/triage/dev.jsonl
"""

import argparse
import collections
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ESCAPE = "something else"
DEPARTMENTS = ["billing", "shipping", "returns", "account", "technical"]
DEPARTMENT_Q = "Which team should handle this message?"

# Written here rather than sampled, so nothing in training resembles them.
OUT_OF_SCOPE = [
    "What time does the sun set in Lisbon tomorrow?",
    "Can you recommend a good sourdough recipe?",
    "Who won the match last night?",
    "Translate 'good morning' into Portuguese please.",
    "My dog has been coughing for two days, should I worry?",
    "What is the capital of Mongolia?",
    "Please write me a poem about the sea.",
    "How do I change a flat tyre?",
]

IN_SCOPE = [
    ("I was charged twice for the same transaction.", "billing"),
    ("There is an extra fee on my statement I did not expect.", "billing"),
    ("My replacement card has not arrived after three weeks.", "shipping"),
    ("When will my new card be delivered?", "shipping"),
    ("I want my money back for a payment I cancelled.", "returns"),
    ("The refund I was promised has not shown up.", "returns"),
    ("I need to change the phone number on my account.", "account"),
    ("I forgot my passcode and cannot log in.", "account"),
    ("My transfer is still pending after two days.", "technical"),
    ("The top up failed but the money left my account.", "technical"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-270m-it")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from jev_infer import Decider
    d = Decider(args.model, args.ckpt)
    print(f"{d.device} · dev acc at training time {d.dev_acc:.3f}\n")

    # -- 1. the held-out split ------------------------------------------
    rows = [json.loads(l) for l in open(args.data)]
    if args.limit:
        rows = rows[:args.limit]
    probs = d._score([{"state": r["state"], "instructions": r["instructions"],
                       "options": r["options"], "gold": 0} for r in rows])

    by_type = collections.defaultdict(lambda: [0, 0])
    by_src = collections.defaultdict(lambda: [0, 0])
    off_by = []
    for r, p in zip(rows, probs):
        n = len(r["options"])
        pred = int(np.argmax(p[:n]))
        ok = pred == r["gold"]
        for d_ in (by_type[r["type"]], by_src[r["src"].split(":")[0]]):
            d_[0] += ok
            d_[1] += 1
        if r["type"] == "score":
            off_by.append(abs(pred - r["gold"]))

    print(f"held out: {len(rows)} questions")
    print(f"  {'type':10s} {'n':>6s} {'acc':>7s}")
    for k in sorted(by_type):
        c, n = by_type[k]
        print(f"  {k:10s} {n:6d} {c / n:7.1%}")
    print(f"\n  {'source':10s} {'n':>6s} {'acc':>7s}")
    for k in sorted(by_src):
        c, n = by_src[k]
        print(f"  {k:10s} {n:6d} {c / n:7.1%}")
    if off_by:
        exact = sum(1 for v in off_by if v == 0) / len(off_by)
        within1 = sum(1 for v in off_by if v <= 1) / len(off_by)
        print(f"\n  score: exact {exact:.1%}, within one level {within1:.1%}")

    # -- 2 & 3. declining, and not declining ----------------------------
    menu = DEPARTMENTS + [ESCAPE]
    q = {"department": {"type": "choice", "instructions": DEPARTMENT_Q,
                        "criteria": menu}}

    print(f"\nout of scope ({len(OUT_OF_SCOPE)} written for this test):")
    declined = 0
    for s in OUT_OF_SCOPE:
        a = d.ask(s, q)["department"]
        hit = a["choice"] == ESCAPE
        declined += hit
        print(f"  {'OK ' if hit else 'MISS'} {a['choice']:15s} {a['confidence']:.3f}  {s[:46]}")
    print(f"  declined {declined}/{len(OUT_OF_SCOPE)} = {declined / len(OUT_OF_SCOPE):.0%}")

    print(f"\nin scope ({len(IN_SCOPE)}): must not fall through to {ESCAPE!r}")
    right = fell = 0
    for s, gold in IN_SCOPE:
        a = d.ask(s, q)["department"]
        right += a["choice"] == gold
        fell += a["choice"] == ESCAPE
        mark = "OK " if a["choice"] == gold else "    "
        print(f"  {mark} {a['choice']:15s} {a['confidence']:.3f}  (want {gold:10s}) {s[:36]}")
    print(f"  correct {right}/{len(IN_SCOPE)}, wrongly declined {fell}")

    # -- 4. does the order of the options matter? -----------------------
    rng = random.Random(0)
    agree = 0
    for s, _ in IN_SCOPE:
        base = d.ask(s, q)["department"]["choice"]
        shuffled = menu[:]
        rng.shuffle(shuffled)
        other = d.ask(s, {"department": {"type": "choice",
                                         "instructions": DEPARTMENT_Q,
                                         "criteria": shuffled}})["department"]["choice"]
        agree += base == other
    print(f"\nsame answer after shuffling the options: "
          f"{agree}/{len(IN_SCOPE)} = {agree / len(IN_SCOPE):.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
