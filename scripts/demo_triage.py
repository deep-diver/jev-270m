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

"""A wide look at what the triage model does, including where it fails.

Grouped so each block asks one question of the model rather than showing the
same easy case ten times: can it route, can it decline, does urgency move
independently of the department, does tone fool it, and what happens outside
anything it was trained on.

    python3 scripts/demo_triage.py --model <dir> --ckpt triage.npz
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DEPARTMENTS = ["billing", "shipping", "returns", "account", "technical",
               "something else"]
QUESTIONS = {
    "department": {"type": "choice",
                   "instructions": "Which team should handle this message?",
                   "criteria": DEPARTMENTS},
    "escalate": {"type": "noul",
                 "instructions": "Does this need urgent human attention right now?"},
    "urgency": {"type": "score",
                "instructions": "How soon does this need an answer?",
                "criteria": ["can wait", "this week", "today", "right now"]},
    "frustration": {"type": "score",
                    "instructions": "How frustrated does the sender sound?",
                    "criteria": ["calm", "annoyed", "frustrated", "furious"]},
}

BLOCKS = [
    ("routing: one clear case per team", [
        "I was charged twice for the same transaction last night.",
        "My replacement card still has not arrived after three weeks.",
        "The refund you promised me two weeks ago has not appeared.",
        "I forgot my passcode and the reset email never arrives.",
        "My transfer has been stuck on pending since Tuesday.",
    ]),
    ("declining: nothing here is support's job", [
        "What time does the sun set in Lisbon tomorrow?",
        "Can you recommend a good sourdough recipe?",
        "Please write me a poem about the sea.",
        "Do you know if it will rain this weekend?",
    ]),
    ("same team, different urgency", [
        "There is a fee on my statement I do not recognise, but no rush at all.",
        "There is a fee on my statement I do not recognise and my rent bounces in an hour.",
        "My card has not arrived yet, I am just checking roughly how long it takes.",
        "My card has not arrived and I fly tomorrow morning with no other way to pay.",
    ]),
    ("tone against substance", [
        "I would be very grateful if someone could look at this: my account is "
        "drained and transfers are still leaving right now.",
        "THIS IS ABSOLUTELY UNACCEPTABLE. I want to know how the monthly fee is "
        "calculated. No hurry.",
    ]),
    ("short, vague, or mixed", [
        "Help",
        "money gone",
        "My card was declined and also I want to change my address and get a refund.",
        "Everything is fine, just saying thanks for the quick service last week.",
    ]),
    ("outside anything it saw in training", [
        "The payment page returns a 500 error and customers cannot check out.",
        "제 카드가 결제되지 않습니다.",
        "Our Kubernetes cluster is dropping pods during the nightly deploy.",
    ]),
]


def bar(p, width=18):
    n = int(round(p * width))
    return "#" * n + "." * (width - n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-270m-it")
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    from jev_infer import Decider
    d = Decider(args.model, args.ckpt)
    print(f"{d.device} · dev {d.dev_acc:.3f}\n")

    for title, cases in BLOCKS:
        print("=" * 72)
        print(title.upper())
        print("=" * 72)
        for state in cases:
            a = d.ask(state, QUESTIONS)
            dep = a["department"]
            urg, fru = a["urgency"], a["frustration"]
            print(f"\n  {state}")
            print(f"    department   {dep['choice']:15s} {dep['confidence']:.3f}  "
                  f"{bar(dep['confidence'])}")
            print(f"    escalate     {a['escalate']['noul']:.3f}"
                  f"{'':11s}{bar(a['escalate']['noul'])}")
            print(f"    urgency      {urg['legend'][str(int(round(urg['score'])))]:15s}"
                  f" {urg['score']:.2f}/3")
            print(f"    frustration  {fru['legend'][str(int(round(fru['score'])))]:15s}"
                  f" {fru['score']:.2f}/3")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
