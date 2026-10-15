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

"""Support messages with urgency that is actually in the text.

The first attempt borrowed Yelp stars as a stand-in for urgency and scored
66% -- a one-star review is angry, not urgent, so the label was measuring the
wrong thing. The second mistake was deriving escalation from the department,
which let the model infer it from the routing and score 99.7% without reading
anything.

Both are fixed the same way: compose the message out of a situation and a
circumstance, and let urgency come from the circumstance. Crucially the two
axes are independent, so every department spans every level of urgency --
a billing question can wait a week and a shipping problem can be an
emergency. A model cannot shortcut from one to the other.

    python3 scripts/synth_urgency.py --n 4000 --out /tmp/urgency.jsonl
"""

import argparse
import itertools
import json
import random
import sys

URGENCY = ["can wait", "this week", "today", "right now"]

# (department, situation). The situation carries no timing of its own.
SITUATIONS = [
    ("billing", "I was charged {amount} for something I did not buy"),
    ("billing", "there is a {amount} fee on my statement I do not recognise"),
    ("billing", "the same payment of {amount} went through twice"),
    ("billing", "my subscription renewed at {amount} after I cancelled it"),
    ("billing", "I was quoted one exchange rate and charged another"),
    ("shipping", "my replacement card has not arrived"),
    ("shipping", "the tracking for my card has not moved"),
    ("shipping", "my card was delivered to the wrong address"),
    ("shipping", "I never received the card you say was sent"),
    ("returns", "the refund I was promised has not appeared"),
    ("returns", "I want to cancel a transfer I just made"),
    ("returns", "I returned the item three weeks ago and heard nothing"),
    ("returns", "I was told I would get {amount} back and nothing came"),
    ("account", "I cannot log in and the reset email never arrives"),
    ("account", "my account is locked after too many attempts"),
    ("account", "I need to change the phone number on my account"),
    ("account", "someone else's name is showing on my profile"),
    ("technical", "my transfer is stuck as pending"),
    ("technical", "the app crashes every time I open the payments tab"),
    ("technical", "my balance has not updated since the transfer"),
    ("technical", "the top up failed but the money left my account"),
]

# The circumstance is what sets urgency, independent of the situation.
CIRCUMSTANCE = {
    3: [  # right now
        "and I am stranded at the airport with no other way to pay",
        "and my card is declining at the hospital pharmacy right now",
        "and I have no money for food tonight",
        "and it is still happening as I type, more charges keep appearing",
        "and I am abroad with no cash and nothing else works",
        "and my rent payment bounces in an hour because of it",
        "and someone is clearly using my account as we speak",
    ],
    2: [  # today
        "and I need this sorted before the shops close today",
        "and I have a payment due tonight",
        "and I am travelling first thing tomorrow morning",
        "and my direct debit goes out this evening",
        "and I need to pay a contractor by the end of today",
    ],
    1: [  # this week
        "and I would like it resolved before the weekend",
        "and I need an answer in the next few days",
        "and my next statement closes on Friday",
        "and I am away from next week so I would like it done before then",
    ],
    0: [  # can wait
        "but there is no rush, whenever you get to it",
        "but nothing is blocked, I am just letting you know",
        "and I am only asking out of curiosity",
        "but it is not causing me any trouble at the moment",
        "and I can carry on as normal in the meantime",
        "but please do not treat this as urgent",
    ],
}

OPENERS = ["Hi, ", "Hello, ", "", "Hi there, ", "Good morning, ", ""]
CLOSERS = [".", ". Thanks.", ". Please help.", ". Can you look into it?",
           ". Let me know.", ". Thank you."]
AMOUNTS = ["£12.40", "£89", "£250", "€35", "$19.99", "£1,200", "€7.50", "£64"]

# A few messages state the timing without any of the phrases above, so the
# model cannot key on a fixed suffix.
BARE = [
    (3, "account", "My card has been blocked and I am at a petrol station unable to pay"),
    (3, "billing", "Someone has drained my account overnight and transfers are still going out"),
    (2, "technical", "The transfer has to clear before five o'clock or I lose the deposit"),
    (2, "shipping", "I fly on Thursday and I still have no card"),
    (1, "returns", "I would like the refund processed sometime this week if possible"),
    (1, "account", "Could you update my address before the next statement goes out"),
    (0, "billing", "Just a question about how the monthly fee is calculated"),
    (0, "shipping", "Wondering roughly how long a new card usually takes to arrive"),
    (0, "technical", "No problem at all, I am just curious why transfers show as pending"),
    (0, "returns", "Whenever convenient, could you confirm the return was received"),
]


def compose(rng):
    dept, situation = rng.choice(SITUATIONS)
    level = rng.choice([0, 1, 2, 3])
    body = situation.format(amount=rng.choice(AMOUNTS))
    text = (rng.choice(OPENERS) + body + " "
            + rng.choice(CIRCUMSTANCE[level]) + rng.choice(CLOSERS))
    return text[0].upper() + text[1:], dept, level


def generate(n, seed=0):
    rng = random.Random(seed)
    seen, out = set(), []
    for level, dept, text in BARE:
        out.append({"text": text + ".", "department": dept, "urgency": level})
        seen.add(text)
    guard = 0
    while len(out) < n and guard < n * 40:
        guard += 1
        text, dept, level = compose(rng)
        if text in seen:
            continue
        seen.add(text)
        out.append({"text": text, "department": dept, "urgency": level})
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    args = ap.parse_args()

    rows = generate(args.n, args.seed)
    print(f"{len(rows)} messages")

    cross = {(d, u): 0 for d in {s[0] for s in SITUATIONS}
             for u in range(len(URGENCY))}
    for r in rows:
        cross[(r["department"], r["urgency"])] += 1
    print(f"\n{'department':12s}" + "".join(f"{u:>12s}" for u in URGENCY))
    for d in sorted({s[0] for s in SITUATIONS}):
        print(f"{d:12s}" + "".join(f"{cross[(d, u)]:12d}"
                                  for u in range(len(URGENCY))))
    print("\nurgency is independent of department, so neither can be guessed "
          "from the other.")

    print("\nsamples:")
    for r in rows[:6]:
        print(f"  [{URGENCY[r['urgency']]:9s}] [{r['department']:9s}] {r['text'][:72]}")

    if args.out:
        with open(args.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
