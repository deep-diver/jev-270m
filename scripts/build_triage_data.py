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

"""Training data for one scenario: triaging an inbound support message.

The general-purpose model routed a payment outage to "returns" with 0.95
confidence. Two things caused that, and both are data problems. Nothing in its
training looked like an incident report, and every question it had ever seen
had a correct answer among the options, so it had never learned to decline.

So this narrows to a single scenario and gives every question a way out:

  department   choice  billing / shipping / returns / account / technical
                       + "something else"
  escalate     noul    does a person need to look at this now
  frustration  score   calm -> furious

A fifth to a quarter of the choice examples are *unanswerable on purpose* --
the message genuinely belongs to none of the listed departments, and the gold
answer is the escape option. clinc_oos supplies real out-of-scope utterances
for this rather than synthetic ones, which is what makes the abstention
trainable rather than a guess.

Urgency and escalation come from synth_urgency.py, where the circumstance that
sets them is written independently of the department. Borrowing review stars
scored 66%, because a one-star review is angry rather than urgent; deriving
escalation from the department scored 99.7% without reading the message at
all. Both numbers were measuring the label, not the model.

    python3 scripts/build_triage_data.py --out data/triage
"""

import argparse
import json
import os
import random
import re
import sys

ESCAPE = "something else"

# Each department is defined by the intents that belong to it, so the labels
# come from real messages rather than from a template.
BANKING_DEPARTMENTS = {
    "billing": ["card_payment_fee_charged", "declined_card_payment",
                "extra_charge_on_statement", "transaction_charged_twice",
                "card_payment_wrong_exchange_rate", "cash_withdrawal_charge",
                "declined_transfer", "transfer_fee_charged",
                "wrong_amount_of_cash_received", "direct_debit_payment_not_recognised"],
    "shipping": ["card_arrival", "card_delivery_estimate", "card_not_working",
                 "lost_or_stolen_card", "order_physical_card",
                 "card_acceptance", "getting_spare_card"],
    "returns": ["refund_not_showing_up", "request_refund",
                "reverted_card_payment?", "cancel_transfer",
                "Refund_not_showing_up"],
    "account": ["edit_personal_details", "change_pin", "passcode_forgotten",
                "unable_to_verify_identity", "verify_my_identity",
                "why_verify_identity", "terminate_account", "age_limit"],
    "technical": ["pending_card_payment", "pending_top_up", "pending_transfer",
                  "balance_not_updated_after_bank_transfer",
                  "top_up_failed", "transfer_not_received_by_recipient",
                  "beneficiary_not_allowed"],
}

URGENCY = ["can wait", "this week", "today", "right now"]
FRUSTRATION = ["calm", "annoyed", "frustrated", "furious"]

DEPARTMENT_Q = "Which team should handle this message?"
ESCALATE_Q = "Does this need urgent human attention right now?"
FRUSTRATION_Q = "How frustrated does the sender sound?"
URGENCY_Q = "How soon does this need an answer?"


def clip(t, n=500):
    return " ".join(str(t).split())[:n]


def load_sources(seed, n_doc):
    from datasets import load_dataset
    out = {}

    d = load_dataset("legacy-datasets/banking77", split="train").shuffle(seed=seed)
    names = d.features["label"].names
    by_dept, other = {k: [] for k in BANKING_DEPARTMENTS}, []
    owned = {i for v in BANKING_DEPARTMENTS.values() for i in v}
    for r in d.select(range(min(n_doc * 3, len(d)))):
        intent = names[r["label"]]
        placed = False
        for dept, intents in BANKING_DEPARTMENTS.items():
            if intent in intents:
                by_dept[dept].append(clip(r["text"]))
                placed = True
                break
        if not placed and intent not in owned:
            other.append(clip(r["text"]))
    out["by_dept"] = by_dept
    out["in_domain_other"] = other

    # Real out-of-scope utterances: someone writing to support about something
    # the desk does not cover. This is what teaches the escape option.
    d = load_dataset("clinc/clinc_oos", "small", split="train").shuffle(seed=seed)
    intents = d.features["intent"].names
    oos_id = intents.index("oos") if "oos" in intents else None
    out["oos"] = [clip(r["text"]) for r in d
                  if oos_id is not None and r["intent"] == oos_id][:n_doc]
    out["off_topic"] = [clip(r["text"]) for r in d.select(range(n_doc))
                        if r["intent"] != oos_id][:n_doc]

    d = load_dataset("dair-ai/emotion", split="train").shuffle(seed=seed)
    emo = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    out["emotion"] = [(clip(r["text"]), emo[r["label"]])
                      for r in d.select(range(n_doc))]

    from synth_urgency import generate as synth
    out["urgency"] = synth(n_doc, seed)
    return out


def department_question(rng, departments):
    """A fixed menu plus the escape option, shuffled so position carries nothing."""
    menu = list(departments)
    rng.shuffle(menu)
    return menu + [ESCAPE]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "triage"))
    ap.add_argument("--n-doc", type=int, default=4000)
    ap.add_argument("--unanswerable", type=float, default=0.22,
                    help="share of choice questions whose answer is the escape option")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)

    src = load_sources(args.seed, args.n_doc)
    for dept, rows in src["by_dept"].items():
        print(f"  {dept:10s} {len(rows):5d}")
    print(f"  {'oos':10s} {len(src['oos']):5d}   "
          f"{'off-topic':10s} {len(src['off_topic']):5d}   "
          f"{'other intent':12s} {len(src['in_domain_other']):5d}")
    print(f"  {'synthetic':10s} {len(src['urgency']):5d}  (urgency + department)")

    recs = []

    def add(kind, state, instructions, options, gold, src_name):
        recs.append({"src": src_name, "type": kind, "state": state,
                     "instructions": instructions, "options": options,
                     "gold": gold})

    # --- department: answerable -------------------------------------------
    for dept, rows in src["by_dept"].items():
        for text in rows:
            options = department_question(rng, src["by_dept"])
            add("choice", text, DEPARTMENT_Q, options, options.index(dept),
                f"dept:{dept}")

    # --- department: the answer is the escape option ----------------------
    pool = src["oos"] + src["off_topic"] + src["in_domain_other"]
    rng.shuffle(pool)
    n_unans = int(len(src["by_dept"]) and
                  args.unanswerable / (1 - args.unanswerable)
                  * sum(len(v) for v in src["by_dept"].values()))
    for text in pool[:n_unans]:
        options = department_question(rng, src["by_dept"])
        add("choice", text, DEPARTMENT_Q, options, options.index(ESCAPE),
            "dept:escape")

    # --- a smaller menu, so the model cannot memorise one option list -----
    for dept, rows in src["by_dept"].items():
        for text in rows[: len(rows) // 3]:
            others = [d for d in src["by_dept"] if d != dept]
            menu = [dept] + rng.sample(others, 2)
            rng.shuffle(menu)
            menu = menu + [ESCAPE]
            add("choice", text, DEPARTMENT_Q, menu, menu.index(dept),
                f"dept3:{dept}")

    # --- escalate, from the circumstance rather than the department --------
    # The two top urgency levels need a person now; the two bottom ones do not.
    # Because urgency is independent of department, the answer cannot be read
    # off the routing.
    for r in src["urgency"]:
        gold = 0 if r["urgency"] >= 2 else 1
        add("noul", r["text"], ESCALATE_Q, ["Yes", "No"], gold,
            f"escalate:{'yes' if gold == 0 else 'no'}")
    for text in src["off_topic"][: len(src["urgency"]) // 4]:
        add("noul", text, ESCALATE_Q, ["Yes", "No"], 1, "escalate:no")

    # --- frustration, from labelled emotion --------------------------------
    emo_to_level = {"calm": 0, "joy": 0, "love": 0, "surprise": 1,
                    "sadness": 1, "fear": 2, "anger": 3}
    for text, emo in src["emotion"]:
        level = emo_to_level.get(emo)
        if level is None:
            continue
        add("score", text, FRUSTRATION_Q, list(FRUSTRATION), level, "frustration")

    # --- urgency, straight from the circumstance that was written in ------
    for r in src["urgency"]:
        add("score", r["text"], URGENCY_Q, list(URGENCY), r["urgency"], "urgency")

    # The same synthetic messages also carry a department, which keeps the
    # routing examples from being only banking77 phrasing.
    for r in src["urgency"]:
        options = department_question(rng, src["by_dept"])
        add("choice", r["text"], DEPARTMENT_Q, options,
            options.index(r["department"]), f"dept_synth:{r['department']}")

    rng.shuffle(recs)
    n_dev = max(200, len(recs) // 20)
    for name, rows in (("dev", recs[:n_dev]), ("train", recs[n_dev:])):
        p = os.path.join(args.out, f"{name}.jsonl")
        with open(p, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name:6s} {len(rows):7,d} -> {p}")

    by_type, escapes, dept_total = {}, 0, 0
    for r in recs:
        by_type[r["type"]] = by_type.get(r["type"], 0) + 1
        if r["src"].startswith("dept"):
            dept_total += 1
            if r["options"][r["gold"]] == ESCAPE:
                escapes += 1
    print("\nby type:", by_type)
    print(f"department questions whose answer is {ESCAPE!r}: "
          f"{escapes}/{dept_total} = {escapes / max(dept_total, 1):.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
