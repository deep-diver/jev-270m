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

"""A `/v1/systemone` front end over vLLM's `/pooling`.

`vllm serve --runner pooling` answers with a raw score vector: correct, but it
leaves the caller to render prompts, match options back to names and take a
softmax. This puts that in the server so a client sends the Jev request shape
and gets typed answers back.

Nothing here runs a model. Rendering and the option mapping live on this side,
the forward pass stays in vLLM, and every question in a request is sent as one
batch so the engine schedules them together.

    uvicorn scripts.systemone_server:app --host 0.0.0.0 --port 6006
"""

import math
import os
import time
import uuid

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

VLLM = os.environ.get("VLLM_URL", "http://127.0.0.1:8111")
MODEL = os.environ.get("JEV_MODEL_DIR", "/home/gemma_merged")
MAX_OPTIONS = 8

app = FastAPI(title="jev-270m", version="0.1.0")


class Question(BaseModel):
    type: str
    instructions: str = ""
    criteria: dict | list | None = None


class Request(BaseModel):
    state: str | dict
    questions: dict[str, Question]
    model: str | None = None
    threshold: float = Field(0.5, ge=0.0, le=1.0)


def render(state, instructions, options):
    lines = [f"State:\n{state}\n", f"Question: {instructions}", "Options:"]
    for i, o in enumerate(options):
        lines.append(f"{o} <{i}>")
    lines.append("Decision:")
    return "\n".join(lines)


def options_for(q: Question):
    """The option list a question type is scored over.

    `noul` is a fixed Yes/No pair; `choice` takes its own labels; `score` takes
    an ordered list and is reported as an expected value over the levels.
    """
    if q.type == "noul":
        return ["Yes", "No"]
    if q.type == "choice":
        if isinstance(q.criteria, dict):
            return list(q.criteria)
        if isinstance(q.criteria, list):
            return list(q.criteria)
        raise HTTPException(400, "choice needs criteria")
    if q.type == "score":
        if not isinstance(q.criteria, list):
            raise HTTPException(400, "score needs an ordered criteria list")
        return list(q.criteria)
    raise HTTPException(400, f"unknown question type {q.type!r}")


def softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [v / s for v in e]


@app.get("/health")
async def health():
    async with httpx.AsyncClient(timeout=10) as c:
        try:
            r = await c.get(f"{VLLM}/health")
            return {"status": "ok" if r.status_code == 200 else "degraded",
                    "vllm": VLLM, "model": MODEL}
        except Exception as e:
            return {"status": "down", "vllm": VLLM, "error": str(e)}


@app.post("/v1/systemone")
async def systemone(req: Request):
    state = req.state if isinstance(req.state, str) else \
        "\n".join(f"{k}: {v}" for k, v in req.state.items())

    names, prompts, option_lists = [], [], []
    for name, q in req.questions.items():
        options = options_for(q)
        if not 2 <= len(options) <= MAX_OPTIONS:
            raise HTTPException(400,
                                f"{name}: need 2..{MAX_OPTIONS} options, got {len(options)}")
        names.append(name)
        option_lists.append(options)
        prompts.append(render(state, q.instructions, options))

    t0 = time.time()
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(f"{VLLM}/pooling",
                         json={"model": req.model or MODEL, "input": prompts})
    if r.status_code != 200:
        raise HTTPException(502, f"vllm: {r.text[:200]}")
    rows = r.json()["data"]

    answers = {}
    for name, options, row in zip(names, option_lists, rows):
        raw = [v[0] if isinstance(v, list) else v for v in row["data"]]
        probs = softmax(raw[:len(options)])
        q = req.questions[name]
        table = {o: round(p, 4) for o, p in zip(options, probs)}
        if q.type == "noul":
            answers[name] = {"type": "noul", "noul": round(probs[0], 4)}
        elif q.type == "choice":
            top = max(range(len(options)), key=lambda i: probs[i])
            answers[name] = {"type": "choice", "choice": options[top],
                             "confidence": round(probs[top], 4),
                             "probabilities": table}
        else:
            value = sum(i * p for i, p in enumerate(probs))
            answers[name] = {"type": "score", "score": round(value, 4),
                             "legend": {str(i): o for i, o in enumerate(options)},
                             "probabilities": {str(i): round(p, 4)
                                               for i, p in enumerate(probs)}}

    return {"id": f"decision-{uuid.uuid4().hex[:16]}",
            "object": "structured_decision",
            "created": int(time.time()),
            "model": req.model or MODEL,
            "answers": answers,
            "usage": {"questions": len(names),
                      "latency_ms": round((time.time() - t0) * 1000)}}
