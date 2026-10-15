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
Fold the adapter into the base weights and write a plain checkpoint.

The training run keeps the backbone frozen and carries the adaptation in
`lora.npz` as stacked arrays, which nothing but our own JAX code can read.
Anything that loads a checkpoint the ordinary way needs it to look like any
other Gemma 3, so the adapter is folded in:

    W_eff = W + (alpha / rank) * B @ A

The pointer head cannot be folded anywhere: it is a second readout, not a
correction to an existing matrix. It is written beside the weights as
`pointer_head.safetensors` for a serving layer to pick up.

    python3 scripts/merge_lora.py --base <dir> --ckpt lora.npz --out <dir>
"""

import argparse
import json
import os
import shutil
import sys

import jax.numpy as jnp
import numpy as np
import safetensors.flax as sf

NUM_LAYERS = 18
PROJ = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "o_proj"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="directory of the base model")
    ap.add_argument("--ckpt", required=True, help="lora.npz from training")
    ap.add_argument("--out", required=True, help="directory to write")
    args = ap.parse_args()

    ck = np.load(args.ckpt)
    rank, dim = int(ck["rank"]), int(ck["dim"])
    scale = float(ck["alpha"]) / rank
    print(f"adapter: rank {rank}, alpha {float(ck['alpha'])}, scale {scale:.3f}, "
          f"head dim {dim}, dev acc {float(ck['dev_acc']):.3f}")

    # safetensors.flax, not numpy: the base ships in bfloat16, which numpy has
    # no dtype for. Each matrix is merged in float32 and written back as it came.
    src = os.path.join(args.base, "model.safetensors")
    weights = sf.load_file(src)
    print(f"base: {len(weights)} tensors from {src}, "
          f"dtype {weights['model.embed_tokens.weight'].dtype}")

    merged, biggest = 0, 0.0
    for short, name in PROJ.items():
        A, B = ck[f"lora.{short}_A"], ck[f"lora.{short}_B"]   # (L, r, in), (L, out, r)
        assert A.shape[0] == B.shape[0] == NUM_LAYERS, (A.shape, B.shape)
        for i in range(NUM_LAYERS):
            key = f"model.layers.{i}.self_attn.{name}.weight"
            orig = weights[key]
            W = orig.astype(jnp.float32)
            delta = jnp.asarray(
                scale * (B[i].astype(np.float32) @ A[i].astype(np.float32)))
            assert delta.shape == W.shape, (key, delta.shape, W.shape)
            biggest = max(biggest, float(jnp.abs(delta).max()
                                         / (jnp.abs(W).max() + 1e-9)))
            weights[key] = (W + delta).astype(orig.dtype)
            merged += 1
    print(f"merged {merged} matrices; largest delta is {biggest:.1%} of its matrix")

    os.makedirs(args.out, exist_ok=True)
    for f in os.listdir(args.base):
        if f != "model.safetensors":
            shutil.copy2(os.path.join(args.base, f), os.path.join(args.out, f))
    sf.save_file(weights, os.path.join(args.out, "model.safetensors"))

    # Kept out of the top level: loaders that glob every *.safetensors beside
    # the config would try to push these into the backbone.
    extras = os.path.join(args.out, "extras")
    os.makedirs(extras, exist_ok=True)
    head = {k[5:]: jnp.asarray(np.asarray(ck[k], dtype=np.float32))
            for k in ck.files if k.startswith("head.")}
    sf.save_file(head, os.path.join(extras, "pointer_head.safetensors"))
    json.dump({"dim": dim, "hidden": int(head["Wq"].shape[1]),
               "keys": sorted(head)},
              open(os.path.join(extras, "pointer_head.json"), "w"), indent=1)

    print(f"-> {args.out}")
    print(f"   model.safetensors        {len(weights)} tensors")
    print(f"   extras/pointer_head.safetensors {sorted(head)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
