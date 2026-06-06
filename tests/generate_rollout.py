"""
Validate that the Generate button is running real attention on a small example.

How this proves it
------------------
1. equivalence.py already showed the JS pipeline in index.html is numerically
   identical to a hand-written Python reference, to 1e-14 across 808 values
   covering every prompt the rollout actually uses.

2. This script does the SAME rollout the JS Generate button performs — load
   the same constants, run the same forward pass, sample the same way — and
   prints every intermediate value: the prompt fed in, the attention pattern
   computed, the top tokens by probability, and the sampled choice.

3. Greedy mode is deterministic: argmax at each step. The tokens the browser
   produces in greedy mode must match this script's output exactly (no RNG
   involved). Sample mode has a different RNG than the browser's
   Math.random(), but the underlying probability distribution at each step is
   provably identical (per equivalence.py). Sampling enough times in the
   browser converges to the same percentages this script prints.

So the chain is:
    fit_transformer.py   ->  produces weights
    rounded constants    ->  pasted into index.html
    JS pipeline          ==  Python reference   (equivalence.py)
    Generate button      ==  this script        (each click is one step here)

There is no caching, no precomputed lookup, no fakery. Each click recomputes
the full attention -> MLP -> unembed pipeline on the growing prompt from
scratch.

Run:
    python3 lines-to-llms/tests/generate_rollout.py
"""

import random
import sys
from pathlib import Path

import numpy as np

# Reuse the parse+forward helpers proven by equivalence.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from equivalence import INDEX, parse_js_array, py_forward, softmax

VOCAB = ['a', 'the', 'fluffy', 'blue', 'creature', 'forest',
         'roamed', 'walked', '.']
PROMPT_IDS = [0, 2, 3, 4]   # a fluffy blue creature
MAX_GEN = 10


def load_constants():
    src = INDEX.read_text()
    script = src.split("<script>")[1].split("</script>")[0]
    return (
        np.array(parse_js_array(script, "DEFAULTS")),
        np.array(parse_js_array(script, "W1")),
        np.array(parse_js_array(script, "B1")),
        np.array(parse_js_array(script, "W2")),
        np.array(parse_js_array(script, "B2")),
    )


def words(ids):
    return " ".join(VOCAB[i] for i in ids)


def rollout(*, mode, T=1.0, seed=None):
    """One full rollout. mode='greedy' or 'sample'. Mirrors JS pickNext exactly."""
    rng = random.Random(seed)
    E, W1, B1, W2, B2 = load_constants()

    label = "GREEDY  (T=0 — deterministic argmax)" if mode == "greedy" \
            else f"SAMPLE  T={T}  seed={seed}"
    print("=" * 72)
    print(f"{label}")
    print(f"start prompt: {words(PROMPT_IDS)}")
    print("=" * 72)

    ids = list(PROMPT_IDS)
    step = 1
    while len(ids) < MAX_GEN:
        out = py_forward(ids, E, W1, B1, W2, B2, use_mlp=True)
        scores = np.array(out["scores"])
        alpha = np.array(out["alpha"])
        logits = np.array(out["logits"])

        if mode == "greedy" or T <= 0:
            probs_T = softmax(logits)
            nxt = int(np.argmax(logits))
            decision = "argmax"
        else:
            probs_T = softmax(logits / T)
            u, cum = rng.random(), 0.0
            nxt = len(probs_T) - 1
            for i, p in enumerate(probs_T):
                cum += p
                if u < cum:
                    nxt = i
                    break
            decision = f"sample (u={u:.3f})"

        print(f"\n  STEP {step}  ·  input = '{words(ids)}'")
        print(f"           Q·K  = [{', '.join(f'{x:+.3f}' for x in scores)}]")
        print(f"           attn = [{', '.join(f'{x*100:4.1f}%' for x in alpha)}]"
              f"   over ({words(ids)})")
        top = np.argsort(-probs_T)[:3]
        print(f"           top  = " + "  ·  ".join(
            f"{VOCAB[v]:>5} {probs_T[v]*100:5.1f}%" for v in top))
        print(f"           ──→  '{VOCAB[nxt]}'  ({decision})")

        ids.append(nxt)
        if VOCAB[nxt] == ".":
            print(f"\n  end of sentence reached.")
            break
        step += 1

    print(f"\nfinal rollout: '{words(ids)}'\n")
    return ids


if __name__ == "__main__":
    # Greedy: deterministic; the browser, in greedy mode, must produce EXACTLY
    # the same tokens. No RNG involved — pure argmax of the same logits.
    rollout(mode="greedy")

    # Sample low-T: distribution almost peaked, mostly same as greedy.
    rollout(mode="sample", T=0.5, seed=42)

    # Sample high-T: distribution flattened; runners-up (floor / rug) win
    # a meaningful share of clicks. The browser will see the same percentages.
    rollout(mode="sample", T=2.0, seed=42)
