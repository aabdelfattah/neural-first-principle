"""
Cross-language equivalence check.

The JS pipeline shipped in `lines-to-llms/index.html` must compute the SAME
mathematical function as the Python reference in `fit_transformer.py`.

How this test works
-------------------
1. Parse `index.html` to extract the trained constants (`DEFAULTS` → `E`,
   `W1`, `B1`, `W2`, `B2`) and the JS forward-pass code itself.
2. Re-implement the forward pass in Python using NumPy and run it on a
   fixed list of prompts × {MLP on, MLP bypassed}.
3. Drive `node` to evaluate the actual JS pipeline (sliced straight out of
   the page, no re-typing) on the same prompts.
4. Compare every intermediate value elementwise. Fail loudly if any pair
   diverges by more than the tolerance.

What this proves
----------------
- The JS in the page is not "hallucinated" or hand-fudged — every line of
  math computes against the same inputs and produces the same outputs as a
  hand-written Python model.
- Python and JavaScript both run on IEEE 754 doubles, so identical
  arithmetic in the same order agrees to ~1e-15. The 1e-9 bar leaves
  generous headroom for any reordering.

Run:   python3 lines-to-llms/tests/equivalence.py
Exit:  0 on success, 1 if anything diverges.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

INDEX = Path(__file__).resolve().parents[1] / "index.html"
TOL = 1e-9

# Prompts × {MLP on, MLP bypassed}. Includes the fixed prompt, every
# generation-rollout step, and a couple of alternative shapes.
# VOCAB: a(0) the(1) fluffy(2) blue(3) creature(4) forest(5)
#        roamed(6) walked(7) .(8)
PROMPTS = [
    [0, 2, 3, 4],            # a fluffy blue creature  (the fixed prompt)
    [0, 2, 3, 4, 6],         # ... roamed
    [0, 2, 3, 4, 7],         # ... walked
    [0, 2, 3, 4, 6, 1],      # ... roamed the
    [0, 2, 3, 4, 6, 1, 5],   # ... roamed the forest
    [0, 2, 3, 4, 6, 1, 5, 8],# ... roamed the forest .
    [4],                     # creature alone
    [4, 6, 1, 5],            # creature roamed the forest
]
USE_MLP_FLAGS = [True, False]


# ---------------------------------------------------------------------------
# Parse a `const NAME = [...]` JS array out of source text. Handles nested
# brackets and JS trailing commas; converts to a Python list via json.loads.
# ---------------------------------------------------------------------------
def parse_js_array(src: str, name: str):
    m = re.search(rf"const\s+{re.escape(name)}\s*=\s*", src)
    if not m:
        raise ValueError(f"{name} not found in source")
    i = m.end()
    if src[i] != "[":
        raise ValueError(f"{name} is not an array literal")
    depth, j = 0, i
    while j < len(src):
        c = src[j]
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                j += 1
                break
        j += 1
    body = src[i:j]
    body = re.sub(r",(\s*[\]\}])", r"\1", body)  # strip JS trailing commas
    return json.loads(body)


# ---------------------------------------------------------------------------
# Python reimplementation of the forward pass. Mirrors the JS in index.html
# line-for-line: identity W_Q/W_K/W_V, residual, 1-hidden-layer ReLU MLP
# (or bypassed), tied unembedding.
# ---------------------------------------------------------------------------
def softmax(v: np.ndarray) -> np.ndarray:
    z = v - v.max()
    e = np.exp(z)
    return e / e.sum()


def py_forward(ids, E, W1, B1, W2, B2, use_mlp=True):
    X = E[ids]                                       # (T, 3)
    q = X[-1]
    scores = X @ q
    alpha = softmax(scores)
    a = alpha @ X
    h1 = X[-1] + a
    z = W1 @ h1 + B1                                 # (6,)
    hid = np.maximum(0.0, z)
    delta = W2 @ hid + B2                            # (3,)
    h2 = h1 + delta if use_mlp else h1.copy()
    logits = E @ h2
    probs = softmax(logits)
    return {
        "scores": scores.tolist(), "alpha": alpha.tolist(),
        "h1": h1.tolist(), "z": z.tolist(), "hid": hid.tolist(),
        "delta": delta.tolist(), "h2": h2.tolist(),
        "logits": logits.tolist(), "probs": probs.tolist(),
    }


# ---------------------------------------------------------------------------
# Slice the JS pipeline straight out of the page and drive node to run it.
# Returns the JS outputs as a list of dicts shaped identically to py_forward.
# ---------------------------------------------------------------------------
def js_run(script_src: str, prompts, mlp_flags):
    consts = script_src[
        script_src.index("const VOCAB"): script_src.index("let activeTok")
    ]
    math_section = script_src[
        script_src.index("const PROMPT_IDS"):
        script_src.index("\n", script_src.index("const argmax")) + 1
    ]
    driver = f"""
const prompts = {json.dumps(prompts)};
const flags = {json.dumps(mlp_flags)};
const out = [];
for (const ids of prompts) for (const um of flags) {{
  const r = pipeline(ids, um);
  out.push({{scores:r.scores, alpha:r.alpha, h1:r.h1, z:r.z, hid:r.hid,
            delta:r.delta, h2:r.h2, logits:r.logits, probs:r.probs}});
}}
process.stdout.write(JSON.stringify(out));
"""
    proc = subprocess.run(
        ["node", "-e", consts + math_section + driver],
        capture_output=True, text=True, check=True,
    )
    return json.loads(proc.stdout)


def main():
    src = INDEX.read_text()
    script = src.split("<script>")[1].split("</script>")[0]

    E = np.array(parse_js_array(script, "DEFAULTS"))
    W1 = np.array(parse_js_array(script, "W1"))
    B1 = np.array(parse_js_array(script, "B1"))
    W2 = np.array(parse_js_array(script, "W2"))
    B2 = np.array(parse_js_array(script, "B2"))
    print(f"loaded constants from index.html: E{E.shape}, "
          f"W1{W1.shape}, B1{B1.shape}, W2{W2.shape}, B2{B2.shape}")

    py_out = [py_forward(ids, E, W1, B1, W2, B2, use_mlp=um)
              for ids in PROMPTS for um in USE_MLP_FLAGS]
    js_out = js_run(script, PROMPTS, USE_MLP_FLAGS)

    assert len(py_out) == len(js_out), "case count mismatch"

    fields = ("scores", "alpha", "h1", "z", "hid", "delta", "h2", "logits", "probs")
    fails = []
    max_diff = 0.0
    n_vals = 0
    for i, (p, j) in enumerate(zip(py_out, js_out)):
        for key in fields:
            for k, (pv, jv) in enumerate(zip(p[key], j[key])):
                n_vals += 1
                d = abs(pv - jv)
                if d > max_diff:
                    max_diff = d
                if d > TOL:
                    fails.append((i, key, k, pv, jv, d))

    cases_desc = [f"{ids} mlp={um}" for ids in PROMPTS for um in USE_MLP_FLAGS]
    print(f"checked {n_vals} values across {len(cases_desc)} cases "
          f"(prompts × MLP on/off)")
    print(f"max observed |Δ| = {max_diff:.2e}  (tolerance {TOL:.0e})")

    if fails:
        print(f"\nFAIL — {len(fails)} values diverged beyond tolerance:")
        for case_i, key, k, pv, jv, d in fails[:20]:
            print(f"  case {case_i} ({cases_desc[case_i]}) {key}[{k}]: "
                  f"py={pv!r} js={jv!r}  Δ={d:.2e}")
        sys.exit(1)

    print("\nOK — Python and JS pipelines are numerically identical.")


if __name__ == "__main__":
    main()
