# Neural Networks from First Principles

An interactive, visual-first tutorial that walks the reader from linear
regression to a working (tiny) language model — and back — all in one page.

**Live:** <https://aabdelfattah.github.io/neural-first-principle/>

## The main tutorial

[`index.html`](index.html) — *From a line to a language model*. Seven stages,
each adds exactly one idea to the one before it. Every number on screen.

1. **Line** — a model is a function with knobs (`ŷ = wx + b`).
2. **Bend** — a sigmoid; one neuron.
3. **Stack** — two bent lines = an MLP that can fit any shape.
4. **Embed** — words enter the same machine, via a lookup table.
5. **Attend** — Q · K dot products, softmax, mix the values, residual update.
6. **MLP** — the Stage-3 machine, vectorised. The reunion.
7. **Generate** — logits → softmax → sample → append → loop.

The fixed prompt is Grant Sanderson's example from 3Blue1Brown's attention
chapter: **`a fluffy blue creature`**. Greedy generation reproduces his exact
sentence — *roamed the forest .* (with "verdant" dropped for vocab compactness).

### Run locally

It's a single static HTML file with everything inline. No build step.

```bash
# Open directly
open index.html

# Or serve over HTTP so URL hash deep-links (#5 for Attend, #6 for MLP) work
python3 -m http.server 8000
# then visit http://localhost:8000/
```

### Test the math

Three independent checks, all green:

```bash
# 1. Browser assertions — open this and look for "19/19 passed"
open tests/math.test.html

# 2. Cross-language equivalence — proves the JS in index.html is the same
#    function as a Python reference, to 1e-14 across 784 intermediate values.
python3 tests/equivalence.py

# 3. Generate simulator — prints the rollout step by step. Greedy mode must
#    match the browser exactly; sample mode matches in distribution.
python3 tests/generate_rollout.py
```

See the `## Validation pipeline` section below for what each one proves.

### Retrain (optional)

Only needed if you want to change the corpus, vocab, or model size:

```bash
python3 tests/fit_transformer.py
# paste the printed JS constants into index.html
python3 tests/equivalence.py     # confirm JS still matches Python
open tests/math.test.html        # update canonical numbers if necessary
```

### Deploy to GitHub Pages

Static files only. Push to `main`, enable Pages on the repo, and that's it.
Embed in a WordPress Custom HTML block:

```html
<iframe
  src="https://aabdelfattah.github.io/neural-first-principle/"
  width="100%" height="1400"
  style="border:0; max-width:1100px; display:block; margin:0 auto;"
  loading="lazy"
  title="From a line to a language model">
</iframe>
```

## Legacy companion articles

Earlier prose-first deep dives, kept for the curious reader who wants the full
backprop derivation:

- [`lines-to-neurons.html`](lines-to-neurons.html) — *Neural Networks: From
  Lines to Neurons.* Linear regression → loss → gradient descent →
  backpropagation → a working neuron. Includes a live training playground.
- [`neurons-to-transformers.html`](neurons-to-transformers.html) — *From
  Neurons to Transformers, Part II.* Stacking neurons → matrices →
  embeddings → attention.

These were the original tutorials and are referenced from the new one. Source
code for the toy network in those articles: [`tests/nn_from_scratch.py`](tests/nn_from_scratch.py).

## Validation pipeline

Five pieces; the JS Generate button is provably running the real attention
math, not a hard-coded trick:

```
┌──────────────────────────┐  TRAINS the model offline (numpy + scipy.optimize).
│ tests/fit_transformer.py │  Self-asserts canonical behaviour. Prints the
└──────────┬───────────────┘  rounded weights as JS arrays.
           │ paste constants ↓
┌──────────────────────────┐  THE PAGE. Constants as plain JS arrays + the
│ index.html               │  JS pipeline that runs the forward pass live
└──────────┬───────────────┘  every time you drag a slider or click Generate.
           │
           ├──▶ tests/math.test.html       independent JS reimplementation;
           │                               asserts canonical numbers. 19/19.
           │
           ├──▶ tests/equivalence.py       parses index.html, runs JS in node
           │                               + a Python reimplementation;
           │                               compares 784 intermediate values.
           │                               Max Δ = 6.4e-14.
           │
           └──▶ tests/generate_rollout.py  simulates the Generate button step
                                           by step in Python. Greedy must
                                           match the browser exactly.
```

What each one proves, in one sentence:

1. **`tests/fit_transformer.py`** — the weights come from a standard ML
   training loop (cross-entropy + L-BFGS-B + L2 on `E`), not magic numbers.
2. **`tests/math.test.html`** — the page's math is internally consistent (an
   independent JS reimplementation agrees on the canonical numbers).
3. **`tests/equivalence.py`** — the JS shipped in `index.html` *is the same
   function* as a Python reimplementation, to floating-point noise. So
   whatever the page does, it's not a trick.
4. **`tests/generate_rollout.py`** — here is the rollout, step by step, in
   Python — what the browser must reproduce in greedy mode.

## Repo contents

| File | What it is |
|---|---|
| `index.html` | The main tutorial (single file, all inline) |
| `CLAUDE.md` | Full spec — also picked up by Claude Code as project context |
| `lines-to-neurons.html` · `neurons-to-transformers.html` · `loss.png` | Legacy prose-first articles + their support image |
| `tests/fit_transformer.py` | Offline weight fitter (numpy + scipy) |
| `tests/math.test.html` | Browser-based math assertions (19/19) |
| `tests/equivalence.py` | JS = Python equivalence test (784 values, 1e-14) |
| `tests/generate_rollout.py` | Step-by-step Python simulator of the Generate button |
| `tests/nn_from_scratch.py` | Support code for the legacy articles |

## Licence

MIT — see [`LICENSE`](LICENSE).

Part of *Neural Networks from First Principles* by
[Ahmed Abdelfattah](https://aabdelfattah.me).
