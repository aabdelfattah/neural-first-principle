# Neural Networks from First Principles

An interactive, visual-first tutorial from linear regression to a working tiny
language model — all in one page, every number on screen.

**Live:** <https://aabdelfattah.github.io/neural-first-principle/>

## The tutorial

[`index.html`](index.html) — *From a line to a language model.* Seven stages,
each adding exactly one idea:

1. **Line** — a model is a function with knobs (`ŷ = wx + b`).
2. **Bend** — a sigmoid; one neuron.
3. **Stack** — two bent lines = an MLP that fits any shape.
4. **Embed** — words enter the same machine via a lookup table.
5. **Attend** — `Q·K` → softmax → mix the values → residual.
6. **MLP** — the Stage-3 machine, vectorised.
7. **Generate** — logits → softmax → sample → append → loop.

Fixed prompt: **`a fluffy blue creature`** (3Blue1Brown's attention example).
Greedy generation reproduces *roamed the forest .*

It's one static HTML file, everything inline, no build step.

```bash
open index.html                 # or: python3 -m http.server 8000
```

Serving over HTTP makes the URL hash deep-links work (`#5` Attend, `#6` MLP).

## Tests

The Generate button runs the real attention math, not a hard-coded trick —
proven three ways:

```bash
open tests/math.test.html        # browser assertions on canonical numbers (19/19)
python3 tests/equivalence.py     # JS in index.html == Python reference (784 values, max Δ 6.4e-14)
python3 tests/generate_rollout.py # step-by-step rollout the browser must match in greedy mode
```

Weights are fit offline by [`tests/fit_transformer.py`](tests/fit_transformer.py)
(cross-entropy + L-BFGS-B) and pasted into `index.html` as constants — no
in-browser training. Re-run it only to change the corpus, vocab, or model size,
then re-run `equivalence.py` to confirm the JS still matches.

## Deploy

Static files; deploy is a push to the `gh-pages` branch. Embed in WordPress
with a Custom HTML block:

```html
<iframe src="https://aabdelfattah.github.io/neural-first-principle/"
  width="100%" height="1400" loading="lazy"
  style="border:0; max-width:1100px; display:block; margin:0 auto;"
  title="From a line to a language model"></iframe>
```

## Repo contents

| File | What it is |
|---|---|
| `index.html` | The tutorial (single file, all inline) |
| `CLAUDE.md` | Full spec — also read by Claude Code as project context |
| `tests/fit_transformer.py` | Offline weight fitter (numpy + scipy) |
| `tests/math.test.html` | Browser math assertions (19/19) |
| `tests/equivalence.py` | JS == Python equivalence test (784 values) |
| `tests/generate_rollout.py` | Python simulator of the Generate button |
| `lines-to-neurons.html` · `neurons-to-transformers.html` | Legacy prose-first deep dives (full backprop derivation) |
| `tests/nn_from_scratch.py` · `loss.png` | Support code + image for the legacy articles |

## Licence

MIT — see [`LICENSE`](LICENSE). By
[Ahmed Abdelfattah](https://aabdelfattah.me).
