# From Lines to LLMs — Interactive Tutorial Requirements

A first-principles, visual-first, click-to-generate tutorial that walks the
reader from linear regression to a working (tiny) language model — and back —
all in one interactive page. Lives in this repo and is served via GitHub Pages;
the WordPress blog at `aabdelfattah.me/projects` will link or embed it.

---

## 1. Vision

> "I want to understand exactly how an LLM picks its next word, from first
> principles, without taking anyone's word for it."

A reader who completes the tutorial should be able to:

1. Explain why a neural network is **just linear regression with a bend**, and
   why stacking those bends lets you fit any curve.
2. Click **Generate** in the browser and watch a real (tiny) model sample a
   word from the fixed prompt `a fluffy blue creature`, with every internal step visible.
3. State the spine sentence themselves (§2).

**Visualisation is the primary medium**, not prose. Text appears as captions,
labels, and short helpers.

### 1.1 Audience

The target reader is **a curious individual** — not an ML student, not a
professional software engineer. Smart, motivated, willing to spend an afternoon
with a calculator, but not paid to know this material.

**Assumed background:**

- High-school algebra (`y = mx + b`, comfortable with variables).
- Heard of neural networks but cannot define "logit" or "softmax" unprompted.
- Zero programming required to *use* the page. Code blocks are revealable for
  the curious; reading them is optional.

**Reference for tone and depth:** Andrej Karpathy's *Neural Networks: Zero to
Hero* series. Accessible to a layperson, never hand-waves. By the end the
reader has computed real numbers and understands the actual mechanism, not a
metaphor for it.

**Implications:**

- **A1 — No unexplained jargon.** Every term ("embedding", "softmax",
  "attention", "logit", "residual") gets a one-sentence definition inline or
  in a hover/fold the first time it appears.
- **A2 — Difficulty curves gently.** Each stage adds exactly **one** new idea.
- **A3 — Depth without intimidation.** Show every number, every matmul, every
  softmax — but always at calculator scale, inside a visual that frames *why*.
- **A4 — Curiosity-driven.** A reader who only clicks **Generate** must still
  come away with the right mental model. A reader who reads every word and
  tweaks every slider gets the full Karpathy-grade depth.
- **A5 — No prerequisite reading.** The page assumes the reader landed cold.

---

## 2. Pedagogical scope — one continuous ladder

The reader walks **one ladder, one rung at a time.** Each rung is a real
interactive stage with its own visualisation and live controls — not an intro
paragraph linking out.

### 2.1 The seven stages (3 foundation : 4 transformer)

| # | Stage | Core idea added | New first-principles content |
|---|-------|------------------|------------------------------|
| 1 | **Line** | A model is a function with knobs. | `ŷ = wx + b`. Slide w/b. Watch the line move and the loss change. |
| 2 | **Bend** | A non-linearity → can curve, not just slope. | `ŷ = σ(wx + b)`. One neuron — the sigmoid bends the line. |
| 3 | **Stack** | Stack neurons → can fit any shape. **This is an MLP.** | Two neurons, subtracted. Fits a non-monotonic target a single line cannot. |
|   | — *now apply to text* — | | |
| 4 | **Embed** | Words plug into the same machine, via lookup. | A token-id picks a row of `E`. Each row is a 3-number vector. |
| 5 | **Attend** | The mixing weights are computed from the data. | Each word's query dotted against every key → softmax → weights → weighted sum of values, plus residual. **This is the new idea transformers add.** |
| 6 | **MLP** | The Stage-3 stack, applied to a vector. | `h₂ = h₁ + W₂·ReLU(W₁·h₁ + b₁) + b₂`. **The reunion: same machine.** |
| 7 | **Generate** | Vectors become words. Pick one. Loop. | Tied unembed → logits → softmax(/T) → P, then sample & append. |

### 2.2 Balance (3 : 4) — not transformer-heavy

A previous draft was 3 : 5, which made the transformer look like a separate
bigger thing — the opposite of the continuity thesis. The current 3 : 4
balance is achieved by collapsing the old `Attend + Mix` into one Attend
stage (scoring and the weighted-sum-of-values are one idea) and the old
`Unembed + Sample` into one Generate climax (turning the vector into a word
and picking one is a single continuous motion).

### 2.3 The spine (non-negotiable thesis)

Stated explicitly on the page; the reader should be able to recite it:

> **Attention decides *which* words to mix. The MLP — the stack of bent lines
> from Stage 3 — decides *what the mix means*. Unembedding turns it back into
> words. Same loss-minimising machine as the line you started with, scaled up.**

### 2.4 Continuity constraints

- **C1.** Every concept the next stage builds on is visible on-screen (not
  "as we saw earlier"). A reader landing halfway down can still follow forward.
- **C2.** Math notation, colour coding, and variable names are **consistent
  across all stages**. `w`, `b`, `x`, `y` in stage 1 mean the same thing in
  stage 7. New symbols (`E`, `Q`, `K`, `V`, `α`, `h`) are introduced once and
  re-used identically.
- **C3.** The same colour grammar (V1–V5 in §7) is used in every stage.
- **C4.** Stage 6 (MLP) **explicitly calls back to Stage 3** ("you already
  built and trained this"). The reader must register that the transformer
  reuses, not replaces, the earlier rungs.

### 2.5 Legacy artefacts (deep-dive companions)

The original two HTMLs in this repo — `lines-to-neurons.html`,
`neurons-to-transformers.html` — remain published as prose-first deep dives
that contain the full backprop derivation. The new tutorial does **not**
re-derive backprop; it links out for the curious reader.

---

## 3. Functional requirements

### F1 — One live model, end to end

All four transformer stages read from a **single shared pipeline** over the
**fixed prompt `a fluffy blue creature`**. Editing an embedding on Stage 4 instantly
changes the attention weights, the MLP output, the logits, and the sampled
word. **Nothing is faked per-stage.**

### F2 — Every stage is interactive, none is "just text"

Each stage has at least one live control (slider / toggle / button) whose
effect is visible *in that stage's own visual*, immediately.

| Stage | Live control |
|---|---|
| Line | sliders for `w`, `b`; Train button; 🎲 random init |
| Bend | sliders; Train; 🎲 |
| Stack | sliders for 2 neurons; σ / ReLU toggle; Train; 🎲 |
| Embed | 12 × 3 embedding sliders |
| Attend | mirrored embedding sliders (live attention) |
| MLP | **bypass toggle** (MLP on vs off — readout flips between `roamed` and `creature` echoed back) |
| Generate | temperature slider; greedy / sample toggle; **Generate** button; reset |

### F3 — The successful journey

Pressing **Generate** must feel like an event: the pipeline visibly responds,
a word lands and appends to the sentence. A first-time reader who only ever
clicks Generate (and the temperature slider) still comes away with a correct
mental model. Repeated clicks extend `a fluffy blue creature` to a sentence stop (`.`),
then offer a reset.

### F4 — Reproducible, not random-looking

With temperature at its default the winning word is stable and sensible
(`roamed`). Numbers shown are the real numbers the pipeline computed — no
decorative values. Generation is the only stochastic part, and only in
`sample` mode.

### F5 — Step-through scrubber

A horizontal scrubber along the top shows the **seven stages**. Clicking a
chip switches the panel. Stage state persists in the URL fragment (e.g.
`#6` deep-links to MLP) so any stage is shareable. No auto-advance.

### F6 — Hand-verifiable math

Every internal number is shown, not just pictures. Each stage's matrices /
vectors are reachable in a "Show numbers" reveal or directly in the visual.
The `tests/math.test.html` page asserts every canonical number with the same
constants as `index.html` (see §5.6).

### F7 — Continuous narrative

All rungs live on the same page in scrollable order. The transition from
stage 3 → 4 is **explicitly signposted** ("now apply to text"), so the reader
registers that the same machinery is now being applied to language.

---

## 4. Non-functional requirements

### N1 — Visual-first, text-minimal

- N1.1 — Body text per stage ≤ **~60 words**. Anything longer hides behind a
  "Why?" fold.
- N1.2 — Diagrams and animations carry the explanation. If a stage can't be
  understood from its visual alone (plus one helper caption), the visual
  isn't good enough yet.
- N1.3 — No walls of equations. Equations live inside the visuals, pinned
  next to the elements they govern.

### N2 — Responsive

- N2.1 — Works on mobile (≥ 375 px wide). Sliders and the Generate button
  remain operable on a phone.
- N2.2 — On narrow viewports the scrubber stays scroll-snappable.

### N3 — Accessible

- N3.1 — All interactive elements are keyboard-reachable; visible focus rings.
- N3.2 — WCAG AA colour contrast on text against surfaces.
- N3.3 — `aria-label` / `aria-live` for the Generate button and changing
  probability values so screen readers announce updates.
- N3.4 — Respects `prefers-reduced-motion`: snaps to final state instead of
  long animations.

### N4 — Performance

- N4.1 — First contentful paint < 1 s on a fresh load.
- N4.2 — No build step.
- N4.3 — No JS dep over ~50 kB gzipped. Math in plain JS; visuals in inline
  SVG (D3 allowed but optional, currently unused).
- N4.4 — Works fully offline once loaded — no analytics, no API calls.

### N5 — Reproducibility & verification

- N5.1 — Every displayed number is reproducible by running the pure-math
  helpers in isolation. `tests/math.test.html` asserts the canonical numbers.
- N5.2 — `tests/equivalence.py` proves the JS pipeline in `index.html` is
  **numerically identical** to the Python reference in `fit_transformer.py`
  (tolerance 1e-9). See §5.5.

### N6 — Hostable on GitHub Pages

- N6.1 — Static output only. Deploy = push to `gh-pages`.
- N6.2 — Root URL `https://aabdelfattah.github.io/neural-first-principle/`
  serves the tutorial. Legacy HTMLs remain reachable at their existing paths.
- N6.3 — Embeds in a WordPress Custom HTML block via one `<iframe>` snippet.

### N7 — Open

- N7.1 — MIT license.
- N7.2 — Authorship + link back to `aabdelfattah.me` in the footer.
- N7.3 — `README.md` explains how to run locally, deploy, fork.

---

## 5. The model (concrete spec)

### 5.1 Dimensions and vocabulary

The example sentence is Grant Sanderson's literal example from 3Blue1Brown's
attention chapter — *"a fluffy blue creature roamed the [verdant] forest."* We
dropped "verdant" (and "fox", "ran") to keep the vocab to 9 tokens (cognitive
load). The prompt still puts `creature` as the asker, matching 3B1B's worked
example, and greedy rollout reproduces the sentence one token at a time.

| | |
|---|---|
| Vocabulary (9 tokens) | `a · the · fluffy · blue · creature · forest · roamed · walked · .` |
| Embedding dim | `d = 3` |
| Fixed prompt | `a fluffy blue creature` (token ids `[0, 2, 3, 4]`) |
| MLP hidden width | `H = 6` (ReLU) |

### 5.2 Forward pass (mirrored in JS and Python)

```
X      = E[ids]                                      # (T, 3)
q      = X[-1]                                        # identity W_Q
scores = X @ q                                        # identity W_K, no √d scaling
α      = softmax(scores)
a      = α @ X                                        # identity W_V
h1     = X[-1] + a                                    # residual
h2     = h1 + W2 · ReLU(W1 · h1 + b1) + b2            # MLP block
logits = E @ h2                                       # tied unembedding (U = Eᵀ)
P      = softmax(logits)
```

### 5.3 Weights — fit offline, baked into the page

The script `fit_transformer.py` fits `E`, `W1`, `b1`, `W2`, `b2` on a tiny
weighted toy corpus built around 3B1B's sentence (`a fluffy blue creature
roamed/walked …`) using `scipy.optimize.minimize` with L-BFGS-B and
cross-entropy loss. **L2 weight decay applied only to `E`** keeps embedding
magnitudes small (so attention scores don't blow up), letting the MLP grow to
sharpen the output. The rounded result (2 decimals) is pasted into
`index.html` as constants. **No training runs in the browser.**

### 5.4 Generation

`P_T = softmax(logits / T)`. `sample` = inverse-CDF draw; `greedy` = argmax.
Sampled token id appends to `genIds`; pipeline reruns on the longer prompt.
Length cap = 10 tokens (or stop when `.` is sampled). Reset (↺) restores
`a fluffy blue creature`.

### 5.5 Verified Python ↔ JS equivalence

The JS pipeline in `index.html` is **provably the same function** as the
Python reference in `fit_transformer.py`:

- `tests/equivalence.py` runs the Python pipeline on a fixed list of prompts,
  extracts the JS pipeline constants and code from `index.html`, runs the JS
  via `node` on the same prompts, and asserts every intermediate value
  (`scores`, `α`, `h₁`, `h₂`, `logits`, `P`) agrees to `< 1e-9` (both run on
  IEEE 754 doubles; tighter tolerances are within floating-point noise).
- Run: `python3 tests/equivalence.py`. It fails loud if anything drifts.

This is the guard rail against silent divergence between the reference and
the shipped page.

### 5.6 Canonical numbers (asserted in `tests/math.test.html`)

- `P(roamed | "a fluffy blue creature") ≈ 79.7%`
- `P(walked | "a fluffy blue creature") ≈ 20.0%`
- attention weights for `a / fluffy / blue / creature` ≈ `24.6% / 24.7% / 25.2% / 25.5%`
- attention output `h₁ ≈ [0.026, 0.161, 0.161]`
- greedy rollout from `a fluffy blue creature` is **`roamed → the → forest →
  .`** — reproducing 3B1B's sentence (sans "verdant", which we dropped).
- bypass-MLP argmax for `a fluffy blue creature` is **`creature` itself** —
  without the MLP the model just hands back the asker; the MLP is the
  knowledge of what creatures *do*.

---

## 6. Design references

We are explicitly modelling the look, feel, and interaction grammar of:

| Reference | What we take from it |
|---|---|
| poloclub/transformer-explainer (Polo Chau et al) | Click-to-generate flow, sticky pipeline scrubber, live softmax bars |
| bbycroft.net/llm (Brendan Bycroft) | Pipeline visualisation, every parameter on-screen |
| distill.pub | Visual primacy, hover-to-reveal annotations, careful typography |
| 3Blue1Brown — Neural Networks series | Pedagogical pacing: one tiny step at a time, every step drawn |
| Existing `lines-to-neurons.html` (Ahmed) | Colour palette continuity (extended, not replaced) |

---

## 7. Visual identity

Matches `aabdelfattah.me` (WordPress Twenty Twenty-Five). **Light, clean,
sans-serif.** A deliberate departure from the dark palette of the prior two
tutorials — the new page belongs visually to the blog.

Palette (from the live site's `theme.json`):

| Token | Hex | Role |
|---|---|---|
| `--bg` | `#FFFFFF` | Page background |
| `--text` | `#111111` | Body text and headings |
| `--text-dim` | `#686868` | Captions, labels |
| `--surface` | `#FBFAF3` | Cards, math blocks |
| `--accent` | `#503AA8` | Primary action |
| `--accent-yellow` | `#FFEE58` | Highlight (winning token) |
| `--accent-pink` | `#F6CFF4` | Soft secondary highlight |
| `--neg` | `#E65D75` | Negative-value sign colour (vector cells; see V1) |

Fonts: **Manrope 300** (body and headings), **Fira Code** (numbers, code,
math values).

Visual grammar (applies in every stage):

- **V1** — Rectangles for vectors. Cell fill scales with magnitude; sign
  by colour (purple positive, red negative).
- **V2** — Thin lines for matrix multiplications.
- **V3** — Active token / position highlighted in `--accent-yellow`.
- **V4** — Probability bars fill from 0 → 100% in `--accent`. Winning bar
  uses `--accent-yellow`.
- **V5** — Roles colour-coded: query = `--accent` (`#503AA8`),
  key = blue (`#1976d2`), value = green (`#2e7d32`). Same three colours
  everywhere these roles appear.

---

## 8. Tech stack & layout

Vanilla JS + inline SVG, in the spirit of the legacy HTMLs but a single
self-contained file. No build step. Flat repo so the tutorial lives at the
GitHub Pages root URL (`https://aabdelfattah.github.io/neural-first-principle/`).

```
neural-first-principle/
├── README.md                             run / deploy / verify
├── CLAUDE.md                             you are here (the spec; Claude Code reads this)
├── index.html                            THE PAGE — single-file tutorial (HTML + CSS + JS inline)
├── lines-to-neurons.html                 legacy companion article, untouched
├── neurons-to-transformers.html          legacy companion article, untouched
├── loss.png                              image used by legacy articles
└── tests/
    ├── fit_transformer.py                offline weight fitter (numpy + scipy)
    ├── math.test.html                    browser asserts on canonical numbers (§5.6)
    ├── equivalence.py                    Python ↔ JS numerical equivalence (§5.5)
    ├── generate_rollout.py               step-by-step Python simulator of the Generate button
    └── nn_from_scratch.py                support code for the legacy articles
```

---

## 9. Honesty list (intentional simplifications)

Each gets a plain-prose disclosure on the page or in a fold — no hand-waving:

- Identity `W_Q = W_K = W_V = I` (no learned projection matrices).
- Single attention head, single transformer block.
- **No positional encoding** (order enters only via the residual on the last
  token).
- **No √d scaling** of attention scores.
- **No layer norm.**
- Tied unembedding (`U = Eᵀ`).
- Weights are fit offline / baked in — not trained in-browser, not trained at
  scale.
- "This is GPT *scaled up*" — never "this is how GPT works" unqualified.

Plus two anti-anthropomorphism rules:

- Never "the model understands / knows".
- Never "neurons are like brain neurons".

---

## 10. Out of scope (v1)

- Multi-layer transformers.
- Multi-head attention.
- In-browser training.
- Real tokenisers (BPE, SentencePiece). Word-level vocab only.
- A backprop re-derivation as deep as `lines-to-neurons.html`. We link out.
- Internationalisation. English only.
- OpenGraph / Twitter preview cards.

---

## 11. Success criteria

The tutorial ships when:

1. A person who has not touched ML can click **Generate** and read out a
   sensible continuation of `a fluffy blue creature`. Greedy mode reproduces
   Grant Sanderson's sentence (`roamed → the → forest → .`, sans "verdant").
2. **MLP is a visible, interactive stage** that visually echoes Stage 3 and
   says so.
3. **Temperature visibly reshuffles** the share between `roamed` and `walked`
   between low and high `T`.
4. **One shared pipeline:** editing a Stage-4 embedding propagates live
   through stages 5–7.
5. `tests/math.test.html` is all green on a fresh load.
6. `tests/equivalence.py` is all green.
7. Loads, paints, and is interactive in < 1 s on a cold GitHub Pages request.
8. Linked from `aabdelfattah.me/projects` and embeds cleanly in an iframe.

---

## 12. Decisions log

- **Stack: vanilla JS + inline SVG, no build step.** Matches the existing
  series; zero toolchain barrier for the curious-learner audience (A4).
- **Layout: sticky scrubber** (`wireframes/B-sticky-scrubber.html`). The
  pipeline-as-spine is the single most useful framing device.
- **Fresh visual identity** — light, blog-matching. Old dark SVGs not
  copy-pasted in; one coherent visual language across the whole tutorial.
- **Fixed prompt `a fluffy blue creature`** (no prompt switcher) — Grant
  Sanderson's literal example from 3Blue1Brown. Greedy rollout reproduces the
  rest of his sentence ("…roamed the forest.", with "verdant" dropped for
  vocab compactness). Anchoring on his
  example lets readers cross-reference the video directly.
- **3D embeddings, ~11-word vocab** — "slightly richer" so temperature has
  real runners-up; calculator-exact verification relaxed in favour of
  believable behaviour.
- **7 stages (3 foundation : 4 transformer)** — rebalanced so the transformer
  no longer out-weighs the foundations (was 3 : 5, which violated §2.2's
  "equal weight" principle). Freed an MLP slot by merging old Attend+Mix;
  collapsed old Unembed+Sample into one Generate climax.
- **Weights fit offline, baked in.** Satisfies "no in-browser training" while
  giving believable next-word options.
- **Python ↔ JS verified numerically** (§5.5), so the page can never silently
  drift from the reference model.

---

## 13. Review checklists

Two sub-agents own review. Each completes its checklist for every stage
shipped and again for the whole tutorial at v1. Items are tied back to the
§-numbered requirements so a failure traces to a specific clause.

### 13.1 Science Officer — technical accuracy

Owner: the Science Officer sub-agent. Reads as a domain expert; checks every
claim, every number, every formula. **Goal: nothing on this page can be used
to argue Claude "doesn't know how transformers work."**

**Math correctness**

- [ ] Every numerical example is reproducible from the displayed parameters
  (§F6, §N5.1).
- [ ] `tests/equivalence.py` passes (§N5.2, §5.5).
- [ ] `tests/math.test.html` asserts the §5.6 canonical numbers and passes.
- [ ] Sigmoid formula and derivative shown correctly:
  `σ(z) = 1/(1+e⁻ᶻ)`, `σ'(z) = σ(z)(1−σ(z))`.
- [ ] Softmax uses subtraction of the max for numerical stability in code.
- [ ] Residual is `h = x + a` (input plus delta), not `h = a` (replacement).

**Conceptual accuracy**

- [ ] Stage 1 — best linear MSE on `y = [0,1,1,0]` is ≈ 0.25.
- [ ] Stage 2 — claim "still monotonic, still ≈ 0.25 floor" is true:
  `σ(wx+b)` is monotonic in `x` for any `w`.
- [ ] Stage 3 — defaults can drop loss below 0.01. Hand-verified.
- [ ] Stage 4 — claim "each row of `E` is a tiny vector like `(w,b)`" is
  conceptually fair, not literally identical to the MLP input. Disclosed.
- [ ] Stage 5 — described as **data-dependent mixing weights**, not as
  "the model looks at words like a human."
- [ ] Stage 6 — explicitly states "this is the Stage-3 machine, vectorised."
- [ ] Stage 7 — greedy, temperature, top-k each defined; the loop structure
  of generation is stated.

**Toy-vs-real honesty**

- [ ] Every §9 simplification has a one-line disclosure on-page.
- [ ] No claim that "this is how GPT works" without the qualifier "scaled up".

**Common misconceptions not introduced**

- [ ] Does NOT say "neurons are like brain neurons" or "the model
  understands / knows".
- [ ] Does NOT confuse logits with probabilities anywhere.
- [ ] Does NOT confuse sequence-length with embedding-dimension anywhere.

### 13.2 Product Manager — outcome vs requirements

Owner: the PM sub-agent. Reads as the §1.1 curious learner; plays with every
control; then audits against the spec.

**Audience fit (§1.1)**

- [ ] First appearance of every term (embedding, softmax, logit, attention,
  residual, MLP, sigmoid, MSE) is defined inline or on hover (A1).
- [ ] Each stage adds exactly one new idea on top of the prior one (A2).
- [ ] A reader who only ever clicks **Generate** still gets a faithful mental
  model (A4).
- [ ] No stage assumes a prior article has been read (A5).

**Pedagogical scope (§2)**

- [ ] All 7 stages exist as interactive panels.
- [ ] Stages 1–3 have the same interactive weight as stages 4–7. None is
  "just text".
- [ ] The §2.3 spine sentence appears prominently on the Generate stage.
- [ ] The transition between stage 3 → 4 is visually marked (C1).
- [ ] Stage 6 explicitly calls back to Stage 3 (C4).

**Functional (§3)**

- [ ] **F1** one shared pipeline; embedding edits propagate live.
- [ ] **F2** every stage has its named live control.
- [ ] **F3** Generate animates and appends; ≤ 8 tokens; reset works.
- [ ] **F4** default-T winner is `roamed`; sample mode is the only stochastic
  part.
- [ ] **F5** scrubber works; URL hash deep-links.
- [ ] **F6** every internal number is on screen or in a "Show numbers" fold.
- [ ] **F7** stages live on one page in scrollable order; 3→4 signposted.

**Non-functional (§4)**

- [ ] **N1** body text per stage ≤ ~60 words; longer hides behind a fold.
- [ ] **N2** operable at ≥ 375 px wide.
- [ ] **N3** keyboard reachable, focus rings, WCAG AA, reduced-motion.
- [ ] **N4** FCP < 1 s; no JS dep > 50 kB; no analytics; offline once loaded.
- [ ] **N5** both test files pass on a fresh checkout.
- [ ] **N6** lives at the GH-Pages URL and embeds in an iframe.
- [ ] **N7** MIT licence, footer link, README explains run/deploy/fork.

**Visual identity (§7)**

- [ ] Palette matches §7 table exactly (use a colour-picker).
- [ ] Manrope 300 for body/headings; Fira Code for numbers.
- [ ] V1–V5 visual grammar applied consistently across all 7 stages.

**Success criteria (§11)**

- [ ] All 8 items in §11 pass.

**Sign-off:** PM leaves *ships / blocks*. Blocks list the failing item by §
number so the build can fix and re-submit.
