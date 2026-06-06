"""
Offline weight-fit for the Stage 4-7 toy transformer in lines-to-llms/index.html.

We do NOT train in the browser. This script fits the weights once; the rounded
result is pasted into index.html as constants. The architecture mirrors the JS
pipeline EXACTLY so the baked numbers reproduce in the page:

    X      = E[ids]                     # (T,3) embeddings of the prompt
    q      = X[-1]                       # identity W_Q: last token is the query
    scores = X @ q                       # identity W_K: each token is its own key
    alpha  = softmax(scores)             # attention weights (no sqrt(d) scaling)
    a      = alpha @ X                   # identity W_V: weighted sum of values
    h1     = X[-1] + a                   # residual (3B1B's e' = e + Δe)
    h2     = h1 + (W2 @ relu(W1 @ h1 + b1) + b2)   # the Stage-3 MLP on a vector
    logits = E @ h2                      # tied unembedding (U = E^T)
    P      = softmax(logits)

The example sentence is a trimmed version of Grant Sanderson / 3Blue1Brown's:
    "A fluffy blue creature roamed the [verdant] forest."
We drop "verdant" to keep the vocab to 9 tokens (cognitive load); the prompt
"a fluffy blue creature" still puts `creature` as the asker, matching 3B1B's
worked example exactly. Generation rollout (greedy) reproduces the sentence
minus the second adjective.

Run:  python3 tests/fit_transformer.py
"""

import numpy as np
from scipy.optimize import minimize

np.random.seed(3)

# ---- vocabulary -----------------------------------------------------------
VOCAB = ['a', 'the', 'fluffy', 'blue', 'creature', 'forest',
         'roamed', 'walked', '.']
ID = {w: i for i, w in enumerate(VOCAB)}
V = len(VOCAB)
D = 3            # embedding dim
H = 6            # MLP hidden width

# ---- toy corpus -----------------------------------------------------------
# Weighted so "a fluffy blue creature → roamed" is the clear winner with walked
# as the visible runner-up (temperature reshuffles between the two). The
# remaining sentences anchor the rollout: roamed → the → forest → .
CORPUS = (
    ['a fluffy blue creature roamed the forest .'] * 5 +
    ['a fluffy blue creature walked the forest .'] * 2 +
    ['a blue creature roamed the forest .'] * 2 +
    ['a fluffy creature roamed the forest .'] * 1 +
    ['a fluffy creature walked the forest .'] * 1
)

# Causal next-token examples: every prefix predicts the following token.
EXAMPLES = []  # (prefix_ids, target_id)
for line in CORPUS:
    toks = [ID[w] for w in line.split()]
    for k in range(len(toks) - 1):
        EXAMPLES.append((toks[:k + 1], toks[k + 1]))


def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def unpack(theta):
    i = 0
    E = theta[i:i + V * D].reshape(V, D); i += V * D
    W1 = theta[i:i + H * D].reshape(H, D); i += H * D
    b1 = theta[i:i + H]; i += H
    W2 = theta[i:i + D * H].reshape(D, H); i += D * H
    b2 = theta[i:i + D]; i += D
    return E, W1, b1, W2, b2


def forward(prefix_ids, params):
    E, W1, b1, W2, b2 = params
    X = E[prefix_ids]                       # (T,3)
    q = X[-1]
    scores = X @ q                          # (T,)
    alpha = softmax(scores)
    a = alpha @ X                           # (3,)
    h1 = X[-1] + a
    h2 = h1 + (W2 @ np.maximum(0.0, W1 @ h1 + b1) + b2)
    logits = E @ h2                         # (V,)
    return logits, softmax(logits), alpha


N_PARAMS = V * D + H * D + H + D * H + D
LAMBDA_E = 0.04     # weight decay on E only -> small embeddings -> gentle attention


def loss(theta):
    params = unpack(theta)
    E = params[0]
    nll = 0.0
    for prefix, target in EXAMPLES:
        _, P, _ = forward(prefix, params)
        nll += -np.log(P[target] + 1e-12)
    nll /= len(EXAMPLES)
    return nll + LAMBDA_E * np.sum(E ** 2)


theta0 = np.random.randn(N_PARAMS) * 0.3
res = minimize(loss, theta0, method='L-BFGS-B',
               options={'maxiter': 4000, 'maxfun': 80000})
print(f"fit: success={res.success}  final loss={res.fun:.4f}  iters={res.nit}")

# ---- round to 2 decimals (what actually ships) and RE-VERIFY ---------------
theta_r = np.round(res.x, 2)
params_r = unpack(theta_r)
E, W1, b1, W2, b2 = params_r


def show(label, ids):
    logits, P, alpha = forward(ids, params_r)
    order = np.argsort(-P)
    top = ', '.join(f"{VOCAB[j]} {P[j]*100:4.1f}%" for j in order[:4])
    att = ', '.join(f"{VOCAB[ids[k]]} {alpha[k]*100:.0f}%" for k in range(len(ids)))
    print(f"  {label:36s} -> {top}")
    print(f"  {'':36s}    attn[{att}]")
    return P


print("\nROUNDED-WEIGHT behaviour (this is what the page will show):")
fbc = [ID['a'], ID['fluffy'], ID['blue'], ID['creature']]
P_creature = show("a fluffy blue creature", fbc)
P_after_roamed = show("a fluffy blue creature roamed", fbc + [ID['roamed']])
P_after_the = show("... roamed the", fbc + [ID['roamed'], ID['the']])
P_after_forest = show("... roamed the forest", fbc + [ID['roamed'], ID['the'], ID['forest']])


def temp_dist(ids, T):
    logits, _, _ = forward(ids, params_r)
    return softmax(logits / T)


print("\ntemperature sweep for 'a fluffy blue creature' (roamed/walked):")
for T in (0.5, 1.0, 2.0):
    P = temp_dist(fbc, T)
    print(f"  T={T}:  roamed {P[ID['roamed']]*100:4.1f}%  "
          f"walked {P[ID['walked']]*100:4.1f}%")


# ---- emit JS constants -----------------------------------------------------
def js_mat(name, M):
    rows = ',\n  '.join('[' + ', '.join(f"{v:.2f}" for v in row) + ']' for row in M)
    return f"const {name} = [\n  {rows}\n];"


def js_vec(name, v):
    return f"const {name} = [{', '.join(f'{x:.2f}' for x in v)}];"

print("\n" + "=" * 64 + "\nJS CONSTANTS (paste into index.html)\n" + "=" * 64)
print(f"const VOCAB = {VOCAB};")
print(js_mat('E', E))
print(js_mat('W1', W1))
print(js_vec('B1', b1))
print(js_mat('W2', W2))
print(js_vec('B2', b2))

# canonical numbers for math.test.html
print("\n// canonical (assert in math.test.html):")
print(f"// P(roamed | 'a fluffy blue creature') = {P_creature[ID['roamed']]*100:.1f}%")
print(f"// P(walked | 'a fluffy blue creature') = {P_creature[ID['walked']]*100:.1f}%")

# ---- sanity gate (last, so constants always print above) -------------------
ok = (
    VOCAB[int(np.argmax(P_creature))] == 'roamed' and
    P_creature[ID['walked']] > 0.15 and
    VOCAB[int(np.argmax(P_after_roamed))] == 'the' and
    VOCAB[int(np.argmax(P_after_the))] == 'forest' and
    VOCAB[int(np.argmax(P_after_forest))] == '.'
)
print("\nGATE:", "PASS" if ok else "FAIL (retune corpus weights)")
