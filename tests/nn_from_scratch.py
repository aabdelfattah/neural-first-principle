"""
Tiny neural network from scratch — comparison of 1 vs 2 hidden neurons
on a non-monotonic target (XOR-like in 1D).

Run:
    python3 nn_from_scratch.py

Produces loss.png with:
  - left:  loss curves for both networks
  - right: learned functions vs the training data
"""

import numpy as np
import matplotlib.pyplot as plt

# ── Data: non-monotonic target ──
X = np.array([1.0, 2.0, 3.0, 4.0])
Y = np.array([0.0, 1.0, 1.0, 0.0])
n = len(X)
EPOCHS = 5000
LR = 0.5

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ────────────────────────────────────────────────────────────────────
# 1 hidden neuron — the original tutorial network. Cannot fit the data
# because a single sigmoid + linear output is monotonic in x.
# ────────────────────────────────────────────────────────────────────
def train_1_neuron():
    w1, b1 = 0.5, -0.1
    w2, b2 = 0.3, 0.2
    losses = []

    def forward(x):
        z1 = w1 * x + b1
        a1 = sigmoid(z1)
        y_hat = w2 * a1 + b2
        return z1, a1, y_hat

    def backward(x, y, a1, y_hat):
        dL_dyhat = y_hat - y
        gw2 = dL_dyhat * a1
        gb2 = dL_dyhat
        dL_dz1 = dL_dyhat * w2 * a1 * (1 - a1)
        gw1 = dL_dz1 * x
        gb1 = dL_dz1
        return gw1, gb1, gw2, gb2

    def update_weights(gw1, gb1, gw2, gb2):
        nonlocal w1, b1, w2, b2
        w1 -= LR * gw1 / n
        b1 -= LR * gb1 / n
        w2 -= LR * gw2 / n
        b2 -= LR * gb2 / n

    for _ in range(EPOCHS):
        gw1 = gb1 = gw2 = gb2 = 0.0
        total = 0.0
        for x, y in zip(X, Y):
            z1, a1, y_hat = forward(x)
            total += 0.5 * (y_hat - y) ** 2
            dw1, db1, dw2, db2 = backward(x, y, a1, y_hat)
            gw1 += dw1
            gb1 += db1
            gw2 += dw2
            gb2 += db2
        update_weights(gw1, gb1, gw2, gb2)
        losses.append(total / n)

    def predict(x):
        return w2 * sigmoid(w1 * x + b1) + b2
    return losses, predict


# ────────────────────────────────────────────────────────────────────
# 2 hidden neurons — can express a bump shape, fits the data exactly.
# Initialization is hand-picked so the two sigmoids land in useful regions
# (random init often collapses both neurons into the flat tails → dead gradients).
# ────────────────────────────────────────────────────────────────────
def train_2_neurons():
    W1 = np.array([3.0, -3.0])
    B1 = np.array([-4.5, 10.5])
    W2 = np.array([1.0, 1.0])
    b2 = 0.0
    losses = []

    def forward(x):
        z1 = W1 * x + B1
        a1 = sigmoid(z1)
        y_hat = np.dot(W2, a1) + b2
        return z1, a1, y_hat

    def backward(x, y, a1, y_hat):
        dL_dyhat = y_hat - y
        gW2 = dL_dyhat * a1
        gb2 = dL_dyhat
        dL_dz1 = dL_dyhat * W2 * a1 * (1 - a1)
        gW1 = dL_dz1 * x
        gB1 = dL_dz1
        return gW1, gB1, gW2, gb2

    def update_weights(gW1, gB1, gW2, gb2):
        nonlocal W1, B1, W2, b2
        W1 -= LR * gW1 / n
        B1 -= LR * gB1 / n
        W2 -= LR * gW2 / n
        b2 -= LR * gb2 / n

    for _ in range(EPOCHS):
        gW1 = np.zeros(2); gB1 = np.zeros(2)
        gW2 = np.zeros(2); gb2 = 0.0
        total = 0.0
        for x, y in zip(X, Y):
            z1, a1, y_hat = forward(x)
            total += 0.5 * (y_hat - y) ** 2
            dW1, dB1, dW2, db2 = backward(x, y, a1, y_hat)
            gW1 += dW1
            gB1 += dB1
            gW2 += dW2
            gb2 += db2
        update_weights(gW1, gB1, gW2, gb2)
        losses.append(total / n)

    def predict(x):
        return np.dot(W2, sigmoid(W1 * x + B1)) + b2
    return losses, predict


losses_1, pred_1 = train_1_neuron()
losses_2, pred_2 = train_2_neurons()

print(f"1-neuron final loss: {losses_1[-1]:.5f}  (stuck — can't fit)")
print(f"2-neuron final loss: {losses_2[-1]:.2e}  (~0)")
print("\n1-neuron predictions:")
for x, y in zip(X, Y):
    print(f"  x={x:.1f}  true={y:.1f}  pred={pred_1(x):.3f}")
print("\n2-neuron predictions:")
for x, y in zip(X, Y):
    print(f"  x={x:.1f}  true={y:.1f}  pred={pred_2(x):.3f}")

# ── Plot (dark theme to match tutorial page) ──
BG    = "#14171e"
FG    = "#d4d8e3"
GRID  = "#262b38"
RED   = "#e85d75"   # accent2 — failing 1-neuron
GREEN = "#5be8a0"   # accent4 — working 2-neurons
DOT   = "#f09838"   # accent  — training data

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": FG,
    "xtick.color": FG, "ytick.color": FG,
    "text.color": FG, "axes.titlecolor": FG,
    "grid.color": GRID, "legend.facecolor": BG, "legend.edgecolor": GRID,
})

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(losses_1, label="1 hidden neuron", color=RED, linewidth=2)
axes[0].plot(losses_2, label="2 hidden neurons", color=GREEN, linewidth=2)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss (log scale)")
axes[0].set_title("Training Loss")
axes[0].set_yscale("log")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

xs = np.linspace(0, 5, 300)
axes[1].plot(xs, [pred_1(x) for x in xs], color=RED,   linewidth=2, label="1 neuron (monotonic — stuck)")
axes[1].plot(xs, [pred_2(x) for x in xs], color=GREEN, linewidth=2, label="2 neurons (bump — fits)")
axes[1].scatter(X, Y, color=DOT, s=80, zorder=5, label="training data", edgecolors=FG, linewidth=1)
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].set_title("Learned Function")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.savefig("loss.png", dpi=120)
print("\nSaved plot to loss.png")
