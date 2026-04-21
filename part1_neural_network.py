"""
HW2-2 Part 1: Neural Networks on MNIST
Implements from scratch (NumPy) a softmax-regression classifier (no hidden
layer) and a 1-hidden-layer MLP (128 units, ReLU). Trained with SGD using
cross-entropy loss.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

rng = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Part 0: Load MNIST
# ---------------------------------------------------------------------------
def load_mnist():
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data.astype(np.float32) / 255.0   # scale to [0, 1]
    y = mnist.target.astype(np.int64)
    # The canonical MNIST split: first 60k train, last 10k test
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    # Further carve out a validation set from training data
    perm = rng.permutation(60000)
    val_idx, tr_idx = perm[:10000], perm[10000:]
    return (X_train[tr_idx], y_train[tr_idx],
            X_train[val_idx], y_train[val_idx],
            X_test, y_test)


def one_hot(y, C=10):
    Y = np.zeros((y.size, C), dtype=np.float32)
    Y[np.arange(y.size), y] = 1.0
    return Y


def softmax(z):
    # Numerically stable softmax (subtract row max)
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)


def cross_entropy(probs, y_int):
    n = probs.shape[0]
    return -np.log(probs[np.arange(n), y_int] + 1e-12).mean()


def accuracy(probs, y_int):
    return (probs.argmax(axis=1) == y_int).mean()


# ---------------------------------------------------------------------------
# Part 1-2: Softmax regression  f(x) = softmax(W x + b)
# ---------------------------------------------------------------------------
class SoftmaxRegression:
    def __init__(self, in_dim=784, out_dim=10):
        # He-ish init (small random)
        self.W = rng.normal(0, 0.01, size=(in_dim, out_dim)).astype(np.float32)
        self.b = np.zeros(out_dim, dtype=np.float32)

    def forward(self, X):
        return softmax(X @ self.W + self.b)

    def step(self, X, Y, lr):
        # Y is one-hot; gradient of cross-entropy w.r.t. pre-softmax is (p - Y)
        probs = self.forward(X)
        n = X.shape[0]
        dlogits = (probs - Y) / n
        self.W -= lr * (X.T @ dlogits)
        self.b -= lr * dlogits.sum(axis=0)
        return probs


# ---------------------------------------------------------------------------
# Part 3: one hidden layer (128 units, ReLU) + softmax
# ---------------------------------------------------------------------------
class MLP:
    def __init__(self, in_dim=784, hidden=128, out_dim=10):
        # Kaiming init for ReLU
        self.W1 = rng.normal(0, np.sqrt(2.0 / in_dim),
                             size=(in_dim, hidden)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden),
                             size=(hidden, out_dim)).astype(np.float32)
        self.b2 = np.zeros(out_dim, dtype=np.float32)

    def forward(self, X, cache=False):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(z1, 0)                      # ReLU
        z2 = a1 @ self.W2 + self.b2
        probs = softmax(z2)
        if cache:
            return probs, (X, z1, a1)
        return probs

    def step(self, X, Y, lr):
        probs, (X, z1, a1) = self.forward(X, cache=True)
        n = X.shape[0]
        dz2 = (probs - Y) / n
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)
        self.W2 -= lr * dW2; self.b2 -= lr * db2
        self.W1 -= lr * dW1; self.b1 -= lr * db1
        return probs


# ---------------------------------------------------------------------------
# Training loop (mini-batch SGD)
# ---------------------------------------------------------------------------
def train(model, X_tr, y_tr, X_val, y_val,
          epochs=20, batch_size=128, lr=0.1, patience=3):
    Y_tr = one_hot(y_tr)
    n = X_tr.shape[0]

    history = {"train_acc": [], "val_acc": [],
               "train_loss": [], "val_loss": []}

    best_val = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        perm = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            model.step(X_tr[idx], Y_tr[idx], lr)

        p_tr = model.forward(X_tr)
        p_val = model.forward(X_val)
        tl = cross_entropy(p_tr, y_tr)
        vl = cross_entropy(p_val, y_val)
        ta = accuracy(p_tr, y_tr)
        va = accuracy(p_val, y_val)
        history["train_acc"].append(ta)
        history["val_acc"].append(va)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        print(f"epoch {epoch:2d} | loss tr={tl:.4f} val={vl:.4f} | "
              f"acc tr={ta:.4f} val={va:.4f}")

        # Early-stopping convergence criterion: best val acc hasn't improved
        # for `patience` epochs.
        if va > best_val + 1e-4:
            best_val = va
            best_state = {k: v.copy() for k, v in vars(model).items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Converged (no val-acc improvement for {patience} epochs).")
                break

    if best_state is not None:
        for k, v in best_state.items():
            setattr(model, k, v)
    return history, best_val


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
def plot_weight_images(W, out_path):
    # W has shape (784, 10) — one column per digit class
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        img = W[:, i].reshape(28, 28)
        ax.imshow(img, cmap="seismic",
                  vmin=-np.abs(W).max(), vmax=np.abs(W).max())
        ax.set_title(f"class {i}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_history(history, out_path, title):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history["train_loss"], label="train")
    ax[0].plot(history["val_loss"], label="val")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("cross-entropy")
    ax[0].set_title(f"{title} — loss"); ax[0].legend()
    ax[1].plot(history["train_acc"], label="train")
    ax[1].plot(history["val_acc"], label="val")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("accuracy")
    ax[1].set_title(f"{title} — accuracy"); ax[1].legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading MNIST...")
    X_tr, y_tr, X_val, y_val, X_te, y_te = load_mnist()
    print(f"  train={X_tr.shape}  val={X_val.shape}  test={X_te.shape}")

    # --- Part 1 & 2: softmax regression (no hidden layer) ---
    print("\n=== Part 1/2: softmax regression ===")
    sr = SoftmaxRegression()
    hist_sr, best_val_sr = train(sr, X_tr, y_tr, X_val, y_val,
                                 epochs=30, batch_size=128, lr=0.1,
                                 patience=4)
    test_acc_sr = accuracy(sr.forward(X_te), y_te)
    print(f"softmax regression — best val acc = {best_val_sr:.4f}, "
          f"test acc = {test_acc_sr:.4f}")

    plot_history(hist_sr, "part1_softmax_curves.png", "softmax regression")
    plot_weight_images(sr.W, "part1_weight_images.png")

    # --- Part 3: MLP with 128-unit hidden layer ---
    print("\n=== Part 3: MLP with 128-unit hidden layer ===")
    mlp = MLP(hidden=128)
    hist_mlp, best_val_mlp = train(mlp, X_tr, y_tr, X_val, y_val,
                                   epochs=30, batch_size=128, lr=0.1,
                                   patience=4)
    test_acc_mlp = accuracy(mlp.forward(X_te), y_te)
    print(f"MLP — best val acc = {best_val_mlp:.4f}, "
          f"test acc = {test_acc_mlp:.4f}")

    plot_history(hist_mlp, "part1_mlp_curves.png", "MLP (128 hidden)")

    print("\n=== Summary ===")
    print(f"  softmax regression  best val acc = {best_val_sr:.4f}   "
          f"test = {test_acc_sr:.4f}")
    print(f"  MLP (128 hidden)    best val acc = {best_val_mlp:.4f}   "
          f"test = {test_acc_mlp:.4f}")


if __name__ == "__main__":
    main()
