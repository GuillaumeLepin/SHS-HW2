import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

np.random.seed(42)

# load mnist dataset from openml
def load_mnist():
    print("loading mnist...")
    data = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = data.data / 255.0   # normalize
    X = X.astype(np.float32)
    y = data.target.astype(np.int64)

    # split
    X_train = X[:60000]
    X_test  = X[60000:]
    y_train = y[:60000]
    y_test  = y[60000:]

    # shuffle and make val set
    perm = np.random.permutation(60000)
    val_idx   = perm[:10000]
    train_idx = perm[10000:]

    Xtr  = X_train[train_idx]
    ytr  = y_train[train_idx]
    Xval = X_train[val_idx]
    yval = y_train[val_idx]

    print("train:", Xtr.shape, "val:", Xval.shape, "test:", X_test.shape)

    return Xtr, ytr, Xval, yval, X_test, y_test


# one hot encoding
def one_hot(y, C=10):
    n = len(y)
    Y = np.zeros((n, C))
    for i in range(n):
        Y[i, y[i]] = 1
    return Y.astype(np.float32)


def softmax(z):
    # subtract max to avoid overflow
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def loss(probs, y):
    n = len(y)
    # cross entropy
    return -np.mean(np.log(probs[np.arange(n), y] + 1e-9))


def acc(probs, y):
    return np.mean(probs.argmax(axis=1) == y)


# ===================== softmax regression =====================

class SoftmaxRegression:

    def __init__(self):
        # 784 inputs, 10 classes
        self.W = np.random.randn(784, 10).astype(np.float32) * 0.01
        self.b = np.zeros(10, dtype=np.float32)

    def forward(self, X):
        z = X @ self.W + self.b
        return softmax(z)

    def train_step(self, X, Y_oh, lr):
        probs = self.forward(X)
        N = X.shape[0]
        # gradient
        dz = (probs - Y_oh) / N
        dW = X.T @ dz
        db = dz.sum(axis=0)
        self.W -= lr * dW
        self.b -= lr * db


# ===================== MLP 1 hidden layer =====================

class MLP:

    def __init__(self, hidden=128):
        # kaiming init
        self.W1 = np.random.randn(784, hidden).astype(np.float32) * np.sqrt(2.0 / 784)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = np.random.randn(hidden, 10).astype(np.float32) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(10, dtype=np.float32)

    def forward(self, X, keep_cache=False):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0, z1)   # relu activation
        z2 = a1 @ self.W2 + self.b2
        out = softmax(z2)
        if keep_cache:
            return out, z1, a1
        return out

    def train_step(self, X, Y_oh, lr):
        N = X.shape[0]
        probs, z1, a1 = self.forward(X, keep_cache=True)

        # backprop through output layer
        dz2 = (probs - Y_oh) / N
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)

        # backprop through hidden layer
        da1 = dz2 @ self.W2.T
        dz1 = da1.copy()
        dz1[z1 <= 0] = 0    # relu derivative
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)

        # update
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1


# ===================== training =====================

def train_model(model, Xtr, ytr, Xval, yval, n_epochs=20, batch_size=128, lr=0.1, patience=3):
    Ytr = one_hot(ytr)
    N = Xtr.shape[0]

    train_losses = []
    val_losses   = []
    train_accs   = []
    val_accs     = []

    best_val_acc = 0
    best_weights = None
    no_improve   = 0

    for epoch in range(n_epochs):
        # shuffle every epoch
        idx = np.random.permutation(N)
        Xshuf = Xtr[idx]
        Yshuf = Ytr[idx]

        for start in range(0, N, batch_size):
            end   = start + batch_size
            Xb    = Xshuf[start:end]
            Yb    = Yshuf[start:end]
            model.train_step(Xb, Yb, lr)

        # evaluate
        p_train = model.forward(Xtr)
        p_val   = model.forward(Xval)

        tl = loss(p_train, ytr)
        vl = loss(p_val,   yval)
        ta = acc(p_train,  ytr)
        va = acc(p_val,    yval)

        train_losses.append(tl)
        val_losses.append(vl)
        train_accs.append(ta)
        val_accs.append(va)

        print(f"epoch {epoch+1:2d}/{n_epochs}  train_loss={tl:.4f}  val_loss={vl:.4f}"
              f"  train_acc={ta:.4f}  val_acc={va:.4f}")

        # early stopping
        if va > best_val_acc + 0.0001:
            best_val_acc = va
            # save weights
            best_weights = {}
            for k, v in vars(model).items():
                best_weights[k] = v.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve == patience:
                print(f"  -> early stop (no improvement for {patience} epochs)")
                break

    # reload best weights
    if best_weights is not None:
        for k, v in best_weights.items():
            setattr(model, k, v)

    history = {
        "train_loss": train_losses,
        "val_loss":   val_losses,
        "train_acc":  train_accs,
        "val_acc":    val_accs,
    }
    return history, best_val_acc


# ===================== plots =====================

def plot_curves(history, title, filename):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # loss
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"],   label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy loss")
    axes[0].set_title(title + " - loss")
    axes[0].legend()

    # accuracy
    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"],   label="val")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title(title + " - accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()


def plot_weight_images(W, filename):
    # visualize softmax weights as images
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for digit in range(10):
        row = digit // 5
        col = digit % 5
        img = W[:, digit].reshape(28, 28)
        m = np.abs(W).max()
        axes[row, col].imshow(img, cmap="seismic", vmin=-m, vmax=m)
        axes[row, col].set_title(f"digit {digit}")
        axes[row, col].axis("off")
    plt.suptitle("Softmax weights")
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()


# ===================== main =====================

if __name__ == "__main__":

    Xtr, ytr, Xval, yval, Xte, yte = load_mnist()

    # --- part 1 & 2: softmax regression ---
    print("\n=== Softmax Regression ===")
    sr = SoftmaxRegression()
    hist_sr, bv_sr = train_model(sr, Xtr, ytr, Xval, yval,
                                  n_epochs=30, batch_size=128, lr=0.1, patience=4)

    te_acc_sr = acc(sr.forward(Xte), yte)
    print(f"best val acc = {bv_sr:.4f}  |  test acc = {te_acc_sr:.4f}")

    plot_curves(hist_sr, "Softmax Regression", "part1_softmax_curves.png")
    plot_weight_images(sr.W, "part1_weight_images.png")

    # --- part 3: MLP ---
    print("\n=== MLP (128 hidden units) ===")
    mlp = MLP(hidden=128)
    hist_mlp, bv_mlp = train_model(mlp, Xtr, ytr, Xval, yval,
                                    n_epochs=30, batch_size=128, lr=0.1, patience=4)

    te_acc_mlp = acc(mlp.forward(Xte), yte)
    print(f"best val acc = {bv_mlp:.4f}  |  test acc = {te_acc_mlp:.4f}")

    plot_curves(hist_mlp, "MLP (128 hidden)", "part1_mlp_curves.png")

    print("\n--- final results ---")
    print(f"softmax regression : test acc = {te_acc_sr:.4f}")
    print(f"mlp (128 hidden)   : test acc = {te_acc_mlp:.4f}")
