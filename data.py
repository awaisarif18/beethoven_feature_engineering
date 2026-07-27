"""
data.py  —  tokens.txt  ->  model-ready tensors (X windows, y targets)

The bridge from the feature-engineered token stream into the MLP's input.

NO train/val/test split: there is exactly ONE piece and the goal is to
MEMORIZE it, so every window is training data (overfitting is the point,
not a failure). This is the one real difference from the Shakespeare
build_dataset step, which split first to measure generalization.

Leaves windows as INTEGERS. The one-hot step happens one layer later, inside
the model's forward — that keeps one-hot-vs-embedding the single visible
change from yesterday, and avoids storing a big float array.

Prints every shape so you can watch the 1-D token line become (N, K) windows.

Run:  python data.py
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

K = 8   # context window: predict token K+1 from the previous K (parity with Shakespeare)


def load_tokens(path="tokens.txt"):
    """tokens.txt -> list of string tokens."""
    return open(path, encoding="utf-8").read().split()


def build_vocab(tokens):
    """Sorted, deterministic mapping token <-> int.

    sorted() is REQUIRED, not cosmetic: Python randomizes set iteration order
    per process, so an unsorted vocab gives a different token->int map on every
    run and silently breaks any saved model. (Same trap as the Shakespeare vocab.)
    """
    vocab = sorted(set(tokens))
    tok2int = {t: i for i, t in enumerate(vocab)}
    int2tok = {i: t for t, i in tok2int.items()}
    return vocab, tok2int, int2tok


def build_dataset(encoded, K):
    """(T,) int array -> X (T-K, K), y (T-K,).  K tokens -> the (K+1)th token."""
    windows = sliding_window_view(encoded, K + 1)   # view, no copy
    X = np.ascontiguousarray(windows[:, :K])
    y = np.ascontiguousarray(windows[:,  K])
    return X, y


def load(path="tokens.txt", K=K):
    """Everything train.py / generate.py need: X, y, and the vocab maps."""
    tokens = load_tokens(path)
    vocab, tok2int, int2tok = build_vocab(tokens)
    encoded = np.array([tok2int[t] for t in tokens], dtype=np.int64)
    X, y = build_dataset(encoded, K)
    return X, y, vocab, tok2int, int2tok


def main():
    tokens = load_tokens()
    vocab, tok2int, int2tok = build_vocab(tokens)
    encoded = np.array([tok2int[t] for t in tokens], dtype=np.int64)

    print(f"tokens      = {len(tokens)}")
    print(f"vocab size  = {len(vocab)}   (V)")
    print(f"vocab       = {vocab}\n")

    X, y = build_dataset(encoded, K)
    print(f"K (context) = {K}")
    print(f"X (windows) = {X.shape}  {X.dtype}    # (N, K) = num windows x context")
    print(f"y (targets) = {y.shape}  {y.dtype}       # (N,)  one next-token per window")
    print(f"N (windows) = {X.shape[0]}  =  {len(tokens)} tokens - {K}\n")

    print("first 3 windows -> target (decoded back to tokens so it's readable):")
    for j in range(min(3, len(X))):
        ctx = " ".join(int2tok[i] for i in X[j])
        print(f"    [{ctx}]  ->  {int2tok[y[j]]}")


if __name__ == "__main__":
    main()