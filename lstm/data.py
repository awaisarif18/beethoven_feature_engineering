"""
lstm/data.py — tokens.txt -> a 70/30 train/test split for the LSTM.

Run from the PROJECT ROOT so tokens.txt is found:
    python lstm/data.py

No sliding windows this time. The LSTM reads the sequence in order, so we only
need the token stream split into a train part and a held-out test part.

The split is CONTIGUOUS (first 70% train, last 30% test) — this is one time
series, so shuffling would leak the answer. It is measured in whole grid COLUMNS
(2 tokens each), so the cut always lands on an even token index: each half
starts on an R: token and decodes to clean MIDI (decode.py pairs tokens as
(R, L) per column).
"""
import numpy as np

TRAIN_FRAC = 0.70


def load_tokens(path="tokens.txt"):
    return open(path, encoding="utf-8").read().split()


def build_vocab(tokens):
    """Sorted, deterministic token <-> int map (sorted is required — a shuffled
    map would silently break a saved model, same trap as the MLP)."""
    vocab = sorted(set(tokens))
    tok2int = {t: i for i, t in enumerate(vocab)}
    int2tok = {i: t for t, i in tok2int.items()}
    return vocab, tok2int, int2tok


def split_index(n):
    """70% point measured in whole grid columns (2 tokens each), so both halves
    are column-aligned and the test half starts on an R: token."""
    n_cols = n // 2
    train_cols = round(n_cols * TRAIN_FRAC)
    return train_cols * 2


def load(path="tokens.txt"):
    tokens = load_tokens(path)
    vocab, tok2int, int2tok = build_vocab(tokens)
    encoded = np.array([tok2int[t] for t in tokens], dtype=np.int64)
    s = split_index(len(encoded))
    train, test = encoded[:s], encoded[s:]
    return encoded, s, train, test, vocab, tok2int, int2tok


def main():
    encoded, s, train, test, vocab, tok2int, int2tok = load()
    N = len(encoded)
    print(f"tokens      = {N}")
    print(f"vocab size  = {len(vocab)}")
    print(f"split index = {s}   (even -> grid-column boundary)\n")

    print(f"train = tokens[:{s}]   -> {len(train):3d} tokens = {len(train)//2} columns "
          f"({len(train)/N:.1%})")
    print(f"test  = tokens[{s}:]  -> {len(test):3d} tokens = {len(test)//2} columns "
          f"({len(test)/N:.1%})\n")

    first_test = int2tok[test[0]]
    ok = "OK, starts on R:" if first_test.startswith("R:") else "WARNING: not an R: token"
    print(f"first test token = {first_test!r}   ({ok})\n")

    print("last 4 train tokens :", [int2tok[i] for i in train[-4:]])
    print("first 4 test tokens :", [int2tok[i] for i in test[:4]])


if __name__ == "__main__":
    main()