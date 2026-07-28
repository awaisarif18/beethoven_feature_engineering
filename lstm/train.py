"""
lstm/train.py — train the LSTM on the 70% split; checkpoint at epoch 2 & 100.

Run from the PROJECT ROOT:
    python lstm/train.py

Trains on the TRAIN region only (teacher forcing: predict each next token from
the real previous tokens). Every few epochs it also reports TEST loss on the
held-out 30% — computed by running the whole true sequence with carried memory
and scoring only the test positions, so the model has the train context in its
hidden state before we judge its test predictions.

Learning rate is 5e-4 (the MLP's 5e-3 x 0.1) — a smaller, steadier step for the
LSTM. Gradient clipping guards the exploding gradients LSTMs are prone to.
"""
import math
import torch
import torch.nn as nn
from data import load
from model import LSTM

EPOCHS = 100
LR = 5e-4                 # = MLP's 5e-3 x 0.1
HIDDEN = 128
CLIP = 1.0
CHECKPOINTS = {2, 100}

# ---- data ----
encoded, s, train, test, vocab, tok2int, int2tok = load()
V = len(vocab)
enc = torch.tensor(encoded)                 # (354,) full sequence
tr = torch.tensor(train)                    # (248,) train region

x_train = tr[:-1].unsqueeze(0)              # (1, 247) inputs
y_train = tr[1:]                            # (247,)   next-token targets


def test_eval(model):
    """Run the full true sequence with carried memory; score only the test region."""
    model.eval()
    with torch.no_grad():
        logits, _ = model(enc[:-1].unsqueeze(0))     # (1, 353, 28)
        logits = logits[0, s - 1:]                    # (106, 28): predictions of the test tokens
        target = enc[s:]                              # (106,):    the real test tokens
        loss = nn.functional.cross_entropy(logits, target).item()
        acc = (logits.argmax(1) == target).float().mean().item()
    model.train()
    return loss, acc


# ---- model / loss / optimizer ----
model = LSTM(V, hidden=HIDDEN)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ---- sanity check ----
with torch.no_grad():
    logits, _ = model(x_train)
    start = criterion(logits[0], y_train).item()
print(f"untrained train loss = {start:.4f}   (expect ~= ln({V}) = {math.log(V):.4f})\n")

# ---- train ----
for epoch in range(1, EPOCHS + 1):
    optimizer.zero_grad()
    logits, _ = model(x_train)                 # (1, 247, 28)
    loss = criterion(logits[0], y_train)       # (247, 28) vs (247,)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), CLIP)
    optimizer.step()

    if epoch == 1 or epoch % 10 == 0 or epoch in CHECKPOINTS:
        tl, ta = test_eval(model)
        print(f"epoch {epoch:3d}   train loss {loss.item():.4f}   "
              f"test loss {tl:.4f}   test acc {ta:5.1%}")

    if epoch in CHECKPOINTS:
        path = f"lstm/ckpt_epoch{epoch}.pt"
        torch.save(model.state_dict(), path)
        print(f"    saved {path}")

# ---- final train memorization ----
with torch.no_grad():
    logits, _ = model(x_train)
    train_acc = (logits[0].argmax(1) == y_train).float().mean().item()
print(f"\nfinal: train acc {train_acc:.1%}  (how much of the train region it memorized)")