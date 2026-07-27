"""
train.py — memorize Für Elise, then save the model.

One piece, one goal: overfit it. We train on ALL 346 windows at once — the whole
dataset is a single batch, so no DataLoader is needed. Runs on CPU; the data is
far too small for the GPU to be worth the transfer.

The loss will NOT reach zero. A few spots in the piece show the same 8-token
history followed by DIFFERENT next tokens (the theme repeats and exits
differently each time), and no model can be right both times. That leftover is
the loss floor — expected, not a bug.

Run:  python train.py
"""
import math
import torch
import torch.nn as nn
from data import load, K
from model import MLP

EPOCHS = 2000
LR = 5e-3

# ---- data: the whole piece, no split ----
X, y, vocab, tok2int, int2tok = load()
V = len(vocab)
X = torch.tensor(X)      # (346, 8) int64
y = torch.tensor(y)      # (346,)   int64

# ---- model / loss / optimizer ----
model = MLP(V, K)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ---- sanity check: an untrained model should be maximally unsure ----
with torch.no_grad():
    start_loss = criterion(model(X), y).item()
print(f"untrained loss = {start_loss:.4f}   (expect ~= ln({V}) = {math.log(V):.4f})\n")

# ---- train ----
for epoch in range(1, EPOCHS + 1):
    optimizer.zero_grad()
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step()
    if epoch == 1 or epoch % 200 == 0:
        print(f"epoch {epoch:5d}   loss {loss.item():.4f}")

# ---- how much did it actually memorize? ----
with torch.no_grad():
    correct = (model(X).argmax(dim=1) == y).sum().item()
print(f"\nmemorized {correct}/{len(y)} windows ({correct/len(y):.1%}) "
      f"— the misses are the ambiguous repeats")

torch.save(model.state_dict(), "model.pt")
print("saved model.pt")