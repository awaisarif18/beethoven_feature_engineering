"""
model.py — the MLP.

Same shape of network as the Shakespeare MLP, with the single change the
mentor's diagram asks for: the input is a fixed ONE-HOT vector, not a learned
nn.Embedding lookup. No dropout — we are memorizing one piece on purpose, and
dropout only fights memorization.

Run:  python model.py     (builds the model, prints the shapes + parameter count)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, vocab_size, K, hidden=256):
        super().__init__()
        self.vocab_size, self.K = vocab_size, K
        self.fc1 = nn.Linear(K * vocab_size, hidden)   # one-hot window feeds straight in
        self.fc2 = nn.Linear(hidden, vocab_size)       # -> logits over the vocab
        with torch.no_grad():                          # start "maximally unsure" so the
            self.fc2.weight *= 0.1                      # untrained loss is ~= ln(V) = 3.33,
            self.fc2.bias.zero_()                       # an easy wiring sanity check

    def forward(self, x, verbose=False):
        # x: (B, K) integers  ->  logits: (B, V)
        oh = F.one_hot(x, self.vocab_size).float()     # (B, K, V)  fixed, not learned
        flat = oh.view(oh.shape[0], -1)                # (B, K*V)   flatten the window
        h = torch.tanh(self.fc1(flat))                 # (B, hidden)
        logits = self.fc2(h)                           # (B, V)
        if verbose:
            print(f"  input    {tuple(x.shape)}      {x.dtype}")
            print(f"  one-hot  {tuple(oh.shape)}   (K x V, fixed)")
            print(f"  flatten  {tuple(flat.shape)}     (K*V into fc1)")
            print(f"  hidden   {tuple(h.shape)}     (after tanh)")
            print(f"  logits   {tuple(logits.shape)}")
        return logits


if __name__ == "__main__":
    V, K = 28, 8
    model = MLP(V, K)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MLP: vocab={V}, K={K}, hidden=256, params={n_params}\n")
    dummy = torch.randint(0, V, (4, K))                # a fake batch of 4 windows
    print("shapes through one forward pass:")
    model(dummy, verbose=True)