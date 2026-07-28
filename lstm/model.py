"""
lstm/model.py — the LSTM.

Replaces the MLP's fixed 8-token window with memory. The LSTM reads the sequence
one token at a time and carries a hidden state (+ cell state) forward, so its
context is not capped at K tokens — that's the whole reason we switched.

Same one-hot input as the MLP (28-dim), a single LSTM layer, and a Linear head
that turns each step's hidden vector into 28 next-token scores. No dropout — we
are still memorizing.

forward(x, hidden) returns (logits, hidden):
  - training passes the WHOLE sequence with hidden=None and uses only the logits
  - generation passes ONE token at a time and threads `hidden` back in, so the
    model remembers what it has already produced

Run:  python lstm/model.py      (builds it, prints shapes + parameter count)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, vocab_size, hidden=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.lstm = nn.LSTM(input_size=vocab_size, hidden_size=hidden,
                            num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden, vocab_size)      # hidden vector -> 28 scores
        with torch.no_grad():                        # start maximally unsure:
            self.fc.weight *= 0.1                     # untrained loss ~= ln(V) = 3.33
            self.fc.bias.zero_()

    def forward(self, x, hidden=None):
        # x: (B, L) integers -> logits: (B, L, V), plus the carried state
        oh = F.one_hot(x, self.vocab_size).float()   # (B, L, V)
        out, hidden = self.lstm(oh, hidden)          # out: (B, L, H)
        logits = self.fc(out)                        # (B, L, V)
        return logits, hidden


if __name__ == "__main__":
    V, H = 28, 128
    model = LSTM(V, hidden=H)
    n = sum(p.numel() for p in model.parameters())
    print(f"LSTM: vocab={V}, hidden={H}, params={n}\n")

    dummy = torch.randint(0, V, (1, 10))             # 1 sequence, length 10
    logits, (h, c) = model(dummy)
    print("shapes through one forward pass (whole sequence):")
    print(f"  input    {tuple(dummy.shape)}       {dummy.dtype}")
    print(f"  one-hot  {tuple(F.one_hot(dummy, V).shape)}")
    print(f"  logits   {tuple(logits.shape)}   (a 28-score prediction at every step)")
    print(f"  hidden h {tuple(h.shape)}    cell c {tuple(c.shape)}   (the carried memory)")