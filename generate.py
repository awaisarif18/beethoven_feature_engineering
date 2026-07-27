"""
generate.py — greedy replay of the memorized piece.

Seeds the model with the true opening (first K tokens), then lets it predict
the rest one token at a time — always the most likely token (greedy / argmax),
sliding the window and feeding its OWN output back.

Because greedy is deterministic, and the piece repeats verbatim then exits
differently, the replay stays perfect for a while and then diverges at the
first ambiguous repeat: the model can make only ONE choice where the real piece
made two, so from there it falls into a loop. That divergence point IS the
lesson — the exact cost of a fixed context window with no sense of "where am I".

Writes replayed_tokens.txt. Turn it back into MIDI by REUSING decode.py:
    python decode.py replayed_tokens.txt furelise.mid

Run:  python generate.py
"""
import torch
from data import load, load_tokens, K
from model import MLP

# ---- vocab + the true token stream (for seeding and comparison) ----
_, _, vocab, tok2int, int2tok = load()
V = len(vocab)
true_tokens = load_tokens()                       # the original 354 tokens
N = len(true_tokens)

# ---- rebuild the model and load the trained weights ----
model = MLP(V, K)
model.load_state_dict(torch.load("model.pt", weights_only=True))
model.eval()

# ---- greedy replay ----
context = [tok2int[t] for t in true_tokens[:K]]   # seed = the true opening
out = list(context)
with torch.no_grad():
    for _ in range(N - K):                        # fill out to the original length
        x = torch.tensor([context])               # (1, K)
        nxt = int(model(x).argmax(dim=1))         # most likely next token
        out.append(nxt)
        context = context[1:] + [nxt]             # slide the window forward

replayed = [int2tok[i] for i in out]
with open("replayed_tokens.txt", "w", encoding="utf-8") as f:
    f.write(" ".join(replayed))

# ---- how long did it stay faithful before diverging? ----
first_diff = next((i for i in range(N) if replayed[i] != true_tokens[i]), N)
matches = sum(a == b for a, b in zip(replayed, true_tokens))
print(f"replayed {N} tokens -> replayed_tokens.txt")
print(f"faithful for the first {first_diff} tokens, then diverges into the loop")
print(f"matches {matches}/{N} of the original overall")
print("\nturn it into MIDI:  python decode.py replayed_tokens.txt furelise.mid")