"""
For each checkpoint it writes two token streams over the held-out 30%:
  test_epochN  — teacher forcing: feed the REAL previous notes, record predictions
  pred_epochN  — free running:    the model generates the section from its OWN notes
Plus truth_test — the real held-out music, as the answer key.

Each token stream is turned into its own named MIDI by reusing decode.py's
builder functions, so decode.py itself stays untouched.
"""
import os
import sys
import torch
import mido

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # find root decode.py
import decode
from data import load
from model import LSTM

HIDDEN = 128


def tokens_to_midi(tokens, mid_path):
    """token list -> named .mid, reusing decode.py's builders (decode stays untouched)."""
    txt_path = mid_path.replace(".mid", ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(" ".join(tokens))
    _, columns = decode.load_columns(txt_path)
    events = decode.columns_to_events(columns)
    rh = [(st, p) for h, st, p in events if h == decode.RH]
    lh = [(st, p) for h, st, p in events if h == decode.LH]
    mid = mido.MidiFile(ticks_per_beat=decode.TICKS_PER_BEAT)
    mid.tracks.append(decode.build_track(rh, "Piano", with_tempo=True))
    mid.tracks.append(decode.build_track(lh, "Piano", with_tempo=False))
    mid.save(mid_path)
    print(f"  wrote {mid_path}")


def teacher_forced_test(model, enc, s):
    """Feed the whole true sequence; return the model's predicted TEST tokens (ids)."""
    model.eval()
    with torch.no_grad():
        logits, _ = model(enc[:-1].unsqueeze(0))      # (1, 353, V)
        pred = logits[0, s - 1:].argmax(1)             # (106,)
    return pred.tolist()


def free_run(model, prime_ids, n):
    """Prime on the train region, then generate n tokens from the model's OWN output."""
    model.eval()
    out = []
    with torch.no_grad():
        logits, hidden = model(prime_ids.unsqueeze(0))     # read the train region
        nxt = int(logits[0, -1].argmax())                   # first held-out token
        out.append(nxt)
        for _ in range(n - 1):
            logits, hidden = model(torch.tensor([[nxt]]), hidden)
            nxt = int(logits[0, -1].argmax())
            out.append(nxt)
    return out


def main():
    encoded, s, train, test, vocab, tok2int, int2tok = load()
    V = len(vocab)
    enc = torch.tensor(encoded)
    tr = torch.tensor(train)
    n_test = len(test)                                  # 106

    # answer key: decode the real held-out music once
    tokens_to_midi([int2tok[i] for i in test], "lstm/truth_test.mid")

    for epoch in (2, 100, 200):
        ckpt = f"lstm/ckpt_epoch{epoch}.pt"
        model = LSTM(V, hidden=HIDDEN)
        model.load_state_dict(torch.load(ckpt, weights_only=True))
        print(f"\n[{ckpt}]")

        tf = teacher_forced_test(model, enc, s)
        tokens_to_midi([int2tok[i] for i in tf], f"lstm/test_epoch{epoch}.mid")

        fr = free_run(model, tr, n_test)
        tokens_to_midi([int2tok[i] for i in fr], f"lstm/pred_epoch{epoch}.mid")

        tf_match = sum(a == b for a, b in zip(tf, test)) / n_test
        fr_match = sum(a == b for a, b in zip(fr, test)) / n_test
        print(f"  test (teacher-forced) matches truth: {tf_match:.1%}")
        print(f"  pred (free-run)       matches truth: {fr_match:.1%}")


if __name__ == "__main__":
    main()