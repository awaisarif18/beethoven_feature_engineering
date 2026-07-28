# Für Elise: MLP → LSTM — presentation script

## 0. The one-sentence version

Say this first: *"We turned a piano piece into text, proved that conversion is
lossless, then fed the text through two different neural nets — one with a
fixed 8-token window, one with memory — to prove the tokens survive a real
model, and to physically see the difference a fixed window makes vs. memory."*

The feature engineering (encode/decode) is the actual deliverable. Both
networks exist to prove the tokens round-trip through a real model.

---

## 1. File map — what lives where, what each one does

### Root — shared, untouched by either phase
| file | what it does |
|---|---|
| `furelise.mid` | source MIDI: 2 tracks, 480 ticks/beat, 199 note-ons, 22 pitches |
| `encode.py` | MIDI → `tokens.txt`. Lays a 16th-note grid (120 ticks/slot), writes one `R:`/`L:` token per hand per slot. Lossless. |
| `decode.py` | `tokens.txt` → MIDI, the exact inverse. Also exposes `parse_token`, `load_columns`, `columns_to_events`, `build_track` — reused directly by `lstm/generate.py` so decode.py itself never changes. |
| `tokens.txt` | 354 tokens, 28-symbol vocabulary — the actual "dataset" both phases train on |

### Root — Phase 1 (MLP), history, done
| file | what it does |
|---|---|
| `data.py` | builds K=8 sliding windows: `X (346,8)`, `y (346,)`. No train/test split — the goal was memorize-and-replay, not generalization. |
| `model.py` | one-hot `(B,8)` → flatten `(B,224)` → `Linear(224,256)` + tanh → `Linear(256,28)`. **64,796 params.** |
| `train.py` | full-batch, 2000 epochs, Adam lr `5e-3`. Loss floored at **0.0475** (10 windows are genuinely ambiguous — same 8-token history, different next token). Memorized **336/346 (97.1%)**. |
| `generate.py` | greedy replay. Faithful for **36 tokens**, then loops — a fixed window can't tell repeated passages apart. |
| `model.pt` | the saved trained MLP weights |

### `lstm/` — Phase 2, current
| file | what it does |
|---|---|
| `data.py` | `tokens.txt` → a **70/30 contiguous split**, cut on a whole grid-column boundary so both halves decode cleanly. No windows — the LSTM reads the sequence in order. |
| `model.py` | `nn.LSTM(28 → 128)` + `Linear(128 → 28)`. **84,508 params.** `forward()` returns `(logits, hidden)` so the hidden/cell state can be threaded across calls. |
| `train.py` | trains only on the train region (teacher forcing), evaluates the held-out region with carried memory. Checkpoints saved at epoch 2, 100, 200. |
| `generate.py` | for each checkpoint: teacher-forced prediction (`test_epochN.mid`) and free-run generation (`pred_epochN.mid`) over the held-out region, plus `truth_test.mid` as the answer key. |

---

## 2. `lstm/data.py` — what runs, in order

```
load_tokens("tokens.txt")   -> list of 354 strings
build_vocab(tokens)         -> vocab (28 strings), tok2int, int2tok
encoded = [tok2int[t] ...]  -> NumPy array (354,)
split_index(354)            -> 248   (124 columns x 2, 70.1% of 177 columns)
train = encoded[:248]       -> (248,)
test  = encoded[248:]       -> (106,)
```

**Note for the mentor:** `data.py` also has a `main()` that prints a summary
table — that only runs if you execute `python lstm/data.py` directly.
`train.py` and `generate.py` only ever call `load()`, never `main()`.

---

## 3. `lstm/model.py` — the forward pass, one call

Whether it's training, testing, or generation, every call runs the same
four-stage pipeline. Only the length `L` and whether `hidden` is reused
differs.

```
x            (1, L)  int matrix        <- token ids
  | one_hot
(1, L, 28)   tensor                    <- L one-hot vectors stacked
  | nn.LSTM
(1, L, 128)  tensor  "out"             <- L hidden vectors, one per step
  + hidden = (h, c), each (1, 1, 128)  <- state AFTER the last step only
  | fc (Linear)
(1, L, 28)   tensor  "logits"          <- L score-vectors, one per step
```

---

## 4. `lstm/train.py` — execution order

```
load()                      -> tensors
model = LSTM(28, hidden=128)
optimizer = Adam(lr=5e-3)
[sanity check] one forward(x_train) call -> untrained loss ~= ln(28) = 3.33

for epoch in 1..200:
    forward(x_train)              (1,247) -> logits (1,247,28)
    loss = cross_entropy(logits[0] (247,28), y_train (247,))   -> scalar
    backward(); clip_grad_norm_(1.0); optimizer.step()
    [every 10 epochs] test_eval(model):
        forward(enc[:-1])          (1,353) -> logits (1,353,28)
        slice test rows            -> (106,28) vs target (106,)  -> scalar loss, scalar acc
    [epoch in {2,100,200}] torch.save(...)   -> writes lstm/ckpt_epochN.pt
```

`forward()` is called 200+ times in this one run — once per epoch, plus
~21 more for the periodic test evaluations.

---

## 5. `lstm/generate.py` — execution order

```
main():
    load()                                    -> tensors (re-derived, deterministic)
    tokens_to_midi(test, "truth_test.mid")    -> the answer key

    for epoch in (2, 100, 200):
        model = LSTM(28, 128)                 <- fresh, untrained object
        model.load_state_dict(torch.load(f"lstm/ckpt_epoch{epoch}.pt"))

        tf = teacher_forced_test(model, enc, s)
            forward(enc[:-1])      (1,353) -> logits (1,353,28) -> slice+argmax -> (106,) ids
        tokens_to_midi(tf, "test_epochN.mid")

        fr = free_run(model, tr, 106)
            forward(tr)             (1,248) -> logits, hidden      [priming call]
            nxt = argmax(logits[0,-1])                              [scalar]
            repeat x105:
                forward([[nxt]], hidden)   (1,1) -> logits (1,1,28), NEW hidden
                nxt = argmax(logits[0,-1])                          [scalar]
                append nxt to output list
        tokens_to_midi(fr, "pred_epochN.mid")

        print match% for tf and fr against the true test tokens
```

**The one sentence that matters most for the mentor:** in every call above
except `free_run`'s loop, `hidden` is computed and thrown away — the model
starts fresh each time. In the loop, `hidden` is explicitly passed back in
on every single call — that's the only mechanism that gives the LSTM memory
across the 105 generated tokens instead of guessing blind at every step.

---

## 6. Two diagrams worth actually drawing on the table

**Diagram A — the forward pass (draw once, reuse for train/test):**
```
[ x: (1,L) ints ]
        |
[ one-hot: (1,L,28) ]
        |
[ LSTM out: (1,L,128) ]  ---- hidden (1,1,128) x2, LAST step only
        |
[ logits: (1,L,28) ]
```

**Diagram B — free-run generation (the part that's genuinely different):**
```
[ Prime call: (1,248) real tokens ] --> hidden
        |
        v
[ Loop step: (1,1) own last guess ] <---+
        |                               |
        +--- new hidden, new guess -----+   (repeats 105x)
```

---

## 7. Results — what to say about overfitting

| epoch | train loss | test loss | test acc |
|---:|---:|---:|---:|
| 2 | 3.30 | 3.24 | 34.9% |
| 50 | 1.73 | 1.58 | 57.5% |
| 100 | 0.71 | 0.93 | 74.5% |
| 130 | 0.43 | **0.76** | 84.9% |
| 200 | 0.06 | 0.99 | 83.0% |

Say: *"Train loss falls the whole way to 0.06 — the model is memorizing the
piece almost perfectly. But test loss bottoms out around epoch 130 and then
climbs back up even as train loss keeps falling. That gap is overfitting,
made visible only because we added the 70/30 split — the MLP never showed
this because it never had a held-out region to fail on."*

At epoch 2, both `test_epoch2.mid` and `pred_epoch2.mid` decode to **empty
MIDI files** — not a bug. The barely-trained model predicts a rest at every
position (rests are the majority token; 34.9% acc is exactly the rest
fraction), and an all-rest token stream has zero notes.

---

## 8. Closing line

*"The MLP proved the round trip works and showed the honest limit of a fixed
window — a 36-token loop. The LSTM proved memory fixes that specific failure,
but the train/test split showed it introduces a different one: overfitting.
Neither model was the point — the token representation surviving both of them
unchanged is."*
