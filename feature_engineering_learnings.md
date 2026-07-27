# Feature Engineering Learnings — Für Elise through an MLP

## What this project was

I took a piano performance stored as a MIDI file, turned it into a line of
text-like tokens, proved that conversion loses nothing, fed those tokens
through a small neural network that memorized the piece, and let the network
play it back. It reproduced the opening perfectly and then fell into a loop.

The **feature engineering** — the MIDI ↔ token conversion — was the real
deliverable. The network sat in the middle only to prove the tokens survive a
round trip through a genuine learning model. The variation from the previous
(Shakespeare) project: **one-hot input instead of a learned embedding**.

Files: `encode.py`, `decode.py`, `data.py`, `model.py`, `train.py`,
`generate.py`. Six files, each one job.

---

## The data at each phase, with shapes

### Input: the raw MIDI

`furelise.mid` is **not audio** — it's a list of timed instructions ("press
pitch 76 now, release later"). The jargon is *events*: note-on / note-off, not
sound. That's why it's tiny. (mido docs: https://mido.readthedocs.io/)

- 2 tracks → track 0 = right hand, track 1 = left hand
- ruler: **480 ticks per beat**
- **199 note-presses** (134 right, 65 left), 22 distinct pitches

Shape to picture: 2-D — time running left-to-right, two hands stacked.

### Phase 1 — Encode: MIDI → tokens (the feature engineering)

Recovered each note's true time, snapped it onto a grid, lined the hands up,
wrote each slot as a symbol, then flattened to one line. Output: `tokens.txt`.

Concepts I learned here:

- **Delta-time.** MIDI stores each event's time as an offset *from the previous
  event*, not as an absolute time. You recover the real time by adding them up
  (`abs_tick += msg.time`). This is the #1 gotcha in reading MIDI.
- **`velocity > 0`.** A "note-on with zero force" is really a *release* in
  disguise, so only note-ons with velocity above zero are real key-presses. The
  code keeps *onsets* only; it ignores how long a note is held.
- **Quantization.** Real timing is smooth and slightly imperfect; a model needs
  clean countable slots. `GRID_DEN = 4` lays a sixteenth-note grid (120 ticks
  per slot) and `round(t / grid)` snaps each note to its nearest slot. Jargon:
  *quantization*. (https://en.wikipedia.org/wiki/Quantization_(music))
- **Flattening / serialization.** The two-hand table gets written out as one
  ribbon, alternating right-cell, left-cell. This is the moment 2-D music
  becomes a 1-D sequence — the shape a language model wants.
- **Chord/rest spelling.** Chord pitches are joined with `+` and `sorted`, so
  the same chord is always spelled the same way (`60+69`, never `69+60`),
  otherwise the alphabet quietly bloats.

Shapes: music became a **177 × 2 table** (177 time-slots × 2 hands) that
flattened into a **354-token line** over an **alphabet of 28 symbols**.

### Phase 2 — Decode + verify: tokens → MIDI, and proof

Ran Phase 1 backwards: paired 354 tokens into 177 slots, expanded to note
events, regenerated durations by the grid rule, wrote `out.mid`. Then
independently re-read the *original* and compared note-for-note.

**Result: PASS — all 199 (hand, slot, pitch) events matched.** The
representation is lossless (for onset + pitch + hand; durations are
grid-regenerated, a deliberate simplification). This is the most important
result: because encode/decode is proven lossless, any later wrong note is
provably the *model's* fault, not the encoding's. Confirmed twice — by this
check and by ear in MuseScore.

### Phase 3 — data.py: tokens → numbered quiz cards

A network can't multiply the string `"R:76"`, so every token got an ID number
0–27 (**encoding**; the numbered slots are *token IDs*). Two dicts: `tok2int`
(word→number, to feed in) and `int2tok` (number→word, to read out).

- **`sorted` vocab is mandatory.** Python shuffles a raw `set`'s order on every
  run (*hash randomization*). Without `sorted`, the token→number map would
  change between runs and silently break the saved model.
- **Sliding-window / next-token prediction.** The task is: given the last `K`
  tokens, predict the next one. A 9-wide frame (`K=8` question + `1` answer)
  slides across the 354-token line, cutting **346 cards** (`354 − 8`). This is
  the same idea underneath large language models.
- **`K = 8` is the memory span** — 8 tokens = 4 grid-steps (hands alternate).
  Small on purpose; it's literally the model's attention span, and the reason
  the replay could only stay faithful for a stretch.

Output: `X` shape **(346, 8)** (questions), `y` shape **(346,)** (answers), left
as integers, **no train/test split** — the goal is to memorize one piece, so
every card is training data.

### Phase 4 — model.py: the shape ladder

For a batch of `B` cards:

```
(B, 8) IDs → one-hot (B, 8, 28) → flatten (B, 224) → fc1+tanh (B, 256) → fc2 → logits (B, 28)
```

- **One-hot encoding.** A bare ID smuggles in a fake ordering (is token 27
  "bigger" than token 1? no — they're unordered categories, like postal codes).
  One-hot turns each ID into 27 zeros and a single 1, so all 28 tokens are
  equal-distance, no fake ranking. (https://en.wikipedia.org/wiki/One-hot)
- **One-hot ≡ embedding.** Multiplying a one-hot by a weight table just *picks
  out one row* — which is exactly what an embedding lookup does. So one-hot +
  a linear layer and `nn.Embedding` are the **same operation**. The exercise
  built the explicit version to make that visible.
- **`nn.Linear` = the model's brain.** A layer of tunable numbers (*weights*)
  plus a *bias*, computing `Wx + b`. These weights *are* the model's entire
  knowledge; training only adjusts them; `model.pt` is just these numbers.
  fc1 is a **224 × 256** matrix, fc2 is **256 × 28**.
  (https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
- **tanh = nonlinearity.** Stacking two linear layers collapses into one (a line
  of a line is a line). `tanh` bends the signal so the network can fit curvy
  patterns — that's what makes it a neural network and not plain arithmetic.
  (https://en.wikipedia.org/wiki/Activation_function)
- **Logits.** The final 28 numbers are raw scores, not probabilities. They
  become probabilities via *softmax*, which lives inside the loss function.
  (https://en.wikipedia.org/wiki/Softmax_function)
- **Batch dimension.** The leading `B` is just how many cards go through at once,
  in parallel. The code carries it untouched, so it never cares if it's 1 or 346.
- **Startup trick.** Shrinking fc2's weights ×0.1 and zeroing its bias makes a
  fresh model maximally unsure, so its loss is exactly `ln(28) ≈ 3.33` — a
  built-in wiring check.

Parameters: `224×256 + 256 = 57,600` (fc1) + `256×28 + 28 = 7,196` (fc2) =
**64,796 total**.

### Phase 5 — train.py: memorize

Fed all 346 cards at once (the whole piece is one batch). 2000 times over, ran
the same four-line rhythm:

```
optimizer.zero_grad()   # clear old gradients (PyTorch ADDS them by default — #1 bug if skipped)
loss = criterion(model(X), y)   # guess, then measure wrongness
loss.backward()         # compute which way each weight should move (backprop)
optimizer.step()        # move every weight one small step downhill
```

- **Cross-entropy loss** = the "how wrong am I" meter: low when the model piled
  probability on the correct token. Runs softmax internally.
  (https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- **Optimizer / learning rate.** The optimizer (`Adam`) adjusts the weights; the
  learning rate (`5e-3`) is the step size. Too big overshoots, too small crawls.
- **Gradient descent + backpropagation.** Treat loss as a hill; the gradient is
  the downhill arrow for each weight; step downhill, repeat.
  (https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)

Results:
- untrained loss **3.3338** ≈ predicted `ln(28) = 3.3322` (wiring check passed)
- loss fell to a floor of **~0.0475** — **not zero**, and that's correct: a few
  spots have the *same* 8-token question with *different* real answers (the
  theme repeats, then exits differently), which no weights can satisfy both
  ways. That leftover is the **loss floor**.
- memorized **336 / 346 cards (97.1%)** — the 10 misses are exactly those
  ambiguous repeats. It learned everything a fixed 8-token window *can* learn.

Output: `model.pt` (the 64,796 trained numbers).

### Phase 6 — generate.py: greedy replay

Seeded with the 8 true opening tokens, then predicted-and-fed-back out to 354
tokens. Output: `replayed_tokens.txt`.

**Result: faithful for the first 36 tokens (18 grid-slots) — the real opening —
then it hit the first ambiguous repeat, could only pick one branch, and dropped
into a loop.** Decoded to MIDI, it played the opening and then repeated, exactly
what I heard. (The `254/354` overall match is *not* fidelity — after diverging,
the loop just kept coinciding with the very repetitive original.)

---

## The two results that matter, and why they differ

This is the key idea of the whole project.

**Teacher forcing (training).** Every question uses the *real* 8 tokens from the
piece; we only ask for the next one. A mistake is isolated — the next card gets
fresh correct context. This is what the `336/346` measures: **did it memorize?**
Yes, everything memorizable.

**Free-running / autoregressive (generation).** The model eats its *own* output
— each prediction slides into the window and becomes part of the next question.
A single mistake **poisons everything after it**. This is what the `36 tokens`
measures: **can it reconstruct on its own?** Only briefly, then it drifts.

The gap between **97% memorized** and **36 tokens of faithful playback** is the
whole lesson. The jargon for a small error snowballing this way is **exposure
bias**: the model was only ever trained on perfect context, but at generation it
has to survive on its own imperfect output.

The `decode.py` mismatch on the replayed file (82 missing / 75 extra notes) is
not a bug — `decode.py` always compares against the *original*, so feeding it
the model's drifted version correctly measures how far free-running wandered.

---

## The honest limitation

A fixed-window memorizer has no sense of *where it is* in a piece — it only sees
the last 8 tokens. Für Elise is built from exact repeats (a collision check
found ~82% of the token stream is covered by verbatim-repeating passages, and a
handful of 8-token windows are followed by different next-tokens with no way to
tell them apart at any window size). So the loop isn't a failure to fix — it's
the correct, visible limit of *this* model on *this* kind of data. Text worked
in the Shakespeare version because English 8-grams are nearly unique; music
breaks it because it's repetition by design.

Fixes, if exact full replay were ever the goal (not needed here): give the model
a sense of position (a step index / positional signal), which makes every window
unique; or use teacher-forced replay to *audit* memorization without the drift.

---

## Transferable things to remember

- MIDI is **events, not audio**; timing is **delta** (relative), recover it by
  summing.
- **Quantization** (snapping to a grid) and **flattening** (2-D → 1-D ribbon)
  are the two moves that turn messy real-world data into something a sequence
  model can eat. That's the essence of feature engineering here.
- Always **`sorted`** your vocabulary — a shuffled token→ID map silently breaks
  a saved model.
- **One-hot for unordered categories**; and one-hot + a linear layer *is* an
  embedding.
- A model's knowledge lives entirely in its **weight matrices**; training only
  nudges those numbers.
- Every network needs a **nonlinearity** between linear layers or they collapse
  into one.
- The training loop is always **zero_grad → forward → backward → step**; skipping
  `zero_grad` is the classic bug.
- **Loss won't hit zero** when the data has the same input mapped to different
  outputs — that floor is information-theoretic, not a tuning failure.
- **Teacher forcing measures what a model learned; free-running reveals what it
  can do alone.** The gap between them is *exposure bias*, and it's why a model
  that scores 97% can still only reproduce a short opening before drifting.
