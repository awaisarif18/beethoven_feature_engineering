"""
encode.py  —  MIDI  ->  single-line token stream (tokens.txt)

The feature-engineering step. Turns two-handed piano MIDI into a 1-D line of
tokens that an MLP can eat. Prints the data at every stage so you can see the
2-D music being flattened into 1-D.

Token format, one column per sixteenth-note time-step:
    R:76 L:.      right hand plays pitch 76, left hand silent
    R:.  L:45     right hand silent, left hand plays 45
    R:60+69 L:.   right hand plays a chord (two notes at once)
'.' is a rest.  '+' joins the notes of a chord.

Run:  python encode.py furelise.mid
"""
import sys
import mido

GRID_DEN = 4          # 4 => sixteenth-note grid (a quarter note = 4 steps)
RH, LH = 0, 1         # track 0 = right hand, track 1 = left hand (this file)


def read_track_onsets(track, ticks_per_beat):
    """One track -> list of (grid_step, pitch), snapped to the grid.

    mido gives message .time as a DELTA from the previous message, so we keep a
    running total to recover the absolute tick of each note-on.
    """
    grid_ticks = ticks_per_beat // GRID_DEN        # e.g. 480 // 4 = 120
    abs_tick = 0
    raw = []                                       # (abs_tick, pitch) before snapping
    for msg in track:
        abs_tick += msg.time
        if msg.type == "note_on" and msg.velocity > 0:  
            raw.append((abs_tick, msg.note))
    # snap each absolute tick to the nearest whole grid step
    snapped = [(round(t / grid_ticks), pitch) for (t, pitch) in raw]
    return raw, snapped, grid_ticks


def build_columns(rh_snapped, lh_snapped):
    """Two lists of (step, pitch) -> list of (step, rh_token, lh_token)."""
    def group_by_step(snapped):
        d = {}
        for step, pitch in snapped:
            d.setdefault(step, []).append(pitch)
        return d

    rh = group_by_step(rh_snapped)
    lh = group_by_step(lh_snapped)

    def token(pitches):
        if not pitches:
            return "."
        return "+".join(str(p) for p in sorted(pitches))   # chord -> 60+69

    last_step = max(list(rh) + list(lh))
    columns = []
    for step in range(last_step + 1):
        columns.append((step, token(rh.get(step, [])), token(lh.get(step, []))))
    return columns


def main(path):
    mid = mido.MidiFile(path)
    print(f"file: {path}")
    print(f"ticks_per_beat (quarter note) = {mid.ticks_per_beat}")
    print(f"tracks = {len(mid.tracks)}\n")

    # ---- stage 1: raw note-ons per hand ----
    rh_raw, rh_snap, grid = read_track_onsets(mid.tracks[RH], mid.ticks_per_beat)
    lh_raw, lh_snap, _    = read_track_onsets(mid.tracks[LH], mid.ticks_per_beat)
    print(f"grid step = {grid} ticks  ({GRID_DEN} steps per quarter note)")
    print(f"right hand: {len(rh_raw)} notes | left hand: {len(lh_raw)} notes\n")

    print("STAGE 1 — raw note-ons (absolute tick, pitch), first 8 of right hand:")
    for t, p in rh_raw[:8]:
        print(f"    tick {t:5d}   pitch {p}")

    # ---- stage 2: after snapping to the grid ----
    print("\nSTAGE 2 — same notes after snapping to grid steps:")
    for (t, p), (s, _) in list(zip(rh_raw, rh_snap))[:8]:
        print(f"    tick {t:5d}  ->  step {s:3d}   pitch {p}")

    # ---- stage 3: grid columns (both hands lined up in time) ----
    columns = build_columns(rh_snap, lh_snap)
    print(f"\nSTAGE 3 — {len(columns)} grid columns (both hands per step), first 8:")
    for step, r, l in columns[:8]:
        print(f"    step {step:3d}   R:{r:<6} L:{l}")

    # ---- stage 4: the single line ----
    tokens = []
    for _, r, l in columns:
        tokens.append(f"R:{r}")
        tokens.append(f"L:{l}")
    line = " ".join(tokens)

    print(f"\nSTAGE 4 — flattened to one line of {len(tokens)} tokens.")
    print("    first 16 tokens:", " ".join(tokens[:16]))

    vocab = sorted(set(tokens))
    print(f"\nvocabulary size = {len(vocab)} distinct tokens:")
    print("   ", vocab)

    with open("tokens.txt", "w", encoding="utf-8") as f:
        f.write(line)
    print(f"\nwrote tokens.txt  ({len(line)} characters, {len(tokens)} tokens)")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "furelise.mid"
    main(path)