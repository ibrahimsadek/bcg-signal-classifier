#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
patch7_make_block_gate_only.py
===============================
Generates a BLOCK-GATING-ONLY variant of run_hr_pipeline.py.

WHY
---
patch6 established that 99.1% of the gate's downstream harm comes from
INTERVAL-level restriction, not from block rejection:

    interval-level effect : +0.718 bpm   (43/46 subjects worse)
    selection effect      : +0.007 bpm   (nothing)

Mechanism, measured on identical blocks: the gate strips 27.8% of seconds,
and because J-J intervals require CONSECUTIVE peaks, that removes 41.6% of
valid J-J intervals (21.2 -> 12.4 per block). Averaging over 42% fewer
intervals is noisier, so the block HR estimate degrades.

In run_hr_pipeline.py the column `is_bcg` currently does two different jobs:

  1. bcg_fraction = n_bcg_seconds / n_seconds      (line ~1394)
     -> drives BLOCK-level gating against beta.  THIS IS THE GATE'S PURPOSE.

  2. valid_mask over gate-accepted seconds only    (lines ~441, ~691)
     -> restricts where J-peaks may be detected INSIDE a retained block.
        THIS IS WHAT HURTS.

`--disable_gate` switches BOTH off at once, which is why the patch1 ablation
could not separate them. This patcher introduces a second column,
`is_bcg_peak`, used only for job 2, so the two can be controlled independently:

    (default)          is_bcg = prediction   is_bcg_peak = prediction
    --block_gate_only  is_bcg = prediction   is_bcg_peak = 1   <-- NEW
    --disable_gate     is_bcg = 1            is_bcg_peak = 1

The new mode keeps the quality gate deciding which blocks are usable, while
letting J-peak detection run over the whole retained block.

NON-DESTRUCTIVE
---------------
Your original run_hr_pipeline.py is NEVER modified. This writes a new file,
run_hr_pipeline_blockgate.py, and verifies every edit applied before saving.
If any anchor fails to match, it aborts and writes nothing.

USAGE (from the repo root):

    python patch7_make_block_gate_only.py
    patch7_run_block_gate_only.bat          (runs the new variant + compares)
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


# (description, anchor, replacement, expected_count)
EDITS = [
    (
        "peak masking (basic detector) uses is_bcg_peak",
        '''    bcg_rows = block[block["is_bcg"] == 1]
    if bcg_rows.empty:
        return np.asarray([], dtype=np.float64)

    block_start_sample = int(block["window_start_sample"].min())
    block_end_sample = int(block["window_end_sample"].max())
    if block_end_sample - block_start_sample < 3:
        return np.asarray([], dtype=np.float64)

    block_signal = filtered_signal[block_start_sample:block_end_sample]
    candidate_peaks_rel = detect_candidate_j_peaks(''',
        '''    _peak_col = "is_bcg_peak" if "is_bcg_peak" in block.columns else "is_bcg"
    bcg_rows = block[block[_peak_col] == 1]
    if bcg_rows.empty:
        return np.asarray([], dtype=np.float64)

    block_start_sample = int(block["window_start_sample"].min())
    block_end_sample = int(block["window_end_sample"].max())
    if block_end_sample - block_start_sample < 3:
        return np.asarray([], dtype=np.float64)

    block_signal = filtered_signal[block_start_sample:block_end_sample]
    candidate_peaks_rel = detect_candidate_j_peaks(''',
        1,
    ),
    (
        "peak masking (improved detector) uses is_bcg_peak",
        '''    bcg_rows = block[block["is_bcg"] == 1]
    if bcg_rows.empty:
        return np.asarray([], dtype=np.float64), {}''',
        '''    _peak_col = "is_bcg_peak" if "is_bcg_peak" in block.columns else "is_bcg"
    bcg_rows = block[block[_peak_col] == 1]
    if bcg_rows.empty:
        return np.asarray([], dtype=np.float64), {}''',
        1,
    ),
    (
        "build_second_level_table gains block_gate_only parameter",
        '''    prominence_coef: float = 0.05,
    disable_gate: bool = False,
) -> pd.DataFrame:''',
        '''    prominence_coef: float = 0.05,
    disable_gate: bool = False,
    block_gate_only: bool = False,
) -> pd.DataFrame:''',
        1,
    ),
    (
        "second-level table emits is_bcg_peak alongside is_bcg",
        '''                "is_bcg": 1 if disable_gate else int(row.pred_index == 1),''',
        '''                "is_bcg": 1 if disable_gate else int(row.pred_index == 1),
                # is_bcg drives bcg_fraction -> BLOCK-level gating.
                # is_bcg_peak drives J-peak masking INSIDE a retained block.
                # --block_gate_only keeps the former and disables the latter.
                "is_bcg_peak": 1 if (disable_gate or block_gate_only) else int(row.pred_index == 1),''',
        1,
    ),
    (
        "CLI flag --block_gate_only",
        '''    parser.add_argument("--disable_gate", action="store_true",''',
        '''    parser.add_argument("--block_gate_only", action="store_true", help="Ablation: keep BLOCK-level quality gating (bcg_fraction vs beta) but do NOT restrict J-peak detection to gate-accepted seconds within a retained block. Isolates block-level gating from interval-level restriction.")  # noqa: E501
    parser.add_argument("--disable_gate", action="store_true",''',
        1,
    ),
    (
        "PRIMARY diagnostic path uses is_bcg_peak (this is what feeds hr_bcg_bpm)",
        '''        valid_bcg_seconds = block[(block["is_bcg"] == 1) & (block["j_peak_sample_abs"].notna())]''',
        '''        _peak_col = "is_bcg_peak" if "is_bcg_peak" in block.columns else "is_bcg"
        valid_bcg_seconds = block[(block[_peak_col] == 1) & (block["j_peak_sample_abs"].notna())]''',
        1,
    ),
    (
        "Viterbi peak selection uses is_bcg_peak",
        '''    bcg_rows = block_seconds[(block_seconds["is_bcg"] == 1)].reset_index(drop=True)''',
        '''    _peak_col = "is_bcg_peak" if "is_bcg_peak" in block_seconds.columns else "is_bcg"
    bcg_rows = block_seconds[(block_seconds[_peak_col] == 1)].reset_index(drop=True)''',
        1,
    ),
]

# call-site edits: pass the flag through (both call sites)
CALLSITE_EDIT = (
    "thread block_gate_only through build_second_level_table call sites",
    "        disable_gate=disable_gate,\n",
    "        disable_gate=disable_gate,\n        block_gate_only=block_gate_only,\n",
)

# function-signature edits for the outer wrapper that holds disable_gate.
# NOTE: anchored on the adaptive_thresh_mult line above it so this cannot
# match build_second_level_table's own signature, which EDITS already handled.
SIG_EDITS = [
    (
        "process_recording wrapper signature",
        """    adaptive_thresh_mult: float = 1.2,
    disable_gate: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:""",
        """    adaptive_thresh_mult: float = 1.2,
    disable_gate: bool = False,
    block_gate_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:""",
    ),
]

ARGS_EDITS = [
    (
        "pass CLI flag into the pipeline (site 1)",
        "            disable_gate=args.disable_gate,",
        "            disable_gate=args.disable_gate,\n            block_gate_only=getattr(args, \"block_gate_only\", False),",  # noqa: E501
    ),
    (
        "pass CLI flag into the pipeline (site 2)",
        '        disable_gate=bool(getattr(args, "disable_gate", False)),',
        '        disable_gate=bool(getattr(args, "disable_gate", False)),\n        block_gate_only=bool(getattr(args, "block_gate_only", False)),',  # noqa: E501
    ),
]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate a block-gating-only variant of run_hr_pipeline.py")
    ap.add_argument("--source", type=Path, default=Path("run_hr_pipeline.py"))
    ap.add_argument("--dest", type=Path, default=Path("run_hr_pipeline_blockgate.py"))
    args = ap.parse_args()

    if not args.source.exists():
        raise SystemExit(f"{args.source} not found -- run from the repo root.")

    src = args.source.read_text(encoding="utf-8")
    out = src
    applied, failed = [], []

    for desc, anchor, repl, expected in EDITS:
        count = out.count(anchor)
        if count != expected:
            failed.append(f"{desc}: expected {expected} match(es), found {count}")
            continue
        out = out.replace(anchor, repl, expected)
        applied.append(desc)

    # signature edits -- only the FIRST occurrence (the pipeline wrapper)
    for desc, anchor, repl in SIG_EDITS:
        n = out.count(anchor)
        if n < 1:
            failed.append(f"{desc}: anchor not found")
            continue
        out = out.replace(anchor, repl, 1)
        applied.append(f"{desc} (first of {n} occurrence(s))")

    desc, anchor, repl = CALLSITE_EDIT
    n = out.count(anchor)
    if n < 1:
        failed.append(f"{desc}: anchor not found")
    else:
        out = out.replace(anchor, repl)
        applied.append(f"{desc} ({n} site(s))")

    for desc, anchor, repl in ARGS_EDITS:
        if anchor not in out:
            failed.append(f"{desc}: anchor not found")
            continue
        out = out.replace(anchor, repl, 1)
        applied.append(desc)

    print("=" * 74)
    print("BLOCK-GATE-ONLY PATCHER")
    print("=" * 74)
    print()
    for a in applied:
        print(f"  [ok]   {a}")
    for f in failed:
        print(f"  [FAIL] {f}")
    print()

    if failed:
        print("ABORTED -- no file written.")
        print()
        print("Your run_hr_pipeline.py differs from the version this patcher expects.")
        print("Nothing was modified. Send me the failing anchor(s) and I'll re-target.")
        raise SystemExit(1)

    # ---- completeness check -------------------------------------------------
    # Every `is_bcg` reference must now be one of:
    #   * the column definition itself (line ~759)
    #   * n_bcg_seconds -> bcg_fraction  (block gating: MUST stay on is_bcg)
    #   * the --disable_gate help text
    #   * a `_peak_col = "is_bcg_peak" if ... else "is_bcg"` fallback line
    # Anything else is a J-peak selection path that was missed, which would
    # silently leave interval-level restriction active. Fail loudly instead.
    allowed_markers = (
        '"is_bcg": 1 if disable_gate',
        '"is_bcg_peak": 1 if (disable_gate or block_gate_only)',
        'n_bcg_seconds = int(block["is_bcg"].sum())',
        'parser.add_argument("--disable_gate"',
        '_peak_col = "is_bcg_peak" if',
    )
    leftovers = []
    for i, line in enumerate(out.splitlines(), start=1):
        if "is_bcg" not in line:
            continue
        if line.lstrip().startswith("#"):          # comments are not code paths
            continue
        if any(mark in line for mark in allowed_markers):
            continue
        if "block[_peak_col]" in line or "block_seconds[_peak_col]" in line:
            continue
        leftovers.append(f"    line {i}: {line.strip()[:96]}")

    print("COMPLETENESS CHECK")
    print("-" * 74)
    if leftovers:
        print("  [FAIL] unpatched is_bcg reference(s) on a peak-selection path:")
        for item in leftovers:
            print(item)
        print()
        print("  These would leave interval-level restriction partly active, so the")
        print("  run would look like it worked while changing nothing that matters.")
        print("  ABORTED -- no file written. Send me these lines and I'll re-target.")
        raise SystemExit(1)
    print("  [ok]   no unpatched is_bcg peak-selection paths remain")
    print("  [ok]   bcg_fraction still uses is_bcg (block gating preserved)")
    print()

    # syntax check before writing
    try:
        compile(out, str(args.dest), "exec")
    except SyntaxError as exc:
        print(f"ABORTED -- generated code has a syntax error: {exc}")
        raise SystemExit(1)

    if args.dest.exists():
        backup = args.dest.with_suffix(".py.bak")
        shutil.copy2(args.dest, backup)
        print(f"  (existing {args.dest.name} backed up to {backup.name})")

    args.dest.write_text(out, encoding="utf-8")

    print(f"Wrote: {args.dest}")
    print(f"Your original {args.source.name} was NOT modified.")
    print()
    print("NEXT:")
    print("  patch7_run_block_gate_only.bat")
    print()
    print("  or manually:")
    print(f"    python {args.dest.name} --model_dir cv_output_nested_v2\\final_model \\")
    print("        --input_dir NewData_processed --output_dir gate_ablation\\blockonly0.50 \\")
    print("        --glob \"Sub*_aligned_data_ecg.txt\" --block_gate_only \\")
    print("        --bcg_fraction_threshold 0.50 --prominence_coef 0.12 \\")
    print("        --min_valid_jj_intervals 1")
    print("=" * 74)


if __name__ == "__main__":
    main()
