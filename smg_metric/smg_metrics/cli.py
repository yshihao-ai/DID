"""CLI entry point for smg_metrics evaluation.

Usage::

    # Single-file quality (13 metrics: 12 MusPy + OOK)
    smg-eval -m generated.mid

    # Single-file + structural + rhythmic
    smg-eval -m generated.mid -S -R

    # Pairwise note-level + bar-level + chord-level + rhythmic (11 metrics incl. CS)
    smg-eval -p generated.mid -r reference.mid

    # Distribution-level metrics (PD, DD, SC_sim, PCE_sim, GSC)
    smg-eval -p generated.mid -r reference.mid -d

    # Advanced metrics (KL, OA, CI, CTS, CR, ReconAcc)
    smg-eval -p generated.mid -r reference.mid -a

    # All metrics
    smg-eval -m gen.mid -p gen.mid -r ref.mid -d -a -S -R

    # Select a single metric
    smg-eval -m generated.mid --only pce
    smg-eval -p gen.mid -r ref.mid --only ca

    # Batch directory
    smg-eval --pred_dir ./pred/ --ref_dir ./ref/

    # JSON output
    smg-eval -m generated.mid --json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

__all__ = ["main"]

# ── Formatting helpers ────────────────────────────────────────────

def _fmt(val: Any) -> str:
    """Format a metric value for display."""
    if isinstance(val, float):
        if val != val:  # NaN
            return "NaN"
        return f"{val:.4f}"
    return str(val)

def _print_result(d: dict[str, Any], indent: int = 2, col_width: int = 24) -> None:
    """Pretty-print a result dict with aligned columns."""
    sp = " " * indent
    for k, v in d.items():
        print(f"{sp}{k:<{col_width}} = {_fmt(v)}")

def _json_safe(obj: Any) -> Any:
    """Convert NaN/inf floats to ``None`` for standards-compliant JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj

# ── Lazy imports (speed up CLI startup) ───────────────────────────

def _import_single():
    from smg_metrics.single import single_file, single_file_structural, single_file_rhythmic
    return single_file, single_file_structural, single_file_rhythmic

def _import_pair():
    from smg_metrics.pair import pair_eval, pair_eval_structural
    return pair_eval, pair_eval_structural

def _import_dist():
    from smg_metrics.distribution import compute_all
    return compute_all

def _import_adv():
    from smg_metrics.advanced import compute_all
    return compute_all

# ── Metric registry for --only flag ──────────────────────────────

_SINGLE_METRICS = {
    'pce', 'ebr', 'sc', 'pisr', 'polyphony', 'polyphony_rate',
    'pitch_range', 'n_pitches_used', 'n_pitch_classes_used', 'emr', 'pe', 'dpc',
    'ook',  # Out-of-Key percentage (FGG 2025)
}

_STRUCTURAL_SINGLE_METRICS = {'che', 'ngram_div'}
_RHYTHMIC_SINGLE_METRICS = {'mean_ioi', 'rhythmic_intensity', 'rhythmic_density', 'voice_number', 'gs'}

_PAIR_METRICS = {
    'note_f1', 'notei_f1', 'mel_f1', 'i_iou', 'ver',
    'sim_chr', 'sim_grv', 'ca', 'cs', 'onset_xor', 'note_overlap',
}

_STRUCTURAL_PAIR_METRICS = {'melody_match', 'tonal_dist'}
_DIST_METRICS = {'pd', 'dd', 'sc_sim', 'pce_sim', 'gsc'}
_ADV_METRICS = {
    'kl_duration', 'kl_ioi', 'kl_pitch',
    'oa_duration', 'oa_ioi', 'oa_pitch_range', 'oa_density',
    'ci_precision', 'ci_recall', 'ci_f1', 'cts',
    'cr_pred', 'cr_ref', 'recon_acc',
}

_ALL_METRICS = (
    _SINGLE_METRICS | _STRUCTURAL_SINGLE_METRICS | _RHYTHMIC_SINGLE_METRICS
    | _PAIR_METRICS | _STRUCTURAL_PAIR_METRICS | _DIST_METRICS | _ADV_METRICS
)

# ── Runners ───────────────────────────────────────────────────────

def _run_single(midi: str, root: int, mode: str, only: set[str] | None) -> dict[str, Any]:
    single_file, _, _ = _import_single()
    result = single_file(midi, root=root, mode=mode).to_dict()
    if only:
        result = {k: v for k, v in result.items() if k in only}
    return result

def _run_structural_single(midi: str, only: set[str] | None) -> dict[str, Any]:
    _, single_file_structural, _ = _import_single()
    result = single_file_structural(midi).to_dict()
    if only:
        result = {k: v for k, v in result.items() if k in only}
    return result

def _run_rhythmic_single(midi: str, only: set[str] | None) -> dict[str, Any]:
    _, _, single_file_rhythmic = _import_single()
    result = single_file_rhythmic(midi).to_dict()
    if only:
        result = {k: v for k, v in result.items() if k in only}
    return result

def _run_pair(pred: str, ref: str, only: set[str] | None) -> dict[str, Any]:
    pair_eval, _ = _import_pair()
    result = pair_eval(pred, ref).to_dict()
    if only:
        result = {k: v for k, v in result.items() if k in only}
    return result

def _run_structural_pair(pred: str, ref: str, only: set[str] | None) -> dict[str, Any]:
    _, pair_eval_structural = _import_pair()
    result = pair_eval_structural(pred, ref).to_dict()
    if only:
        result = {k: v for k, v in result.items() if k in only}
    return result

def _run_dist(pred: str, ref: str, only: set[str] | None) -> dict[str, Any]:
    compute_all = _import_dist()
    result = compute_all(pred, ref).to_dict()
    if only:
        result = {k: v for k, v in result.items() if k in only}
    return result

def _run_adv(pred: str, ref: str, only: set[str] | None) -> dict[str, Any]:
    compute_all = _import_adv()
    result = compute_all(pred, ref).to_dict()
    if only:
        result = {k: v for k, v in result.items() if k in only}
    return result

def _run_ook(midi: str, only: set[str] | None) -> dict[str, Any]:
    from smg_metrics.out_of_key import compute_ook
    if only and 'ook' not in only:
        return {}
    val = compute_ook(midi)
    return {'ook': val}

def _run_batch(pred_dir: str, ref_dir: str, root: int, mode: str) -> dict[str, float]:
    single_file, _, _ = _import_single()
    pair_eval, _ = _import_pair()

    preds = sorted(Path(pred_dir).glob("*.mid"))
    refs = sorted(Path(ref_dir).glob("*.mid"))
    ###
    if preds and refs:
        ref_map = {r.name: r for r in refs}
        missing = [p.name for p in preds if p.name not in ref_map]
        if missing:
            sys.exit(
                "Error: missing reference MIDI files for predictions: "
                + ", ".join(missing)
            )
        refs = [ref_map[p.name] for p in preds]
    ###
    if not preds:
        sys.exit(f"Error: no MIDI files in {pred_dir}")
    if not refs:
        sys.exit(f"Error: no MIDI files in {ref_dir}")
    n = min(len(preds), len(refs))
    print(f"\n{'='*60}")
    print(f"Batch evaluation: {n} pairs")
    print(f"{'='*60}\n")

    singles_p, pairs = [], []
    for i in range(n):
        p, r = str(preds[i]), str(refs[i])
        print(f"[{i+1}/{n}] {preds[i].name} vs {refs[i].name}")
        singles_p.append(single_file(p, root=root, mode=mode).to_dict())
        pair = pair_eval(p, r).to_dict()
        pairs.append(pair)
        for k, v in pair.items():
            print(f"    {k:<12} = {_fmt(v)}")
        print()

    summary: dict[str, float] = {}
    for key in singles_p[0]:
        vals = [s[key] for s in singles_p if isinstance(s[key], (int, float)) and s[key] == s[key]]
        if vals:
            summary[f"[pred] {key}"] = sum(vals) / len(vals)
    for key in pairs[0]:
        vals = [p[key] for p in pairs if isinstance(p[key], (int, float)) and p[key] == p[key]]
        if vals:
            summary[key] = sum(vals) / len(vals)

    print(f"\n{'='*60}")
    print(f"Batch summary ({n} pairs)")
    print(f"{'='*60}\n")
    for k, v in summary.items():
        print(f"  {k:<25} = {v:.4f}")
    return summary

# ── Main ──────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        prog="smg-eval",
        description="smg-metrics v5.2 — Objective evaluation metrics for Symbolic Music Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  smg-eval -m generated.mid\n"
            "  smg-eval -m gen.mid -S -R\n"
            "  smg-eval -p gen.mid -r ref.mid\n"
            "  smg-eval -p gen.mid -r ref.mid -d\n"
            "  smg-eval -p gen.mid -r ref.mid -a\n"
            "  smg-eval -m gen.mid -p gen.mid -r ref.mid -d -a -S -R\n"
            "  smg-eval -m gen.mid --only pce ebr gs\n"
            "  smg-eval -p gen.mid -r ref.mid --only ca note_f1\n"
            "  smg-eval --pred_dir ./pred/ --ref_dir ./ref/\n"
            "\n"
            "Metric categories:\n"
            "  Single-file (13):  PCE, EBR, SC, PISR, Poly, PR, Range, Np, Npc, EMR, PE, DPC, OOK\n"
            "  Structural (4):    CHE, Ngram, MelodyMatch, TonalDist\n"
            "  Rhythmic (5):      IOI, RI, RD, VN, GS\n"
            "  Pairwise (11):     F1, F1i, F1mel, I-IoU, VER, simChr, simGrv, CA, CS, XOR, NOvlp\n"
            "  Distribution (5):  PD, DD, SC_sim, PCE_sim, GSC\n"
            "  Advanced (14):     KL×3, OA×4, CI×3, CTS, CR×2, ReconAcc\n"
        ),
    )
    # File arguments
    p.add_argument("-m", "--music", metavar="PATH",
                   help="Single MIDI file for quality metrics (13 metrics: 12 MusPy + OOK)")
    p.add_argument("-p", "--pred", metavar="PATH",
                   help="Predicted / generated MIDI file")
    p.add_argument("-r", "--ref", metavar="PATH",
                   help="Reference / ground-truth MIDI file")
    p.add_argument("--pred_dir", metavar="DIR",
                   help="Directory of predicted MIDI files (batch mode)")
    p.add_argument("--ref_dir", metavar="DIR",
                   help="Directory of reference MIDI files (batch mode)")

    # Options
    p.add_argument("--root", type=int, default=0,
                   help="Root pitch class for PISR (0=C, ..., 11=B)")
    p.add_argument("--mode", type=str, default="major",
                   choices=["major", "minor"],
                   help="Scale mode for PISR (default: major)")

    # Category flags
    p.add_argument("-d", "--dist", action="store_true",
                   help="Include distribution-level metrics (PD, DD, SC_sim, PCE_sim, GSC)")
    p.add_argument("-a", "--advanced", action="store_true",
                   help="Include advanced metrics (KL, OA, CI, CTS, CR, ReconAcc)")
    p.add_argument("-S", "--structural", action="store_true",
                   help="Include structural metrics (CHE, Ngram, MelodyMatch, TonalDist)")
    p.add_argument("-R", "--rhythmic", action="store_true",
                   help="Include rhythmic metrics (IOI, RI, RD, VN, GS)")

    # Metric selection
    p.add_argument("--only", nargs="+", metavar="METRIC",
                   help="Only compute specified metrics (e.g. --only pce ca note_f1)")
    p.add_argument("--list-metrics", action="store_true",
                   help="List all available metric names and exit")

    # Output
    p.add_argument("--json", action="store_true",
                   help="Output results as JSON")
    p.add_argument("--time", action="store_true",
                   help="Print elapsed time")

    args = p.parse_args()

    # --list-metrics
    if args.list_metrics:
        print("Available metrics:")
        print(f"\n  Single-file (13):")
        for m in sorted(_SINGLE_METRICS):
            print(f"    {m}")
        print(f"\n  Structural single-file (2):")
        for m in sorted(_STRUCTURAL_SINGLE_METRICS):
            print(f"    {m}")
        print(f"\n  Rhythmic single-file (5):")
        for m in sorted(_RHYTHMIC_SINGLE_METRICS):
            print(f"    {m}")
        print(f"\n  Pairwise (11):")
        for m in sorted(_PAIR_METRICS):
            print(f"    {m}")
        print(f"\n  Structural pairwise (2):")
        for m in sorted(_STRUCTURAL_PAIR_METRICS):
            print(f"    {m}")
        print(f"\n  Distribution (5):")
        for m in sorted(_DIST_METRICS):
            print(f"    {m}")
        print(f"\n  Advanced (14):")
        for m in sorted(_ADV_METRICS):
            print(f"    {m}")
        return

    # Parse --only
    only: set[str] | None = None
    if args.only:
        only = set(args.only)
        unknown = only - _ALL_METRICS
        if unknown:
            sys.exit(f"Unknown metrics: {', '.join(sorted(unknown))}\n"
                     f"Use --list-metrics to see available metrics.")

    t0 = time.monotonic()
    result: dict[str, Any] | None = None

    # ── Single-file mode ──
    if args.music:
        needs_single = only is None or bool(only & _SINGLE_METRICS)
        needs_struct_s = (args.structural or (only and only & _STRUCTURAL_SINGLE_METRICS))
        needs_rhythm_s = (args.rhythmic or (only and only & _RHYTHMIC_SINGLE_METRICS))

        # Auto-detect needed categories if no --only and no category flags
        if only is None and not args.structural and not args.rhythmic:
            needs_struct_s = False
            needs_rhythm_s = False

        if needs_single:
            result = _run_single(args.music, args.root, args.mode, only)
            if not args.json:
                print(f"\n{'='*60}\nSingle-file: {args.music}\n{'='*60}\n")
                _print_result(result)
            
            # Add OOK if requested (part of single-file metrics)
            ook_result = _run_ook(args.music, only)
            if ook_result:
                if not args.json:
                    _print_result(ook_result)
                result.update(ook_result)

        if needs_struct_s:
            struct = _run_structural_single(args.music, only)
            if not args.json and struct:
                print("\n-- Structural (single) --")
                _print_result(struct)
            if result is None:
                result = {}
            result.update(struct)

        if needs_rhythm_s:
            rhythm = _run_rhythmic_single(args.music, only)
            if not args.json and rhythm:
                print("\n-- Rhythmic (single) --")
                _print_result(rhythm)
            if result is None:
                result = {}
            result.update(rhythm)

    # ── Pairwise mode ──
    if args.pred and args.ref:
        if result is None:
            result = {}

        needs_pair = only is None or bool(only & _PAIR_METRICS)
        needs_struct_p = (args.structural or (only and only & _STRUCTURAL_PAIR_METRICS))
        needs_rhythm_p = (args.rhythmic or (only and only & _RHYTHMIC_SINGLE_METRICS))
        needs_dist = (args.dist or (only and only & _DIST_METRICS))
        needs_adv = (args.advanced or (only and only & _ADV_METRICS))

        # Auto-detect: if no flags and no --only, run basic pair
        if only is None and not args.structural and not args.rhythmic and not args.dist and not args.advanced:
            needs_struct_p = False
            needs_rhythm_p = False
            needs_dist = False
            needs_adv = False

        if not args.music and needs_pair:
            if not args.json:
                print(f"\n{'='*60}\nPairwise: {args.pred} vs {args.ref}\n{'='*60}\n")

        if needs_pair:
            pair = _run_pair(args.pred, args.ref, only)
            if not args.json:
                _print_result(pair)
            result.update(pair)

        if needs_struct_p:
            # Single-file structural on pred
            struct_s = _run_structural_single(args.pred, only)
            if not args.json and struct_s:
                print("\n-- Structural (single) --")
                _print_result(struct_s)
            result.update(struct_s)

            # Pairwise structural
            struct_p = _run_structural_pair(args.pred, args.ref, only)
            if not args.json and struct_p:
                print("\n-- Structural (pair) --")
                _print_result(struct_p)
            result.update(struct_p)

        if needs_rhythm_p:
            # Single-file rhythmic on pred
            rhythm_s = _run_rhythmic_single(args.pred, only)
            if not args.json and rhythm_s:
                print("\n-- Rhythmic (single) --")
                _print_result(rhythm_s)
            result.update(rhythm_s)

        if needs_dist:
            dist = _run_dist(args.pred, args.ref, only)
            if not args.json and dist:
                print("\n-- Distribution --")
                _print_result(dist)
            result.update(dist)

        if needs_adv:
            adv = _run_adv(args.pred, args.ref, only)
            if not args.json and adv:
                print("\n-- Advanced --")
                _print_result(adv)
            result.update(adv)

    # ── Batch mode ──
    elif not args.music and args.pred_dir and args.ref_dir:
        result = _run_batch(args.pred_dir, args.ref_dir, args.root, args.mode)

    # ── No input ──
    elif result is None:
        p.print_help()
        return

    # ── JSON output ──
    if args.json and result is not None:
        print(json.dumps(_json_safe(result), indent=2, ensure_ascii=False))

    # ── Timing ──
    if args.time:
        elapsed = time.monotonic() - t0
        print(f"\nElapsed: {elapsed:.2f}s", file=sys.stderr)
