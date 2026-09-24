#!/usr/bin/env python3
import ast
import dataclasses
import json
import math
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from component.arguments import DidMusicArguments, EvaluateArguments, NO_PLAN_BASELINE_VERSION
from component.collator import DidDataCollator

EXPECTED_FEATURES = ["type", "chord", "beat", "tempo", "instrument", "pitch", "duration", "velocity"]


def assert_true(x, msg):
    if not x:
        raise AssertionError(msg)


def load_music_cfg(project_root: Path, rel='train_args/did_Pop909_tiny_acc.json'):
    raw = json.loads((project_root / rel).read_text())
    names = {f.name for f in dataclasses.fields(DidMusicArguments) if f.init}
    music = {k: v for k, v in raw.items() if k in names}
    p = Path(music['vocab_file_path'])
    if not p.is_absolute():
        music['vocab_file_path'] = str((project_root / p).resolve())
    return DidMusicArguments(**music)


def token(cfg, type_name, chord=0, beat=0, tempo=0, inst=0, pitch=0, dur=0, vel=0):
    # Resolve type id from prepared vectors to avoid hard-coding vocab ids.
    type_map = {
        'CHORD': cfg.chord_note_id[0],
        'INST': cfg.inst_note_id[0],
        'BEAT': cfg.beat_note_id[0],
        'NOTE': cfg.note_note_id[0],
        'STO': cfg.sto_note_id[0],
        'EOS': cfg.eos_note_id[0],
    }
    return [type_map[type_name], chord, beat, tempo, inst, pitch, dur, vel]


def main():
    project_root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT
    assert_true(NO_PLAN_BASELINE_VERSION == '2026-08-18-global-previous-ln2', 'wrong restore version')
    print('PASS restore version')

    # Active dataclass schema must contain no PLAN knobs.
    did_fields = {f.name for f in dataclasses.fields(DidMusicArguments)}
    eval_fields = {f.name for f in dataclasses.fields(EvaluateArguments)}
    assert_true(not any('plan' in x.lower() for x in did_fields), f'PLAN fields in DidMusicArguments: {did_fields}')
    assert_true(not any('plan' in x.lower() for x in eval_fields), f'PLAN fields in EvaluateArguments: {eval_fields}')
    print('PASS zero PLAN config fields')

    cfg = load_music_cfg(project_root)
    assert_true(cfg.note_features == EXPECTED_FEATURES, f'expected DID8, got {cfg.note_features}')
    assert_true(getattr(cfg, 'type_vocab_size') == 7, f'expected 7 type tokens, got {cfg.type_vocab_size}')
    print('PASS DID8 / 7-type schema')

    # Source-level zero active PLAN contract for files restored by this patch.
    for rel in ['component/arguments.py','component/dataset.py','component/collator.py','musicmodel/did_model.py']:
        text = (ROOT / rel).read_text()
        bad = re.findall(r'(?i)(use_harmonic_plan|plan_bass|plan_set|plan_motion|plan_tension|plan_novelty|plan_code|PairwiseSonoritySetHead)', text)
        assert_true(not bad, f'legacy PLAN identifiers in {rel}: {bad[:8]}')
    print('PASS zero active PLAN implementation identifiers')

    # Global + immediate previous raw NOTE memory markers must remain.
    model = (ROOT / 'musicmodel/did_model.py').read_text()
    required = [
        'distant_bar_summary[:, 1:, :] = bar_summary[:, :-1, :]',
        'local_note_tokens = note_tokens_full[:, :-1, :, :]',
        'prev_note_proj = self._compute_note_embeddings(prev_notes.unsqueeze(1))[:, 0, :, :]',
        'prev_notes = cur_notes',
    ]
    for needle in required:
        assert_true(needle in model, f'missing Global+Previous contract: {needle}')
    print('PASS Global + previous-raw-NOTE source contract')

    # +ln2 must literally be finite log(2), not log(1) and not a hard mask.
    m = re.search(r'ln2\s*=\s*torch\.log\(\s*torch\.tensor\(\s*([0-9.]+)', model, flags=re.S)
    assert_true(m is not None, 'cannot locate ln2 decode bias')
    assert_true(abs(float(m.group(1)) - 2.0) < 1e-12, f'ln2 source is log({m.group(1)}), expected log(2)')
    print('PASS +ln2 source = log(2)')

    # Accompaniment condition must stop exactly before first target INST. No virtual PLAN boundary.
    coll = DidDataCollator(cfg, max_seq_length=8)
    chord_id = cfg.chord_n_n_id
    melody_inst = cfg.condition_instrument_prefix_ids[0]
    # pick a known non-condition instrument id from vocab (PIANO 49 is used in this project, but token id may differ).
    piano_inst = 49 if getattr(cfg, 'instrument_vocab_size') > 49 else max(1, getattr(cfg, 'instrument_vocab_size') - 1)
    b0 = cfg.beat_0_id
    bar = torch.tensor([
        token(cfg,'CHORD',chord=chord_id,beat=b0),
        token(cfg,'INST',inst=melody_inst),
        token(cfg,'BEAT',chord=chord_id,beat=b0,inst=melody_inst),
        token(cfg,'NOTE',inst=melody_inst,pitch=40,dur=1,vel=1),
        token(cfg,'INST',inst=piano_inst),
        token(cfg,'BEAT',chord=chord_id,beat=b0,inst=piano_inst),
        token(cfg,'NOTE',inst=piano_inst,pitch=45,dur=1,vel=1),
        token(cfg,'STO'),
    ], dtype=torch.long)
    source = bar.view(1,1,*bar.shape)
    mask = torch.ones(1,1,bar.shape[0],dtype=torch.long)
    time_mask = torch.ones(1,1,dtype=torch.long)
    cond_ids, cond_mask = coll._extract_accompaniment_condition(source, mask, time_mask)
    cond_len = int(cond_mask[0,0].sum())
    assert_true(cond_len == 4, f'expected CHORD+melody prefix length 4, got {cond_len}')
    assert_true(int(cond_ids[0,0,cond_len-1,0]) == cfg.note_note_id[0], 'condition should end on melody NOTE')
    print('PASS accompaniment [CHORD][MELODY] -> target boundary')

    print('ALL 7 NO-PLAN RESTORE CONTRACT TESTS PASSED')

if __name__ == '__main__':
    main()
