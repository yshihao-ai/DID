#!/usr/bin/env python3
import dataclasses
import json
import math
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Lightweight transformers-compatible stubs if transformers isn't installed.
try:
    import transformers  # noqa
except Exception:
    tr = types.ModuleType('transformers')
    outmod = types.ModuleType('transformers.modeling_outputs')
    class Qwen2Config:
        def __init__(self, **kw): self.__dict__.update(kw)
    class DummyOutput:
        def __init__(self, last_hidden_state): self.last_hidden_state = last_hidden_state
    class Qwen2Model(nn.Module):
        def __init__(self, config):
            super().__init__(); self.config=config; self.proj=nn.Linear(config.hidden_size, config.hidden_size)
        def forward(self, inputs_embeds=None, attention_mask=None, **kwargs):
            return DummyOutput(self.proj(inputs_embeds))
    class PreTrainedModel(nn.Module):
        config_class = Qwen2Config
        def __init__(self, config=None): super().__init__(); self.config=config
        def post_init(self): pass
    class CausalLMOutputWithPast:
        def __init__(self, loss=None, logits=None, **kw): self.loss=loss; self.logits=logits
    tr.Qwen2Config=Qwen2Config; tr.Qwen2Model=Qwen2Model; tr.PreTrainedModel=PreTrainedModel
    outmod.CausalLMOutputWithPast=CausalLMOutputWithPast
    sys.modules['transformers']=tr; sys.modules['transformers.modeling_outputs']=outmod

from component.arguments import DidMusicArguments
from musicmodel.did_model import DidForMusicLM
from transformers import Qwen2Config


def load_cfg():
    raw=json.loads((PROJECT/'train_args/did_Pop909_tiny_acc.json').read_text())
    names={f.name for f in dataclasses.fields(DidMusicArguments) if f.init}
    d={k:v for k,v in raw.items() if k in names}
    p=Path(d['vocab_file_path'])
    if not p.is_absolute(): d['vocab_file_path']=str((PROJECT/p).resolve())
    # Make smoke tiny while respecting head divisibility.
    d.update(time_decoder_layers=1,time_decoder_hidden_size=48,time_decoder_attention_heads=6,time_decoder_intermediate_size=96,
             note_decoder_layers=1,note_decoder_hidden_size=48,note_decoder_attention_heads=6,note_decoder_intermediate_size=96,
             bar_encoder_layers=1,bar_encoder_heads=6,pos_proj_dim=48,note_proj_dim=48,
             condition_encoder_layers=1,condition_encoder_heads=6,condition_cross_attention_heads=6)
    return DidMusicArguments(**d)


def main():
    cfg=load_cfg()
    tc=Qwen2Config(vocab_size=1, hidden_size=cfg.time_decoder_hidden_size, num_hidden_layers=cfg.time_decoder_layers,
                   num_attention_heads=cfg.time_decoder_attention_heads, intermediate_size=cfg.time_decoder_intermediate_size)
    nc=Qwen2Config(vocab_size=1, hidden_size=cfg.note_decoder_hidden_size, num_hidden_layers=cfg.note_decoder_layers,
                   num_attention_heads=cfg.note_decoder_attention_heads, intermediate_size=cfg.note_decoder_intermediate_size)
    model=DidForMusicLM(tc,nc,cfg)
    active=[x for x in dir(model) if 'plan' in x.lower()]
    if active: raise AssertionError(f'active PLAN attrs remain: {active}')
    print('PASS model initializes with zero PLAN attrs')

    # Numeric +ln2 behavior: construct logits and invoke the project's actual decode helper.
    helper=getattr(model, '_soft_bias_pitch_logits_with_active_chord_d51')
    vocab=cfg.pitch_vocab_size
    logits=torch.zeros(1,vocab)
    # Find a pitch token whose absolute PC is C (0). pitch midi = token + offset.
    c_token=next(i for i in range(1,vocab) if (i+cfg.pitch_offset)%12==0)
    before=logits.clone()
    # Avoid assumptions about chord/instrument token naming: use the model's own
    # parsed lookup tables and select any valid chord containing PC=C plus the
    # exact token id resolved as PIANO (Instrument_49).
    chord_candidates=torch.nonzero(model.chord_pc_membership[:,0], as_tuple=False).flatten()
    if chord_candidates.numel()==0:
        raise AssertionError('no parsed chord contains pitch class C')
    chord_id=int(chord_candidates[0].item())
    piano_candidates=torch.nonzero(model.d51_piano_instrument_mask, as_tuple=False).flatten()
    if piano_candidates.numel()!=1:
        raise AssertionError(f'expected exactly one PIANO token, got {piano_candidates.tolist()}')
    piano_id=int(piano_candidates[0].item())
    out=helper(logits, torch.tensor([chord_id]), torch.tensor([piano_id]))
    delta=float((out-before)[0,c_token])
    if abs(delta-math.log(2.0))>1e-5:
        raise AssertionError(f'chord-tone bias={delta}, expected {math.log(2.0)}')
    print(f'PASS numeric PIANO chord-tone bias = {delta:.8f} = ln2')
    print('ALL 2 NO-PLAN MODEL SMOKE TESTS PASSED')

if __name__=='__main__': main()
