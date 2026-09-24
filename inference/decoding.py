import os, sys
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
from collections import defaultdict

from music21 import converter
import muspy
import miditoolkit
from miditoolkit.midi.containers import Marker, Instrument, TempoChange, Note, TimeSignature

from .midi2audio import FluidSynth
from data_representation.constants import PROGRAM_INSTRUMENT_MAP

class MuteWarn:
  def __enter__(self):
    self._init_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")

  def __exit__(self, exc_type, exc_val, exc_tb):
    sys.stdout.close()
    sys.stdout = self._init_stdout

def save_score_image_from_midi(midi_fn, file_name):
  assert isinstance(midi_fn, str)
  with MuteWarn():
    convert = converter.parse(midi_fn)
    convert.write('musicxml.png', fp=file_name)

def save_pianoroll_image_from_midi(midi_fn, file_name):
    assert isinstance(midi_fn, str)
    
    try:
        # 尝试生成 pianoroll 可视化
        midi_obj_muspy = muspy.read_midi(midi_fn)
        
        # MusPy 在部分 MIDI 上可能触发 IndexError
        midi_obj_muspy.show_pianoroll(track_label='program', preset='frame')
        
        plt.gcf().set_size_inches(20, 10)
        plt.savefig(file_name)
        plt.close()  # 正常关闭
        
    except IndexError:
        # 单独处理 MusPy 索引越界，保证主流程不中断
        logger.warning(f"[Visualization Skipped] MusPy IndexError for file: {midi_fn}. MIDI is saved, but PNG is skipped.")
        plt.close()  # 发生异常也要关闭画板，避免内存泄漏
        
    except Exception as e:
        # 兜底处理其他可视化错误
        logger.warning(f"[Visualization Skipped] Unknown error for file: {midi_fn}. Reason: {e}")
        plt.close()
        
def save_wav_from_midi(midi_fn, file_name, qpm=80):
  assert isinstance(midi_fn, str)
  return
  with MuteWarn():
    music = muspy.read_midi(midi_fn)
    music.tempos[0].qpm = qpm
    music.write_audio(file_name, rate=44100, gain=3)

# def save_wav_from_midi_fluidsynth(midi_fn, file_name, gain=3):
#   assert isinstance(midi_fn, str)
#   fs = FluidSynth(gain=gain)
#   fs.midi_to_audio(midi_fn, file_name)

import subprocess

def save_wav_from_midi_fluidsynth(midi_fn, file_name, gain=3):
    return
    assert isinstance(midi_fn, str)
    
    sf2_path = "/home/xxx/.fluidsynth/default_sound_font.sf2" 
    
    if not os.path.exists(sf2_path):
        logger.error(f"SoundFont file not found at: {sf2_path}")
        return

    cmd = [
        "fluidsynth",
        "-ni",
        "-g", str(gain),
        sf2_path,
        midi_fn,
        "-F", file_name,
        "-r", "44100"
    ]

    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to convert MIDI to WAV. Command: {' '.join(cmd)}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during synthesis: {e}")

class MidiDecoder4REMI:
  def __init__(
      self, 
      vocab, 
      in_beat_resolution, 
      dataset_name
  ):
    self.vocab = vocab
    self.in_beat_resolution = in_beat_resolution
    self.dataset_name = dataset_name
    if dataset_name == 'SymphonyMIDI':
      self.gain = 0.7
    elif dataset_name == 'SOD' or dataset_name == 'LakhClean':
      self.gain = 1.1
    elif dataset_name == 'Pop1k7' or dataset_name == 'Pop909':
      self.gain = 2.5
    else:
      self.gain = 1.5

  def _normalize_output_program(self, program_id):
      if self.dataset_name == "Pop909":
          return 0
      return int(program_id)


  def _output_track_name(self, source_program):
      source_program = int(source_program)
      if self.dataset_name == "Pop909":
          role_names = {0: "MELODY", 24: "BRIDGE", 49: "PIANO"}
          return role_names[source_program]
      return PROGRAM_INSTRUMENT_MAP[source_program]

  def _append_tracks_to_midi(self, midi_obj, instr_notes_dict):
    for source_program in sorted(instr_notes_dict.keys()):
        notes = instr_notes_dict[source_program]
        if len(notes) == 0: continue
        notes.sort(key=lambda n: (n.start, n.pitch, n.end, n.velocity))
        output_program = self._normalize_output_program(source_program)
        track_name = self._output_track_name(source_program)
        is_drum = (int(source_program) == 114 and self.dataset_name != "Pop909")
        instrument_track = Instrument(program=output_program, is_drum=is_drum, name=track_name)
        instrument_track.notes = notes
        midi_obj.instruments.append(instrument_track)


  def __call__(self, generated_output, output_path=None):
    '''
    generated_output: list of tensor, the tensor
    '''
    idx2event = self.vocab.idx2event
    if generated_output.dim() == 2:
      generated_output = generated_output.squeeze(0)
    events = [idx2event[token.item()] for token in generated_output]

    midi_obj = miditoolkit.midi.parser.MidiFile()
    if 'tempo' not in idx2event.keys():
      default_tempo = 95
      midi_obj.tempo_changes.append(
        TempoChange(tempo=default_tempo, time=0))
    default_ticks_per_beat = 480
    default_in_beat_ticks  =  480 // self.in_beat_resolution
    cur_pos = 0
    bar_pos = 0
    cur_bar_resol = 0
    beat_pos = 0
    cur_instr = 0 if not self.dataset_name == 'BachChorale' else 53
    instr_notes_dict = defaultdict(list)
    for i in range(len(events)-2):
      cur_event = events[i]
      # print(cur_event)
      name = cur_event.split('_')[0]
      attr = cur_event.split('_')
      if name == 'Bar':
        bar_pos += cur_bar_resol
        if 'time' in cur_event:
          cur_num, cur_denom = attr[-1].split('/')
          new_bar_resol = int(default_ticks_per_beat * int(cur_num) * (4 / int(cur_denom)))
          cur_bar_resol = new_bar_resol
          midi_obj.time_signature_changes.append(
            TimeSignature(numerator=int(cur_num), denominator=int(cur_denom), time=bar_pos))
      elif name == 'Beat':
        beat_pos = int(attr[1])
        cur_pos = bar_pos + beat_pos * default_in_beat_ticks
      elif name == 'Chord':
        chord_text = attr[1] + '_' + attr[2]
        midi_obj.markers.append(Marker(text=chord_text, time=cur_pos))
      elif name == 'Tempo':
        midi_obj.tempo_changes.append(
            TempoChange(tempo=int(attr[1]), time=cur_pos))
      elif name == 'Instrument':
        cur_instr = int(attr[1])

      else:
        if len(self.vocab.feature_list) == 7 or len(self.vocab.feature_list) == 8:
          if 'Note_Pitch' in events[i] and \
          'Note_Duration' in events[i+1] and \
          'Note_Velocity' in events[i+2]:
            pitch = int(events[i].split('_')[-1])
            duration = int(events[i+1].split('_')[-1])
            duration = duration * default_in_beat_ticks
            end = cur_pos + duration 
            velocity = int(events[i+2].split('_')[-1])
            instr_notes_dict[cur_instr].append(
              Note(
                pitch=pitch, 
                start=cur_pos, 
                end=end, 
                velocity=velocity))
        elif len(self.vocab.feature_list) == 4 or len(self.vocab.feature_list) == 5:
          if 'Note_Pitch' in events[i] and \
          'Note_Duration' in events[i+1]:
            pitch = int(events[i].split('_')[-1])
            duration = int(events[i+1].split('_')[-1])
            duration = duration * default_in_beat_ticks
            end = cur_pos + duration 
            velocity = 90
            instr_notes_dict[cur_instr].append(
              Note(
                pitch=pitch, 
                start=cur_pos, 
                end=end, 
                velocity=velocity))
          
    # save midi  
    self._append_tracks_to_midi(midi_obj, instr_notes_dict)

    if isinstance(output_path, str) or isinstance(output_path, Path):
      output_path = str(output_path)
      midi_obj.dump(output_path)
      save_pianoroll_image_from_midi(output_path, output_path.replace('.mid', '.png'))
      save_score_image_from_midi(output_path, output_path.replace('.mid', '_score.png'))
      save_wav_from_midi_fluidsynth(output_path, output_path.replace('.mid', '.wav'), gain=self.gain)
    return midi_obj

class MidiDecoder4CP(MidiDecoder4REMI):
  def __init__(self, vocab, in_beat_resolution, dataset_name):
    super().__init__(vocab, in_beat_resolution, dataset_name)
  
  def _update_chord_tempo(self, midi_obj, cur_pos, token_with_7infos, feature2idx):
    if len(feature2idx) == 7 or len(feature2idx) == 8:
      # chord
      if token_with_7infos[feature2idx['chord']] != 'CONTI' and token_with_7infos[feature2idx['chord']] != 0:
        midi_obj.markers.append(
          Marker(text=str(token_with_7infos[feature2idx['chord']]), time=cur_pos))
      # tempo
      if token_with_7infos[feature2idx['tempo']] != 'CONTI' and token_with_7infos[feature2idx['tempo']] != 0 and token_with_7infos[feature2idx['tempo']] != "Tempo_N_N":
        tempo = int(token_with_7infos[feature2idx['tempo']].split('_')[-1])
        midi_obj.tempo_changes.append(
          TempoChange(tempo=tempo, time=cur_pos))
      return midi_obj
    elif len(feature2idx) == 4 or len(feature2idx) == 5:
      return midi_obj
    
  def __call__(self, generated_output, output_path=None):
    '''
    generated_output: tensor, batch x seq_len x num_types
    num_types includes: type, tempo, chord,'beat, pitch, duration, velocity
    '''
    idx2event = self.vocab.idx2event
    feature_keys = self.vocab.feature_list
    feature2idx = {key: idx for idx, key in enumerate(feature_keys)}

    midi_obj = miditoolkit.midi.parser.MidiFile()
    if len(feature2idx) == 4 or len(feature2idx) == 5:
      default_tempo = 95
      midi_obj.tempo_changes.append(
        TempoChange(tempo=default_tempo, time=0))
    default_ticks_per_beat = 480
    default_in_beat_ticks  =  480 // self.in_beat_resolution
    cur_pos = 0
    bar_pos = 0
    cur_bar_resol = 0
    beat_pos = 0
    instr_notes_dict = defaultdict(list)
    generated_output = generated_output.squeeze(0)
    for i in range(len(generated_output)):
      token_with_7infos = []
      for kidx, key in enumerate(feature_keys):
        token_with_7infos.append(idx2event[key][generated_output[i][kidx].item()])
      # type token
      if 'time_signature' in token_with_7infos[feature2idx['type']]:
        cur_num, cur_denom = token_with_7infos[feature2idx['type']].split('_')[-1].split('/')
        bar_pos += cur_bar_resol
        new_bar_resol = int(default_ticks_per_beat * int(cur_num) * (4 / int(cur_denom)))
        cur_bar_resol = new_bar_resol
        midi_obj.time_signature_changes.append(
          TimeSignature(numerator=int(cur_num), denominator=int(cur_denom), time=bar_pos))
      elif token_with_7infos[feature2idx['type']] == 'Metrical':
        if 'time_signature' in token_with_7infos[feature2idx['beat']]:
          cur_num, cur_denom = token_with_7infos[feature2idx['beat']].split('_')[-1].split('/')
          bar_pos += cur_bar_resol
          new_bar_resol = int(default_ticks_per_beat * int(cur_num) * (4 / int(cur_denom)))
          cur_bar_resol = new_bar_resol
          midi_obj.time_signature_changes.append(
            TimeSignature(numerator=int(cur_num), denominator=int(cur_denom), time=bar_pos))
        elif token_with_7infos[feature2idx['beat']] == 'Bar':
          bar_pos += cur_bar_resol
        elif 'Beat' in str(token_with_7infos[feature2idx['beat']]):
          beat_pos = int(token_with_7infos[feature2idx['beat']].split('_')[1])
          cur_pos = bar_pos + beat_pos * default_in_beat_ticks # ticks
          # chord and tempo
          midi_obj = self._update_chord_tempo(midi_obj, cur_pos, token_with_7infos, feature2idx)
      elif token_with_7infos[feature2idx['type']] == 'Note':
        # instrument token
        if len(feature2idx) == 8 or len(feature2idx) == 5:
          if token_with_7infos[feature2idx['instrument']] != 0 and token_with_7infos[feature2idx['instrument']] != 'CONTI':
            cur_instr = int(token_with_7infos[feature2idx['instrument']].split('_')[-1])
        else:
          cur_instr = 0 if not self.dataset_name == 'BachChorale' else 53
        try:
          pitch = token_with_7infos[feature2idx['pitch']].split('_')[-1]
          duration = token_with_7infos[feature2idx['duration']].split('_')[-1]
          duration = int(duration) * default_in_beat_ticks
          if len(feature2idx) == 7 or len(feature2idx) == 8:
            velocity = token_with_7infos[feature2idx['velocity']].split('_')[-1]
          else:
            velocity = 80
          end = cur_pos + duration
          instr_notes_dict[cur_instr].append(
            Note(
              pitch=int(pitch), 
              start=cur_pos, 
              end=end, 
              velocity=int(velocity))
            )
        except:
          continue
      else: # when new bar started without beat
        continue

    # save midi
    self._append_tracks_to_midi(midi_obj, instr_notes_dict)

    if isinstance(output_path, str) or isinstance(output_path, Path):
      output_path = str(output_path)
      midi_obj.dump(output_path)
      save_pianoroll_image_from_midi(output_path, output_path.replace('.mid', '.png'))
      save_wav_from_midi_fluidsynth(output_path, output_path.replace('.mid', '.wav'), gain=self.gain)
    return midi_obj
  
class MidiDecoder4NB(MidiDecoder4REMI):
  def __init__(self, vocab, in_beat_resolution, dataset_name):
    super().__init__(vocab, in_beat_resolution, dataset_name)
  
  def _update_additional_info(self, midi_obj, cur_pos, token_with_7infos, feature2idx):
    if len(feature2idx) == 7 or len(feature2idx) == 8:
      # chord
      if token_with_7infos[feature2idx['chord']] != 'CONTI' and token_with_7infos[feature2idx['chord']] != 0 and token_with_7infos[feature2idx['chord']] != 'Chord_N_N':
        midi_obj.markers.append(
          Marker(text=str(token_with_7infos[feature2idx['chord']]), time=cur_pos))
      # tempo
      if token_with_7infos[feature2idx['tempo']] != 'CONTI' and token_with_7infos[feature2idx['tempo']] != 0 and token_with_7infos[feature2idx['tempo']] != "Tempo_N_N":
        tempo = int(token_with_7infos[feature2idx['tempo']].split('_')[-1])
        midi_obj.tempo_changes.append(
          TempoChange(tempo=tempo, time=cur_pos))
      return midi_obj
    elif len(feature2idx) == 4 or len(feature2idx) == 5:
      return midi_obj
    
  def __call__(self, generated_output, output_path=None):
    '''
    generated_output: tensor, batch x seq_len x num_types
    num_types includes: type, beat, chord, tempo, intrument, pitch, duration, velocity
    '''
    idx2event = self.vocab.idx2event
    feature_keys = self.vocab.feature_list
    feature2idx = {key: idx for idx, key in enumerate(feature_keys)}

    midi_obj = miditoolkit.midi.parser.MidiFile()
    if len(feature2idx) == 4 or len(feature2idx) == 5:
      default_tempo = 95
      midi_obj.tempo_changes.append(
        TempoChange(tempo=default_tempo, time=0))
    default_ticks_per_beat = 480
    default_in_beat_ticks = 480 // self.in_beat_resolution
    cur_pos = 0
    bar_pos = 0
    cur_bar_resol = 0
    beat_pos = 0
    instr_notes_dict = defaultdict(list)
    generated_output = generated_output.squeeze(0)
    for i in range(len(generated_output)):
      token_with_7infos = []
      for kidx, key in enumerate(feature_keys):
        token_with_7infos.append(idx2event[key][generated_output[i][kidx].item()])
      # type token
      if token_with_7infos[feature2idx['type']] == 'Empty_Bar' or token_with_7infos[feature2idx['type']] == 'SNN':
        bar_pos += cur_bar_resol
      elif 'NNN' in token_with_7infos[feature2idx['type']]:
        cur_num, cur_denom = token_with_7infos[feature2idx['type']].split('_')[-1].split('/')
        bar_pos += cur_bar_resol
        new_bar_resol = int(default_ticks_per_beat * int(cur_num) * (4 / int(cur_denom)))
        cur_bar_resol = new_bar_resol
        midi_obj.time_signature_changes.append(
          TimeSignature(numerator=int(cur_num), denominator=int(cur_denom), time=bar_pos))
      # instrument token
      if len(feature2idx) == 8 or len(feature2idx) == 5:
        if token_with_7infos[feature2idx['instrument']] != 0 and token_with_7infos[feature2idx['instrument']] != 'CONTI':
          cur_instr = int(token_with_7infos[feature2idx['instrument']].split('_')[-1])
      else:
        cur_instr = 0 if not self.dataset_name == 'BachChorale' else 53
      if 'Beat' in str(token_with_7infos[feature2idx['beat']]) or 'CONTI' in str(token_with_7infos[feature2idx['beat']]):
        if 'Beat' in str(token_with_7infos[feature2idx['beat']]): # when beat is not CONTI beat is updated
          beat_pos = int(token_with_7infos[feature2idx['beat']].split('_')[1])
          cur_pos = bar_pos + beat_pos * default_in_beat_ticks # ticks
        # update chord and tempo
        midi_obj = self._update_additional_info(midi_obj, cur_pos, token_with_7infos, feature2idx)  
        # note
        try:
          pitch = token_with_7infos[feature2idx['pitch']].split('_')[-1]
          duration = token_with_7infos[feature2idx['duration']].split('_')[-1] # duration between 1~192
          duration = int(duration) * default_in_beat_ticks
          if len(feature2idx) == 7 or len(feature2idx) == 8:
            velocity = token_with_7infos[feature2idx['velocity']].split('_')[-1]
          else:
            velocity = 90
          end = cur_pos + duration
          instr_notes_dict[cur_instr].append(
            Note(
              pitch=int(pitch), 
              start=cur_pos, 
              end=end, 
              velocity=int(velocity))
            )
        except:
          continue
      else: # when new bar started without beat
        continue
    
    # save midi
    self._append_tracks_to_midi(midi_obj, instr_notes_dict)

    if isinstance(output_path, str) or isinstance(output_path, Path):
      output_path = str(output_path)
      midi_obj.dump(output_path)
      save_pianoroll_image_from_midi(output_path, output_path.replace('.mid', '.png'))
      save_wav_from_midi_fluidsynth(output_path, output_path.replace('.mid', '.wav'), gain=self.gain)
    return midi_obj
  


class MidiDecoder4DID(MidiDecoder4REMI):
  def __init__(self, vocab, in_beat_resolution, dataset_name, pos_feature_keys=None, note_feature_keys=None):
    super().__init__(vocab, in_beat_resolution, dataset_name)
    if pos_feature_keys is None or note_feature_keys is None:
      raise ValueError("MidiDecoder4DID requires explicit pos_feature_keys and note_feature_keys.")
    self._runtime_pos_feature_keys = list(pos_feature_keys)
    self._runtime_note_feature_keys = list(note_feature_keys)

  @staticmethod
  def _safe_event_lookup(idx2event, key, token_id):
    if key not in idx2event:
      return 0
    table = idx2event[key]
    tid = int(token_id)
    if tid in table:
      return table[tid]
    # Some vocab json may keep int ids as string keys.
    str_tid = str(tid)
    if str_tid in table:
      return table[str_tid]
    return 0

  def __call__(self, pos_input_ids, generated_output, output_path=None, pos_attention_mask=None, include_instruments=None, exclude_instruments=None):
    """
    pos_input_ids   : [T, P]
    generated_output: [T, N, F]
    """
    if (include_instruments is not None and exclude_instruments is not None):
      raise ValueError("include_instruments and exclude_instruments cannot be used together.")
    
    include_instruments = (None if include_instruments is None else {int(v) for v in include_instruments})
    exclude_instruments = (set() if exclude_instruments is None else {int(v) for v in exclude_instruments})

    idx2event = self.vocab.idx2event
    feature_keys = list(self.vocab.feature_list)

    pos_feature_keys = [k for k in self._runtime_pos_feature_keys if k in feature_keys]
    note_feature_keys = [k for k in self._runtime_note_feature_keys if k in feature_keys]
    posfeature2idx = {k: i for i, k in enumerate(pos_feature_keys)}
    notefeature2idx = {k: i for i, k in enumerate(note_feature_keys)}

    midi_obj = miditoolkit.midi.parser.MidiFile()
    if 'tempo' not in pos_feature_keys:
      midi_obj.tempo_changes.append(TempoChange(tempo=95, time=0))

    default_ticks_per_beat = 480
    default_in_beat_ticks = 480 // self.in_beat_resolution

    generated_output = generated_output.squeeze(0)
    if hasattr(pos_input_ids, 'dim') and pos_input_ids.dim() == 3:
      pos_input_ids = pos_input_ids.squeeze(0)
    if pos_attention_mask is not None and hasattr(pos_attention_mask, 'dim') and pos_attention_mask.dim() == 2:
      pos_attention_mask = pos_attention_mask.squeeze(0)

    instr_notes_dict = defaultdict(list)

    bar_pos = 0
    cur_bar_resol = int(default_ticks_per_beat * 4)
    prev_bar_resol = cur_bar_resol
    prev_bar_id = None
    prev_time_sig = None
    song_end = False
    last_emitted_chord = None
    
    if pos_attention_mask is not None:
      if hasattr(pos_attention_mask, 'tolist'):
        valid_steps = int(sum(int(v) for v in pos_attention_mask.tolist()))
      else:
        valid_steps = int(sum(int(v) for v in pos_attention_mask))
    else:
      valid_steps = pos_input_ids.shape[0]
    t_total = min(valid_steps, pos_input_ids.shape[0], generated_output.shape[0])
    for t in range(t_total):
      if song_end:
        break
      saw_chord_prefix = False
      # decode pos tokens
      pos_tokens = []
      for kidx, key in enumerate(pos_feature_keys):
        pos_tokens.append(self._safe_event_lookup(idx2event, key, int(pos_input_ids[t][kidx].item())))

      bar_text = pos_tokens[posfeature2idx['bar']] if 'bar' in posfeature2idx else f'Bar_{t+1}'
      time_sig_text = pos_tokens[posfeature2idx['time_signature']] if 'time_signature' in posfeature2idx else 'time_signature_4/4'
      if 'time_signature' in posfeature2idx:
        num, denom = time_sig_text.split('_')[-1].split('/')
        cur_bar_resol = int(default_ticks_per_beat * int(num) * (4 / int(denom)))

      if bar_text != prev_bar_id:
        if prev_bar_id is not None:
          bar_pos += prev_bar_resol
        prev_bar_id = bar_text
        prev_bar_resol = cur_bar_resol

      if time_sig_text != prev_time_sig:
        if 'time_signature' in posfeature2idx:
          num, denom = time_sig_text.split('_')[-1].split('/')
          midi_obj.time_signature_changes.append(TimeSignature(numerator=int(num), denominator=int(denom), time=bar_pos))
        prev_time_sig = time_sig_text

      if 'tempo' in posfeature2idx:
        tempo_text = pos_tokens[posfeature2idx['tempo']]
        if tempo_text != 0 and tempo_text != 'Tempo_N_N':
          try:
            tempo_val = int(str(tempo_text).split('_')[-1])
            midi_obj.tempo_changes.append(TempoChange(tempo=tempo_val, time=bar_pos))
          except Exception:
            pass

      # decode notes in bar
      notes_t = generated_output[t]
      emitted_chord_on_beat = {}
      cur_instr = 0 if self.dataset_name != 'BachChorale' else 53
      cur_beat_idx = 0
      for n_idx in range(notes_t.shape[0]):
        token = notes_t[n_idx]
        note_tokens = []
        for kidx, key in enumerate(note_feature_keys):
          note_tokens.append(self._safe_event_lookup(idx2event, key, int(token[kidx].item())))

        note_type = note_tokens[notefeature2idx['type']]
        if note_type == 'SOS':
          continue
        if note_type == 'CHORD':
          saw_chord_prefix = True
          chord_beat_idx = 0
          if 'beat' in notefeature2idx:
              beat_text = note_tokens[notefeature2idx['beat']]
              if (isinstance(beat_text, str) and beat_text.startswith('Beat_')):
                  chord_beat_idx = int(beat_text.split('_')[-1])

          chord_name = note_tokens[notefeature2idx['chord']]

          if (isinstance(chord_name, str) and chord_name.startswith('Chord_')):
              if chord_name == 'Chord_N_N':
                  last_emitted_chord = chord_name

              elif chord_name != last_emitted_chord:
                  chord_pos = (
                      bar_pos
                      + chord_beat_idx
                      * default_in_beat_ticks
                  )
                  midi_obj.markers.append(
                      Marker(
                          text=str(chord_name),
                          time=chord_pos,
                      )
                  )
                  last_emitted_chord = chord_name
          continue

        if note_type == 'INST':
          # Unified protocol: instrument for MIDI rendering is taken from NOTE token.
          # INST remains as a structural hint token and is ignored at decoding stage.
          continue
        if note_type == 'BEAT':
          if 'beat' in notefeature2idx:
            beat_text = note_tokens[notefeature2idx['beat']]
            if isinstance(beat_text, str) and 'Beat_' in beat_text:
              try:
                cur_beat_idx = int(beat_text.split('_')[-1])
              except Exception:
                cur_beat_idx = 0
          if 'tempo' in notefeature2idx:
            tempo_text = note_tokens[notefeature2idx['tempo']]
            if tempo_text != 0 and tempo_text != 'Tempo_N_N':
              try:
                tempo_val = int(str(tempo_text).split('_')[-1])
                tempo_pos = bar_pos + cur_beat_idx * default_in_beat_ticks
                midi_obj.tempo_changes.append(TempoChange(tempo=tempo_val, time=tempo_pos))
              except Exception:
                pass
          if not saw_chord_prefix and 'chord' in notefeature2idx:
            chord_name = note_tokens[notefeature2idx['chord']]
            if (
              isinstance(chord_name, str)
              and chord_name.startswith('Chord_')
              and chord_name != 'Chord_N_N'
            ):
              if cur_beat_idx not in emitted_chord_on_beat:
                chord_pos = bar_pos + cur_beat_idx * default_in_beat_ticks
                midi_obj.markers.append(Marker(text=str(chord_name), time=chord_pos))
                emitted_chord_on_beat[cur_beat_idx] = str(chord_name)
          continue
        if note_type == 'STO':
          break
        if note_type == 'EOS':
          song_end = True
          break

        if note_type not in {'NOTE', 'CON'}:
          continue

        cur_pos = bar_pos + cur_beat_idx * default_in_beat_ticks

        # Instrument is consumed from NOTE token (instead of INST token).
        if 'instrument' in notefeature2idx:
          inst_text = note_tokens[notefeature2idx['instrument']]
          if inst_text != 0 and inst_text != 'CONTI':
            try:
              cur_instr = int(str(inst_text).split('_')[-1])
            except Exception:
              pass

        if (include_instruments is not None and cur_instr not in include_instruments):
            continue

        if cur_instr in exclude_instruments:
            continue

        # note body
        try:
          pitch = int(str(note_tokens[notefeature2idx['pitch']]).split('_')[-1])
          dur = int(str(note_tokens[notefeature2idx['duration']]).split('_')[-1]) * default_in_beat_ticks
          if 'velocity' in notefeature2idx:
            vel = int(str(note_tokens[notefeature2idx['velocity']]).split('_')[-1])
          else:
            vel = 90
          instr_notes_dict[cur_instr].append(Note(pitch=pitch, start=cur_pos, end=cur_pos + dur, velocity=vel))
        except Exception:
          continue

    # save tracks
    self._append_tracks_to_midi(midi_obj, instr_notes_dict)

    if isinstance(output_path, str) or isinstance(output_path, Path):
      midi_obj.dump(str(output_path))
    return midi_obj
