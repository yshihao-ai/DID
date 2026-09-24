import argparse
import json
import time
import itertools
import copy
from copy import deepcopy
import mido
from pathlib import Path
from multiprocessing import Pool, cpu_count
from collections import Counter, defaultdict
from fractions import Fraction
from typing import List
import traceback
import numpy as np
import pickle
from tqdm import tqdm

import miditoolkit
from miditoolkit.midi.containers import Marker, Instrument, TempoChange, TimeSignature
from chorder import Dechorder

from constants import NUM2PITCH, PROGRAM_INSTRUMENT_MAP, INSTRUMENT_PROGRAM_MAP

'''
This script is designed to preprocess MIDI files and convert them into a structured corpus suitable for symbolic music analysis or model training. 
It handles various tasks, including setting beat resolution, calculating duration, velocity, and tempo bins, and processing MIDI data into quantized musical events. 
'''
POP909_QUALITY_MAP = {
    'maj': 'M',
    'min': 'm',
    'aug': '+',
    'dim': 'o',

    '7': '7',
    'maj7': 'M7',
    'min7': 'm7',
    'dim7': 'o7',
    'hdim7': '/o7',

    'sus2': 'sus2',
    'sus4': 'sus4',

    'maj6': 'M',
    'min6': 'm',
    'minmaj7': 'm',
    'sus4(b7)': 'sus4',
}

POP909_PITCH_TO_PC = {
    'C': 0,
    'C#': 1, 'Db': 1,
    'D': 2,
    'D#': 3, 'Eb': 3,
    'E': 4,
    'F': 5,
    'F#': 6, 'Gb': 6,
    'G': 7,
    'G#': 8, 'Ab': 8,
    'A': 9,
    'A#': 10, 'Bb': 10,
    'B': 11,
}

POP909_INVERSION_TO_OFFSET = {
    '1': 0,
    'b2': 1,
    '2': 2,
    'b3': 3,
    '3': 4,
    '4': 5,
    'b5': 6,
    '5': 7,
    '#5': 8,
    '6': 9,
    'b7': 10,
    '7': 11,
}

def get_tempo_bin(max_tempo:int, ratio:float=1.1):
  bpm = 30
  regular_tempo_bins = [bpm]
  while bpm < max_tempo:
    bpm *= ratio
    bpm = round(bpm)
    if bpm > max_tempo:
      break
    regular_tempo_bins.append(bpm)
  return np.array(regular_tempo_bins)

def split_markers(markers:List[miditoolkit.midi.containers.Marker]):
  '''
  split markers into chord, tempo, label
  '''
  chords = []
  for marker in markers:
    splitted_text = marker.text.split('_')
    if splitted_text[0] != 'global' and 'Boundary' not in splitted_text[0]:
      chords.append(marker)
  return chords

class CorpusMaker():
  def __init__(
      self, 
      dataset_name:str, 
      num_features:int, 
      in_dir:Path, 
      out_dir:Path, 
      debug:bool
  ):
    '''
    Initialize the CorpusMaker with dataset information and directory paths.
    It sets up MIDI paths, output directories, and debug mode, then
    retrieves the beat resolution, duration bins, velocity/tempo bins, and prepares the MIDI file list.
    '''
    self.dataset_name = dataset_name
    self.num_features = num_features
    self.midi_path = in_dir / f"{dataset_name}"
    self.out_dir = out_dir
    self.debug = debug
    self._get_in_beat_resolution()
    self._get_duration_bins()
    self._get_velocity_tempo_bins()
    self._get_min_max_last_time()
    self._prepare_midi_list()
  
  def _parse_pop909_chord_label(self, label: str) -> str:
    if label == 'N':
        return 'N_N_N'

    if '/' in label:
        base, inversion = label.split('/', 1)
    else:
        base = label
        inversion = None

    if ':' not in base:
        raise ValueError(f"Invalid Pop909 chord label: {label}")

    root_name, quality_name = base.split(':', 1)

    if root_name not in POP909_PITCH_TO_PC:
        raise ValueError(f"Unknown Pop909 chord root: {root_name}")

    if quality_name not in POP909_QUALITY_MAP:
        raise ValueError(f"Unknown Pop909 chord quality: {quality_name}")

    root_pc = POP909_PITCH_TO_PC[root_name]
    quality = POP909_QUALITY_MAP[quality_name]

    if inversion is None:
        bass_offset = 0
    else:
        if inversion not in POP909_INVERSION_TO_OFFSET:
            raise ValueError(f"Unknown Pop909 inversion: {label}")
        bass_offset = POP909_INVERSION_TO_OFFSET[inversion]

    bass_pc = (root_pc + bass_offset) % 12

    # 统一使用现有 NUM2PITCH，Bb -> A#，Eb -> D# 等
    root = NUM2PITCH[root_pc]
    bass = NUM2PITCH[bass_pc]

    # 保持 corpus 原有的三字段 marker protocol
    return f'{root}_{quality}_{bass}'

  @staticmethod
  def _nearest_tick_from_seconds(seconds: float, tick_to_time: np.ndarray) -> int:
      right = int(np.searchsorted(tick_to_time, seconds, side='left'))
  
      if right <= 0:
          return 0
  
      if right >= len(tick_to_time):
          return len(tick_to_time) - 1
  
      left = right - 1
  
      if (seconds - tick_to_time[left] <= tick_to_time[right] - seconds):
          return left
  
      return right  
  
  def _load_pop909_chord_markers(self, midi_path: Path, original_tick_to_time: np.ndarray, shift_ticks: int):
      beat_midi_path = midi_path.parent / 'beat_midi.txt'
      chord_midi_path = midi_path.parent / 'chord_midi.txt'
  
      if not beat_midi_path.is_file():
          raise FileNotFoundError(f"Missing Pop909 annotation: {beat_midi_path}")
  
      if not chord_midi_path.is_file():
          raise FileNotFoundError(f"Missing Pop909 annotation: {chord_midi_path}")
  
      # ---------- load official beat times ----------
      beat_times = []
  
      with beat_midi_path.open('r', encoding='utf-8') as f:
          for line in f:
              line = line.strip()
  
              if not line:
                  continue
  
              columns = line.split()
  
              if len(columns) < 1:
                  continue
  
              beat_times.append(float(columns[0]))
  
      if not beat_times:
          raise ValueError(f"No beat annotation in {beat_midi_path}")
      beat_times = np.asarray(beat_times, dtype=np.float64)
  
      markers = []
      # ---------- load official chords ----------
      with chord_midi_path.open('r',  encoding='utf-8') as f:
          for line in f:
              line = line.strip()
  
              if not line:
                  continue
  
              columns = line.split()
  
              if len(columns) < 3:
                  raise ValueError(f"Malformed chord_midi line: {line}")
  
              start_sec = float(columns[0])
              chord_label = columns[2]
  
              # ==========================================
              # 1. chord start 必须先 snap 到 official beat
              # ==========================================
              right = int(np.searchsorted(beat_times, start_sec, side='left'))
  
              candidates = []
  
              if right < len(beat_times):
                  candidates.append(beat_times[right])
  
              if right > 0:
                  candidates.append(beat_times[right - 1])
  
              if not candidates:
                  raise ValueError(
                      f"No beat candidate for chord: "
                      f"{midi_path.name}, {line}"
                  )
  
              snapped_time = min(candidates, key=lambda x: abs(x - start_sec))
  
              alignment_error = abs(snapped_time - start_sec)
  
              # 实际 Pop909 start boundary 的误差远小于这个值
              if alignment_error > 1e-3:
                  raise ValueError(
                      f"Pop909 chord is not beat-aligned: "
                      f"{midi_path.name}, "
                      f"chord_time={start_sec}, "
                      f"nearest_beat={snapped_time}, "
                      f"error={alignment_error}"
                  )
  
              # ==========================================
              # 2. official beat seconds -> ORIGINAL MIDI tick
              # ==========================================
              original_tick = self._nearest_tick_from_seconds(snapped_time, original_tick_to_time)
  
              # ==========================================
              # 3. 使用和 NOTE 完全相同的 phase shift
              # ==========================================
              aligned_tick = (original_tick + shift_ticks)
              marker_text = self._parse_pop909_chord_label(chord_label)
  
              markers.append(Marker(time=int(aligned_tick), text=marker_text))
  
      # ---------- de-duplicate consecutive chords ----------
      dedup_markers = []
      prev_text = None
  
      for marker in markers:
          if marker.text != prev_text:
              dedup_markers.append(marker)
              prev_text = marker.text
  
      return dedup_markers
  
  def _get_in_beat_resolution(self):
    # Retrieve the resolution of quarter note based on the dataset name (e.g., 4 means the minimum resolution sets to 16th note)
    in_beat_resolution_dict = {'BachChorale': 4, 'Pop1k7': 4, 'Pop909': 4, 'SOD': 12, 'LakhClean': 4, 'SymphonyMIDI': 8, 'EMOPIA+': 4, "PIAST": 4}
    self.in_beat_resolution = in_beat_resolution_dict[self.dataset_name]

  def _get_duration_bins(self):
    # Set up regular duration bins for quantizing note lengths, based on the beat resolution.
    base_duration = {4:[1,2,3,4,5,6,8,10,12,16,20,24,28,32],
                     8:[1,2,3,4,6,8,10,12,14,16,20,24,28,32,36,40,48,56,64],
                     12:[1,2,3,4,6,9,12,15,18,24,30,36,42,48,54,60,72,84,96]}
    base_duration_list = base_duration[self.in_beat_resolution]
    self.regular_duration_bins = np.array(base_duration_list)

  def _get_velocity_tempo_bins(self):
    # Define velocity and tempo bins based on whether the dataset is a performance or score type.
    midi_type_dict = {'BachChorale': 'score', 'Pop1k7': 'perform', 'Pop909': 'score', 'SOD': 'score', 'LakhClean': 'score', 'Symphony': 'score', 'EMOPIA+': 'perform', "PIAST": 'perform'}
    midi_type = midi_type_dict[self.dataset_name]
    # For performance-type datasets, set finer granularity of velocity and tempo bins.
    if midi_type == 'perform':
      self.regular_velocity_bins = np.array(list(range(40, 128, 8)) + [127])
      self.regular_tempo_bins = get_tempo_bin(max_tempo=240, ratio=1.04)
    # For score-type datasets, use coarser velocity and tempo bins.
    elif midi_type == 'score':
      self.regular_velocity_bins = np.array([40, 60, 80, 100, 120])
      self.regular_tempo_bins = get_tempo_bin(max_tempo=390, ratio=1.1)

  def _get_min_max_last_time(self):
    '''
    Set the minimum and maximum allowed length of a MIDI track, depending on the dataset.
    0 to 2000 means no limitation
    '''
    last_time_dict = {'BachChorale': (0, 2000), 'Pop1k7': (0, 2000), 'Pop909': (0, 2000), 'SOD': (60, 1000), 'LakhClean': (60, 600), 'Symphony': (60, 1500), 'EMOPIA+': (0, 2000), "PIAST": (0, 2000)}
    self.min_last_time, self.max_last_time = last_time_dict[self.dataset_name]

  def _prepare_midi_list_1(self):
    midi_path = Path(self.midi_path)
    all_files = sorted(list(midi_path.rglob("*.midi")) + list(midi_path.rglob("*.mid")))
    if self.dataset_name == 'Pop909':
        print(f"Filtering Pop909: Keeping numeric IDs and their shifted versions...")
        
        valid_files = []
        for f in all_files:
            stem = f.stem  # 获取不带后缀的文件名
            
            # 逻辑：
            # 1. 如果是纯数字 (例如 "909") -> 保留
            # 2. 如果包含 "_shift" (例如 "909_shift_-2") -> 只要 "_shift" 前面是数字 -> 保留
            
            core_id = stem
            if '_shift' in stem:
                core_id = stem.split('_shift')[0] # 提取 "909"
            
            if core_id.isdigit():
                valid_files.append(f)
            # else: 丢弃 (例如 "903-v1", "903-v2")
        
        self.midi_list = valid_files
        
        print(f"  - Total found: {len(all_files)}")
        print(f"  - After filtering: {len(self.midi_list)}")
        # 打印几个保留的和丢弃的例子，供检查
        print(f"  - Kept examples: {[f.name for f in self.midi_list[:3]]}")
    else:
        self.midi_list = all_files

  @staticmethod
  def _is_all_4_4_from_beat_audio(
      beat_audio_path: Path,
  ) -> bool:

      if not beat_audio_path.is_file():
          return False
      beat_orders = []
      try:
          with beat_audio_path.open("r", encoding="utf-8") as file:
              for line in file:
                  line = line.strip()
                  if not line:
                      continue
                  columns = line.split()
                  if len(columns) < 2:
                      return False
                  try:
                      beat_order_float = float(columns[1])
                  except ValueError:
                      return False
                  beat_order = int(round(beat_order_float))

                  if (abs(beat_order_float - beat_order) > 1e-6):
                      return False
                  beat_orders.append(beat_order)
      except OSError:
          return False

      if not beat_orders:
          return False

      if any(order not in {1, 2, 3, 4} for order in beat_orders):
          return False

      downbeat_indices = [index for index, order in enumerate(beat_orders) if order == 1]

      if len(downbeat_indices) < 2:
          return False
      expected_measure = [1, 2, 3, 4]
      first_downbeat_index = downbeat_indices[0]
      prefix = beat_orders[:first_downbeat_index]

      if prefix:
          expected_prefix = expected_measure[-len(prefix):]
          if prefix != expected_prefix:
              return False
      for left, right in zip(downbeat_indices, downbeat_indices[1:]):
          measure_orders = beat_orders[left:right]
          if measure_orders != expected_measure:
              return False

      last_downbeat_index = downbeat_indices[-1]
      suffix = beat_orders[last_downbeat_index:]
      expected_suffix = expected_measure[:len(suffix)]
      if suffix != expected_suffix:
          return False
      return True
  
  def _prepare_midi_list(self):
    midi_path = Path(self.midi_path)
    all_files = sorted(list(midi_path.rglob("*.midi")) + list(midi_path.rglob("*.mid")))
    if self.dataset_name != "Pop909":
        self.midi_list = all_files
        return
    print("Filtering Pop909 using beat_audio.txt four-beat cycles...")

    valid_files = []
    irregular_tunes = []
    missing_annotations = []

    for file_path in all_files:
        stem = file_path.stem
        core_id = stem
        if not core_id.isdigit():
            continue

        annotation_candidates = [
            file_path.parent / "beat_audio.txt",
            file_path.parent.parent / "beat_audio.txt",
            midi_path / core_id / "beat_audio.txt",
        ]

        beat_audio_path = next((path for path in annotation_candidates if path.is_file()), None,)

        if beat_audio_path is None:
            irregular_tunes.append(stem)
            missing_annotations.append(stem)
            continue
        is_all_4_4 =self._is_all_4_4_from_beat_audio(beat_audio_path)
        if is_all_4_4:
            valid_files.append(file_path)
        else:
            irregular_tunes.append(stem)
    self.midi_list = valid_files

    metadata_dir = ( midi_path.parent.parent / "metadata")
    metadata_dir.mkdir(parents=True, exist_ok=True)
    irregular_path = (metadata_dir / "Pop909_irregular_tunes.json")

    with irregular_path.open("w", encoding="utf-8") as file:
        json.dump(
            sorted(set(irregular_tunes)),
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"  - Candidate MIDI files: {len(all_files)}")
    print(f"  - Kept all-4/4 files: {len(valid_files)}")
    print(f"  - Filtered files: {len(set(irregular_tunes))}")
    print(f"  - Missing beat_audio.txt: {len(set(missing_annotations))}")
    print(f"  - Irregular list:{irregular_path}")

  def make_corpus(self) -> None:
    '''
    Main method to process the MIDI files and create the corpus data.
    It supports both single-processing (debug mode) and multi-processing for large datasets.
    '''
    print("preprocessing midi data to corpus data")
    # check the corpus folder is already exist and make it if not
    Path(self.out_dir).mkdir(parents=True, exist_ok=True)
    Path(self.out_dir / f"corpus_{self.dataset_name}").mkdir(parents=True, exist_ok=True)
    Path(self.out_dir / f"midi_{self.dataset_name}").mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    if self.debug:
      # single processing for debugging
      broken_counter = 0
      success_counter = 0
      for file_path in tqdm(self.midi_list, total=len(self.midi_list)):
        message = self._mp_midi2corpus(file_path)
        if message == "error":
          broken_counter += 1
        elif message == "success":
          success_counter += 1
    else:
    # Multi-threaded processing for faster corpus generation.
      broken_counter = 0
      success_counter = 0
      # with Pool(cpu_count()) as p:
      with Pool(16) as p:
        for message in tqdm(p.imap(self._mp_midi2corpus, self.midi_list), total=len(self.midi_list)):
          if message == "error":
            broken_counter += 1
          elif message == "success":
            success_counter += 1
    print(f"Making corpus takes: {time.time() - start_time}s, success: {success_counter}, broken: {broken_counter}")

  def _mp_midi2corpus(self, file_path:Path):
    # Converts a single MIDI file to corpus format and saves both the corpus and MIDI.
    try:
      # --- 新增的代码 START ---
      # 在这一步就把脏数据拦截下来，防止后面卡死
      if not self._is_valid_midi(file_path):
        return "error"
      # --- 新增的代码 END ---
      midi_obj = self._analyze(file_path)
      corpus, midi_obj = self._midi2corpus(midi_obj)
      # Save corpus as a pickle file and the corresponding MIDI object.
      filename = file_path.stem + ".pkl"  # Get the stem (filename without extension) of the original file path
      save_path = Path(self.out_dir) / f"corpus_{self.dataset_name}" / filename  # Create a new Path object for saving
      with save_path.open('wb') as f:
        pickle.dump(corpus, f)
      midiname = file_path.stem + ".mid"
      save_path = Path("../dataset/represented_data/corpus") / f"midi_{self.dataset_name}" / midiname
      midi_obj.dump(save_path)
      del midi_obj, corpus
      return "success"
    except Exception as e:
      # 使用分隔线让错误信息在大量日志中更显眼
      # print(f"\n{'!'*20} ERROR REPORT {'!'*20}")
      # print(f"出错文件 (File): {file_path.name}")
      # print(f"完整路径: {file_path}")
      # print("-" * 10 + " 错误堆栈 (Traceback) " + "-" * 10)
      
      # # 这行代码是核心：它会打印出具体是哪一行代码崩了
      # traceback.print_exc() 
      
      # print(f"{'!'*54}\n")
      return "error"
       
  def _is_valid_midi(self, file_path: Path, max_seconds=6000):
    '''
    检查 MIDI 文件是否有效且长度在允许范围内。
    max_seconds: 默认 600秒 (10分钟)，超过这个长度（包括脏数据导致的超长时长）直接跳过。
    '''
    try:
      # clip=True 可以修复轻微格式错误，防止报错
      mid = mido.MidiFile(file_path, clip=True)
      
      # 1. 检查总时长
      # mido 会自动计算所有轨道的总秒数。
      # 如果是脏数据 (42亿 tick)，mid.length 会变成几百万秒，这里直接就会返回 False。
      if mid.length > max_seconds:
        print(f"[Skip] {file_path.name}: Too long ({mid.length:.1f}s > {max_seconds}s)")
        return False
        
      # 2. 检查是否有轨道
      if len(mid.tracks) == 0:
        return False
            
      return True
      
    except Exception as e:
      # 文件头彻底损坏，无法读取
      print(f"[Skip] {file_path.name}: Corrupted file ({e})")
      return False
      
  def _check_length(self, last_time:float):
    if last_time < self.min_last_time or last_time > self.max_last_time:
      raise ValueError(f"last time {last_time} is out of range")
  
  def _align_pop909_bar_phase(self, midi_obj, midi_path: Path) -> None:
    beat_midi_path = midi_path.parent / "beat_midi.txt"
    if not beat_midi_path.is_file():
      raise FileNotFoundError(f"Missing Pop909 annotation: {beat_midi_path}")
  
    if midi_obj.ticks_per_beat % self.in_beat_resolution != 0:
      raise ValueError(
        f"Pop909 ticks_per_beat={midi_obj.ticks_per_beat} cannot be divided "
        f"into {self.in_beat_resolution} steps per beat"
      )
  
    tick_to_time = midi_obj.get_tick_to_time_mapping()
    if len(tick_to_time) == 0:
      raise ValueError(f"Empty tick-to-time mapping: {midi_path}")
  
    step_ticks = midi_obj.ticks_per_beat // self.in_beat_resolution
    bar_steps = 4 * self.in_beat_resolution
    downbeat_phases = []
  
    with beat_midi_path.open("r", encoding="utf-8") as file:
      for line in file:
        columns = line.split()
        if not columns:
          continue
        if len(columns) < 3:
          raise ValueError(f"Malformed beat_midi.txt line: {line.rstrip()}")
  
        # beat_midi.txt 第三列表示四拍小节的下拍标记
        if float(columns[2]) <= 0.5:
          continue
  
        beat_time = float(columns[0])
  
        # 不使用 MIDI 有效时间范围之外的标注
        if beat_time < tick_to_time[0] or beat_time > tick_to_time[-1]:
          continue
  
        # 将秒数转换为最接近的 MIDI tick
        right_tick = int(
          np.searchsorted(tick_to_time, beat_time, side="left")
        )
  
        if right_tick <= 0:
          nearest_tick = 0
        elif right_tick >= len(tick_to_time):
          nearest_tick = len(tick_to_time) - 1
        else:
          left_tick = right_tick - 1
  
          if (
            beat_time - tick_to_time[left_tick]
            <= tick_to_time[right_tick] - beat_time
          ):
            nearest_tick = left_tick
          else:
            nearest_tick = right_tick
  
        quantized_step = int(round(nearest_tick / step_ticks))
        downbeat_phases.append(quantized_step % bar_steps)
  
    if not downbeat_phases:
      raise ValueError(f"No four-beat downbeat found in {beat_midi_path}")
  
    # 使用整首歌中出现次数最多的下拍相位，避免只依赖第一条可能不稳定的标注
    phase_steps = Counter(downbeat_phases).most_common(1)[0][0]
  
    # 将该相位移动到下一个完整小节的 Beat_0
    shift_steps = (-phase_steps) % bar_steps
  
    # 本身已经与 tick 0 对齐，不做任何修改
    if shift_steps == 0:
      return 0, tick_to_time
  
    shift_ticks = shift_steps * step_ticks
  
    # 移动所有音符；只改变绝对位置，不改变音高、时值和力度
    for instrument in midi_obj.instruments:
      for note in instrument.notes:
        note.start += shift_ticks
        note.end += shift_ticks
  
    # 速度事件也必须同步移动
    initial_tempo = 120.0
    for tempo in sorted(
      midi_obj.tempo_changes,
      key=lambda item: item.time,
    ):
      if tempo.time > 0:
        break
      initial_tempo = tempo.tempo
  
    for tempo in midi_obj.tempo_changes:
      tempo.time += shift_ticks
  
    # 移动后仍需保证 tick 0 有正确速度
    midi_obj.tempo_changes.insert(
      0,
      TempoChange(
        tempo=initial_tempo,
        time=0,
      ),
    )
  
    midi_obj.max_tick += shift_ticks
    return shift_ticks, tick_to_time
    
  def _analyze(self, midi_path:Path):
    # Loads and analyzes a MIDI file, performing various checks and extracting chords.
    midi_obj = miditoolkit.midi.parser.MidiFile(midi_path)
    if self.dataset_name == "Pop909":
        midi_obj.time_signature_changes = [
            TimeSignature(
                numerator=4,
                denominator=4,
                time=0,
            )
        ]
    # 调整轨道信息
    if self.dataset_name == 'Pop909':
        # 1. (可选) 调试模式下打印原始轨道信息，用于确认
        print(f"\n[Check Pop909] File: {midi_path.name}")
        for i, track in enumerate(midi_obj.instruments):
            print(f"  - Track {i}: Name='{track.name}', Program={track.program}, Notes={len(track.notes)}")

        # 2. 强制重新分配 Program ID 以防止合并
        # POP909 的轨道名称通常包含 MELODY, BRIDGE, PIANO
        for instr in midi_obj.instruments:
            # 统一转大写处理，防止大小写差异
            t_name = instr.name.upper()
            
            # 策略：Melody 保持钢琴(0), Bridge 变吉他(24), Piano 变弦乐(48)
            # 这样在 _make_instr_notes 中会被归类到不同的 Key，从而分离
            
            if 'MELODY' in t_name:
                instr.program = 0   # Acoustic Grand Piano (保持不变)
            elif 'BRIDGE' in t_name:
                instr.program = 24  # Acoustic Guitar (代理)
            elif 'PIANO' in t_name:
                instr.program = 48  # String Ensemble 1 (代理)
            else:
                # 兜底策略：如果是主旋律轨道（通常 POP909 0轨或1轨是旋律）
                # 这里可以根据实际打印出的 track index 进一步细化
                raise ValueError("Error")
                exit()
    # --- 新增代码 END ---

    # check length
    mapping = midi_obj.get_tick_to_time_mapping()
    last_time = mapping[midi_obj.max_tick]
    self._check_length(last_time)
    
    midi_obj.instruments = [ins for ins in midi_obj.instruments if len(ins.notes) > 0]
    if len(midi_obj.instruments) == 0:
        raise ValueError("No valid instruments found (all tracks are empty)")
    
    # for ins in midi_obj.instruments:
    #   # delete instrument with no notes
    #   if len(ins.notes) == 0:
    #     midi_obj.instruments.remove(ins)
    #     continue
    #   notes = ins.notes
    #   notes = sorted(notes, key=lambda x: (x.start, x.pitch))

    # three steps to merge instruments
    self._merge_percussion(midi_obj)
    self._pruning_instrument(midi_obj)
    self._limit_max_track(midi_obj)
    
    pop909_shift_ticks = 0
    pop909_original_tick_to_time = None
    if self.dataset_name == 'Pop909':
      pop909_shift_ticks, pop909_original_tick_to_time = self._align_pop909_bar_phase(midi_obj, midi_path)

    if self.num_features == 7 or self.num_features == 8:
      # ============================================
      # Pop909: official chord_midi.txt
      # ============================================
      if self.dataset_name == 'Pop909':
          markers = self._load_pop909_chord_markers(
              midi_path=midi_path,
              original_tick_to_time=pop909_original_tick_to_time,
              shift_ticks=pop909_shift_ticks,
          )
          midi_obj.markers = markers
  
      # ============================================
      # Other datasets: ORIGINAL Dechorder
      # ============================================
      else:
        # in case of 7 or 8 features, we need to extract chords
        new_midi_obj = self._pruning_notes_for_chord_extraction(midi_obj)
        chords = Dechorder.dechord(new_midi_obj)
        markers = []
        for cidx, chord in enumerate(chords):
          if chord.is_complete():
            chord_text = NUM2PITCH[chord.root_pc] + '_' + chord.quality + '_' + NUM2PITCH[chord.bass_pc]
          else:
            chord_text = 'N_N_N'
          markers.append(Marker(time=int(cidx*new_midi_obj.ticks_per_beat), text=chord_text))
        
        # de-duplication
        prev_chord = None
        dedup_chords = []
        for m in markers:
          if m.text != prev_chord:
            prev_chord = m.text
            dedup_chords.append(m)
  
        # return midi
        midi_obj.markers = dedup_chords
    return midi_obj

  def _pruning_grouped_notes_from_quantization(self, instr_grid:dict):
    '''
    In case where notes are grouped in the same quant_time but with different start time, unintentional chords are created
    rule1: if notes have half step interval, delete the shorter one
    rule2: if notes do not share 50% of duration of the shorter note, delete the shorter one
    '''
    for instr in instr_grid.keys():
      time_list = sorted(list(instr_grid[instr].keys()))
      for time in time_list:
        notes = instr_grid[instr][time]
        if len(notes) == 1:
          continue
        else:
          new_notes = []
        # sort in pitch with ascending order
        notes.sort(key=lambda x: x.pitch)
        for i in range(len(notes)-1):
          # if start time is same add to new_notes
          if notes[i].start == notes[i+1].start:
            new_notes.append(notes[i])
            new_notes.append(notes[i+1])
            continue
          if notes[i].pitch == notes[i+1].pitch or notes[i].pitch + 1 == notes[i+1].pitch:
            # select longer note
            if notes[i].end - notes[i].start > notes[i+1].end - notes[i+1].start:
              new_notes.append(notes[i])
            else:
              new_notes.append(notes[i+1])
          else:
            # check how much duration they share
            shared_duration = min(notes[i].end, notes[i+1].end) - max(notes[i].start, notes[i+1].start)
            shorter_duration = min(notes[i].end - notes[i].start, notes[i+1].end - notes[i+1].start)
            # unless they share more than 80% of duration, select longer note (pruning shorter note)
            if shared_duration / shorter_duration < 0.8:
              if notes[i].end - notes[i].start > notes[i+1].end - notes[i+1].start:
                new_notes.append(notes[i])
              else:
                new_notes.append(notes[i+1])
            else:
              if len(new_notes) == 0:
                new_notes.append(notes[i])
                new_notes.append(notes[i+1])
              else:
                new_notes.append(notes[i+1])
        instr_grid[instr][time] = new_notes

  def _midi2corpus(self, midi_obj:miditoolkit.midi.parser.MidiFile):
    # Checks if the ticks per beat in the MIDI file is lower than the expected resolution.
    # If it is, raise an error.
    if midi_obj.ticks_per_beat < self.in_beat_resolution:
      raise ValueError(f'[x] Irregular ticks_per_beat. {midi_obj.ticks_per_beat}')

    # Ensure there is at least one time signature change in the MIDI file.
    if len(midi_obj.time_signature_changes) == 0:
      raise ValueError('[x] No time_signature_changes')
    
    # Ensure there are no duplicated time signature changes.
    time_list = [ts.time for ts in midi_obj.time_signature_changes]
    if len(time_list) != len(set(time_list)):
      raise ValueError('[x] Duplicated time_signature_changes')
    
    # If the dataset is 'LakhClean' or 'SymphonyMIDI', verify there are at least 4 tracks.
    if self.dataset_name == 'LakhClean' or self.dataset_name == 'SymphonyMIDI':
      if len(midi_obj.instruments) < 4:
        raise ValueError('[x] We will use more than 4 tracks in Lakh Clean dataset.')
    
    # Calculate the resolution of ticks per beat as a fraction.
    in_beat_tick_resol = Fraction(midi_obj.ticks_per_beat, self.in_beat_resolution)
    
    # Extract the initial time signature (numerator and denominator) and calculate the number of ticks for the first bar.
    initial_numerator = midi_obj.time_signature_changes[0].numerator
    initial_denominator = midi_obj.time_signature_changes[0].denominator
    first_bar_resol = int(midi_obj.ticks_per_beat * initial_numerator * (4 / initial_denominator))

    # --- load notes --- #
    instr_notes = self._make_instr_notes(midi_obj)
    
    # --- load information --- #
    # load chords, labels
    chords = split_markers(midi_obj.markers)
    chords.sort(key=lambda x: x.time)

    # load tempos
    tempos = midi_obj.tempo_changes
    tempos.sort(key=lambda x: x.time)

    # --- process items to grid --- #
    # compute empty bar offset at head
    first_note_time = min([instr_notes[k][0].start for k in instr_notes.keys()])
    last_note_time = max([instr_notes[k][-1].start for k in instr_notes.keys()])

    quant_time_first = int(round(first_note_time  / in_beat_tick_resol)) * in_beat_tick_resol
    offset = quant_time_first // first_bar_resol # empty bar
    offset_by_resol = offset * first_bar_resol

    # --- process notes --- #
    instr_grid = dict()
    for key in instr_notes.keys():
      notes = instr_notes[key]
      note_grid = defaultdict(list)
      for note in notes:
        # skip notes out of range, below C-1 and above C8
        if note.pitch < 12 or note.pitch >= 120:
          continue

        # in case when the first note starts at slightly before the first bar
        note.start = note.start - offset_by_resol if note.start - offset_by_resol > 0 else 0
        note.end = note.end - offset_by_resol if note.end - offset_by_resol > 0 else 0

        # relative duration
        # skip note with 0 duration
        note_duration = note.end - note.start
        relative_duration = round(note_duration / in_beat_tick_resol)
        if relative_duration == 0:
          continue
        if relative_duration > self.in_beat_resolution * 8: # 8 beats
          relative_duration = self.in_beat_resolution * 8
          
        # use regular duration bins
        note.quantized_duration = self.regular_duration_bins[np.argmin(abs(self.regular_duration_bins-relative_duration))]

        # quantize start time 
        quant_time = int(round(note.start / in_beat_tick_resol)) * in_beat_tick_resol

        # velocity
        note.velocity = self.regular_velocity_bins[
          np.argmin(abs(self.regular_velocity_bins-note.velocity))]

        # append
        note_grid[quant_time].append(note)

      # set to track
      instr_grid[key] = note_grid
    
    # --- pruning grouped notes --- #
    self._pruning_grouped_notes_from_quantization(instr_grid)
  
    # --- process chords --- #
    chord_grid = defaultdict(list)
    for chord in chords:
      # quantize
      chord.time = chord.time - offset_by_resol
      chord.time  = 0 if chord.time < 0 else chord.time
      quant_time = int(round(chord.time / in_beat_tick_resol)) * in_beat_tick_resol
      chord_grid[quant_time].append(chord)

    # --- process tempos --- #
    # first_notes_list = []
    # for instr in instr_grid.keys():
    #   time_list = sorted(list(instr_grid[instr].keys()))
    #   first_quant_time = time_list[0]
    #   first_notes_list.append(first_quant_time)
    # quant_first_note_time = min(first_notes_list)
    # --- process tempos --- #
    first_notes_list = []
    
    # [关键修改] 修复 Line 411 IndexError
    for instr in instr_grid.keys():
      time_list = sorted(list(instr_grid[instr].keys()))
      # 必须检查列表是否为空
      if len(time_list) > 0:
          first_quant_time = time_list[0]
          first_notes_list.append(first_quant_time)
    
    # 如果所有轨道都无法提取出第一个音符时间
    if not first_notes_list:
        raise ValueError("No quantized notes found to determine start time")

    quant_first_note_time = min(first_notes_list)

    tempo_grid = defaultdict(list)
    for tempo in tempos:
      # quantize
      tempo.time = tempo.time - offset_by_resol if tempo.time - offset_by_resol > 0 else 0
      quant_time = int(round(tempo.time / in_beat_tick_resol)) * in_beat_tick_resol
      tempo.tempo = self.regular_tempo_bins[
        np.argmin(abs(self.regular_tempo_bins-tempo.tempo))]
      if quant_time < quant_first_note_time:
        tempo_grid[quant_first_note_time].append(tempo)
      else:
        tempo_grid[quant_time].append(tempo)
    if len(tempo_grid[quant_first_note_time]) > 1:
      tempo_grid[quant_first_note_time] = [tempo_grid[quant_first_note_time][-1]]
    
    # --- process time signature --- #
    quant_time_signature = deepcopy(midi_obj.time_signature_changes)
    quant_time_signature.sort(key=lambda x: x.time)
    for ts in quant_time_signature:
      ts.time = ts.time - offset_by_resol if ts.time - offset_by_resol > 0 else 0
      ts.time = int(round(ts.time / in_beat_tick_resol)) * in_beat_tick_resol

    # --- make new midi object to check processed values --- #
    new_midi_obj = miditoolkit.midi.parser.MidiFile()
    new_midi_obj.ticks_per_beat = midi_obj.ticks_per_beat
    new_midi_obj.max_tick = midi_obj.max_tick
    for instr_idx in instr_grid.keys():
      new_instrument = Instrument(program=instr_idx)
      new_instrument.notes = [y for x in instr_grid[instr_idx].values() for y in x]
      new_midi_obj.instruments.append(new_instrument)
    new_midi_obj.markers = [y for x in chord_grid.values() for y in x]
    new_midi_obj.tempo_changes = [y for x in tempo_grid.values() for y in x]
    new_midi_obj.time_signature_changes = midi_obj.time_signature_changes

    # make corpus
    song_data = {
      'notes': instr_grid,
      'chords': chord_grid,
      'tempos': tempo_grid,
      'metadata': {
        'first_note': first_note_time,
        'last_note': last_note_time,
        'time_signature': quant_time_signature,
        'ticks_per_beat': midi_obj.ticks_per_beat,
        }
      }
    return song_data, new_midi_obj

  def _make_instr_notes(self, midi_obj):
    '''
    This part is important, we can use three different ways to merge instruments
    1st option: compare the number of notes and choose tracks with more notes
    2nd option: merge all instruments with the same tracks
    3rd option: leave all instruments as they are. differentiate tracks with different track number
    
    In this version we choose to use the 2nd option as it helps to reduce the number of tracks and sequence length
    '''
    instr_notes = defaultdict(list)
    for instr in midi_obj.instruments:
      instr_idx = instr.program
      # change instrument idx
      instr_name = PROGRAM_INSTRUMENT_MAP.get(instr_idx)
      if instr_name is None:
        continue
      new_instr_idx = INSTRUMENT_PROGRAM_MAP[instr_name]
      instr_notes[new_instr_idx].extend(instr.notes)
      instr_notes[new_instr_idx].sort(key=lambda x: (x.start, -x.pitch))
    return instr_notes

  # refered to SymphonyNet "https://github.com/symphonynet/SymphonyNet"
  def _merge_percussion(self, midi_obj:miditoolkit.midi.parser.MidiFile):
    '''
    merge drum track to one track
    '''
    drum_0_lst = []
    new_instruments = []
    for instrument in midi_obj.instruments:
      if len(instrument.notes) == 0:
        continue
      if instrument.is_drum:
        drum_0_lst.extend(instrument.notes)
      else:
        new_instruments.append(instrument)
    if len(drum_0_lst) > 0:
      drum_0_lst.sort(key=lambda x: x.start)
      # remove duplicate
      drum_0_lst = list(k for k, _ in itertools.groupby(drum_0_lst))
      drum_0_instrument = Instrument(program=114, is_drum=True, name="percussion")
      drum_0_instrument.notes = drum_0_lst
      new_instruments.append(drum_0_instrument)
    midi_obj.instruments = new_instruments

  # referred to mmt "https://github.com/salu133445/mmt"
  def _pruning_instrument(self, midi_obj:miditoolkit.midi.parser.MidiFile):
    '''
    merge instrument number with similar intrument category
    ex. 0: Acoustic Grand Piano, 1: Bright Acoustic Piano, 2: Electric Grand Piano into 0: Acoustic Grand Piano
    '''
    new_instruments = []
    for instr in midi_obj.instruments:
      instr_idx = instr.program
      # change instrument idx
      instr_name = PROGRAM_INSTRUMENT_MAP.get(instr_idx)
      if instr_name != None:
        new_instruments.append(instr)
    midi_obj.instruments = new_instruments

  # refered to SymphonyNet "https://github.com/symphonynet/SymphonyNet"
  def _limit_max_track(self, midi_obj:miditoolkit.midi.parser.MidiFile, MAX_TRACK:int=16):
      '''
      merge track with least notes to other track with same program
      and limit the maximum amount of track to 16
      '''
      if not midi_obj.instruments:
          return midi_obj
      if len(midi_obj.instruments) == 1:
        if midi_obj.instruments[0].is_drum:
          midi_obj.instruments[0].program = 114
          midi_obj.instruments[0].is_drum = False
        return midi_obj
      good_instruments = midi_obj.instruments
      good_instruments.sort(
          key=lambda x: (not x.is_drum, -len(x.notes)))  # place drum track or the most note track at first
      assert good_instruments[0].is_drum == True or len(good_instruments[0].notes) >= len(
          good_instruments[1].notes), tuple(len(x.notes) for x in good_instruments[:3])
      # assert good_instruments[0].is_drum == False, (, len(good_instruments[2]))
      track_idx_lst = list(range(len(good_instruments)))
      if len(good_instruments) > MAX_TRACK:
          new_good_instruments = copy.deepcopy(good_instruments[:MAX_TRACK])
          # print(midi_file_path)
          for id in track_idx_lst[MAX_TRACK:]:
              cur_ins = good_instruments[id]
              merged = False
              new_good_instruments.sort(key=lambda x: len(x.notes))
              for nid, ins in enumerate(new_good_instruments):
                  if cur_ins.program == ins.program and cur_ins.is_drum == ins.is_drum:
                      new_good_instruments[nid].notes.extend(cur_ins.notes)
                      merged = True
                      break
              if not merged:
                  pass
          good_instruments = new_good_instruments

      assert len(good_instruments) <= MAX_TRACK, len(good_instruments)
      for idx, good_instrument in enumerate(good_instruments):
          if good_instrument.is_drum:
              good_instruments[idx].program = 114
              good_instruments[idx].is_drum = False
      midi_obj.instruments = good_instruments

  def _pruning_notes_for_chord_extraction(self, midi_obj:miditoolkit.midi.parser.MidiFile):
    '''
    extract notes for chord extraction
    '''
    new_midi_obj = miditoolkit.midi.parser.MidiFile()
    new_midi_obj.ticks_per_beat = midi_obj.ticks_per_beat
    new_midi_obj.max_tick = midi_obj.max_tick
    new_instrument = Instrument(program=0, is_drum=False, name="for_chord")
    new_instruments = []
    new_notes = []
    for instrument in midi_obj.instruments:
      if instrument.program == 114 or instrument.is_drum: # pass drum track
        continue
      valid_notes = [note for note in instrument.notes if note.pitch >= 21 and note.pitch <= 108]
      new_notes.extend(valid_notes)
    new_notes.sort(key=lambda x: x.start)
    new_instrument.notes = new_notes
    new_instruments.append(new_instrument)
    new_midi_obj.instruments = new_instruments
    return new_midi_obj

def get_argument_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-d",
      "--dataset",
      required=True,
      choices=("BachChorale", "Pop1k7", "Pop909", "SOD", "LakhClean", "SymphonyMIDI", "EMOPIA+", "PIAST"),
      type=str,
      help="dataset names",
  )
  parser.add_argument(
      "-f",
      "--num_features",
      required=True,
      choices=(4, 5, 7, 8),
      type=int,
      help="number of features",
  )
  parser.add_argument(
      "-i",
      "--in_dir",
      default="../dataset/MIDI_dataset/",
      type=Path,
      help="input data directory",
  )
  parser.add_argument(
      "-o",
      "--out_dir",
      default="../dataset/represented_data/corpus/",
      type=Path,
      help="output data directory",
  )
  parser.add_argument(
      "--debug",
      action="store_true",
      help="enable debug mode",
  )
  return parser

def main():
  parser = get_argument_parser()
  args = parser.parse_args()
  corpus_maker = CorpusMaker(args.dataset, args.num_features, args.in_dir, args.out_dir, args.debug)
  corpus_maker.make_corpus()

if __name__ == "__main__":
  main()
