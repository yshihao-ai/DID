import miditoolkit
import copy
import argparse
import random
import sys
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def pitch_shift_midi(file_path: Path, output_dir: Path, shift_range: range, num_augmentations: int = None):
    """
    读取一个 MIDI 文件，随机抽取指定数量的移调版本并保存。
    """
    try:
        midi_obj = miditoolkit.midi.parser.MidiFile(str(file_path))
        
        all_pitches = []
        for instrument in midi_obj.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    all_pitches.append(note.pitch)
        
        if not all_pitches:
            return "skipped_no_notes"

        min_pitch = min(all_pitches)
        max_pitch = max(all_pitches)

        valid_shifts = []
        for shift in shift_range:
            if shift == 0: continue
            if (min_pitch + shift >= 0) and (max_pitch + shift <= 127):
                valid_shifts.append(shift)
        
        target_shifts = valid_shifts
        if num_augmentations is not None:
            if len(valid_shifts) > num_augmentations:
                target_shifts = random.sample(valid_shifts, num_augmentations)
            else:
                target_shifts = valid_shifts

        saved_count = 0

        for shift in target_shifts:
            new_midi = copy.deepcopy(midi_obj)

            for instrument in new_midi.instruments:
                if instrument.is_drum:
                    continue
                for note in instrument.notes:
                    note.pitch += shift

            # 这里的命名规则必须与下面的清理函数匹配
            # 格式: original_shift_+1.mid
            sign = "+" if shift > 0 else ""
            new_filename = f"{file_path.stem}_shift_{sign}{shift}{file_path.suffix}"
            
            # 如果是在源文件夹下增强，output_dir 就等于 file_path.parent
            # 但这里我们统一使用传入的 output_dir
            save_path = output_dir / new_filename
            
            new_midi.dump(str(save_path))
            saved_count += 1
            
        return f"success_{saved_count}"

    except Exception as e:
        return "error"

def process_wrapper(args):
    return pitch_shift_midi(*args)

def remove_augmented_files(target_dir: Path):
    """
    清理函数：删除指定目录下所有由本脚本生成的增强文件。
    识别特征：文件名中包含 '_shift_'
    """
    print(f"Searching for augmented files in: {target_dir} ...")
    
    # 递归查找所有包含特定标记的文件
    # 这里的 pattern 对应 pitch_shift_midi 中的命名规则
    aug_files = list(target_dir.rglob("*_shift_*"))
    aug_files = [f for f in aug_files if f.suffix.lower() in ['.mid', '.midi']]

    if not aug_files:
        print("No augmented files found.")
        return

    print(f"\nFound {len(aug_files)} augmented files.")
    print("Example files:")
    for f in aug_files[:5]:
        print(f" - {f.name}")
    if len(aug_files) > 5:
        print(" ...")

    # 安全确认
    print(f"\nWARNING: You are about to DELETE {len(aug_files)} files.")
    user_input = input("Are you sure? Type 'yes' to proceed: ").strip().lower()

    if user_input == 'yes':
        deleted_count = 0
        for f in tqdm(aug_files, desc="Deleting"):
            try:
                f.unlink() # 删除文件
                deleted_count += 1
            except Exception as e:
                print(f"Failed to delete {f.name}: {e}")
        print(f"\nCleanup complete. Deleted {deleted_count} files.")
    else:
        print("\nOperation cancelled.")

def main():
    parser = argparse.ArgumentParser(description="MIDI Pitch Augmentation & Cleanup Tool")
    
    # 输入目录是必须的
    parser.add_argument("-i", "--in_dir", type=str, required=True, help="Input directory containing .mid files")
    
    # 输出目录 (如果是清理模式，这个参数可以忽略，或者是源目录)
    parser.add_argument("-o", "--out_dir", type=str, required=False, help="Output directory (Optional if using --clean)")
    
    parser.add_argument("--min_shift", type=int, default=-5, help="Minimum semitone shift (default: -5)")
    parser.add_argument("--max_shift", type=int, default=6, help="Maximum semitone shift (default: 6)")
    parser.add_argument("-n", "--num", type=int, default=None, help="Number of augmentations per file")
    
    # [新增] 清理模式开关
    parser.add_argument("--clean", action="store_true", help="CLEANUP MODE: Delete all augmented files (*_shift_*) in the input directory")

    args = parser.parse_args()
    
    in_dir = Path(args.in_dir)
    if not in_dir.exists():
        print(f"Error: Input directory '{in_dir}' does not exist.")
        sys.exit(1)

    # ==========================
    # 分支 1: 清理模式
    # ==========================
    if args.clean:
        remove_augmented_files(in_dir)
        sys.exit(0)

    # ==========================
    # 分支 2: 增强模式
    # ==========================
    # 如果没有指定 out_dir，默认输出到 in_dir (即源文件夹增强)
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = in_dir
        print(f"No output directory specified. Augmenting directly into source: {out_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    
    midi_files = list(in_dir.rglob("*.mid")) + list(in_dir.rglob("*.midi"))
    # 过滤掉已经是增强过的文件，避免对增强文件再次增强 (shift_shift...)
    midi_files = [f for f in midi_files if "_shift_" not in f.name]
    
    print(f"Found {len(midi_files)} original MIDI files to process.")
    
    if args.num is not None:
        print(f"Mode: Randomly selecting {args.num} shifts per file.")
    else:
        print(f"Mode: Generating ALL valid shifts.")

    shift_range = range(args.min_shift, args.max_shift + 1)
    
    
    tasks = []
    for f in midi_files:
        # 如果是源文件夹增强模式，我们把生成的路径设为该文件原本的文件夹
        if args.out_dir is None or Path(args.out_dir) == in_dir:
            target_save_dir = f.parent
        else:
            target_save_dir = out_dir
        
        tasks.append((f, target_save_dir, shift_range, args.num))

    with Pool(cpu_count()) as p:
        results = list(tqdm(p.imap(process_wrapper, tasks), total=len(tasks)))
        
    success_files = len([r for r in results if r.startswith("success")])
    total_generated = sum([int(r.split('_')[1]) for r in results if r.startswith("success")])
    
    print(f"Processing complete.")
    print(f"Original files processed: {success_files}/{len(midi_files)}")
    print(f"Total new files generated: {total_generated}")

if __name__ == "__main__":
    main()