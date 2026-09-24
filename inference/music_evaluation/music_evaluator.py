# Modified from: https://github.com/RichardYang40148/mgeval/blob/master/__main__.py

from argparse import ArgumentParser
import glob
import copy
import os
import numpy as np
import pandas as pd
from collections import defaultdict
import pretty_midi
from pprint import pprint
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
import random
from mgeval import core_non_drum_multi as core, utils

# matplotlib.use('Agg')
import muspy

def get_muspy_metrics_df(set1_paths, set2_paths):
    """
    计算 MusPy 指标并返回一个 DataFrame。
    包含 Set1 和 Set2 的绝对数值（Mean）。
    """
    metrics_map = {
        'scale_consistency': muspy.scale_consistency,
        'groove_consistency': lambda m: muspy.groove_consistency(m, measure_resolution=4), # 默认按2小节计算
        'pitch_entropy': muspy.pitch_entropy,
        'pitch_class_entropy': muspy.pitch_class_entropy
    }
    
    # 辅助函数：计算单个集合的所有指标
    def compute_set_metrics(file_paths):
        set_scores = defaultdict(list)
        for path in file_paths:
            try:
                music = muspy.read_midi(path)
                for name, func in metrics_map.items():
                    try:
                        val = func(music)
                        if np.isnan(val) or np.isinf(val):
                            val = 0.0
                        set_scores[name].append(val)
                    except Exception as e:
                        print(e)
                        set_scores[name].append(0.0)
            except Exception:
                for name in metrics_map.keys():
                    set_scores[name].append(0.0)
        return set_scores

    print("  -> Calculating MusPy metrics...")
    set1_scores = compute_set_metrics(set1_paths) # 字典: key -> list of values
    set2_scores = compute_set_metrics(set2_paths)

    
    # 计算均值
    attributes = list(metrics_map.keys())
    set1_means = [np.mean(set1_scores[attr]) if set1_scores[attr] else 0.0 for attr in attributes]
    set2_means = [np.mean(set2_scores[attr]) if set2_scores[attr] else 0.0 for attr in attributes]
    
    # 构建 DataFrame
    df = pd.DataFrame({
        'attribute': attributes,
        'Set1_Mean': set1_means,
        'Set2_Mean': set2_means
    })
    return df
# =====================================================================

# ==================== [新增] 引入你提供的 Figaro 类 ====================
import tempfile
import shutil
import atexit
from tqdm import tqdm
from figaro.input_representation import InputRepresentation, remi2midi
from figaro.constants import BAR_KEY, EOS_TOKEN

# ==================== [新增] 基于 REMI 的裁剪函数 ====================
def crop_dataset_with_remi(file_list, target_bars, label="set"):
    """
    使用 InputRepresentation 将 MIDI 转为 Event 序列，
    截取前 target_bars 小节，然后还原为 MIDI。
    """
    if not file_list:
        return []
    
    if InputRepresentation is None:
        raise ImportError("缺少 representation.py，无法执行 REMI 裁剪。")

    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix=f"mgeval_remi_crop_{label}_")
    print(f"Creating temporary cropped dataset at: {temp_dir}")
    
    # 注册退出时清理
    def cleanup():
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temp dir: {temp_dir}")
    atexit.register(cleanup)

    new_file_list = []
    
    print(f"Quantizing and Cropping {len(file_list)} files to {target_bars} bars...")
    iterator = tqdm(file_list, desc=f"Processing {label}")
    
    for midi_path in iterator:
        try:
            # 1. 读取并量化 (do_extract_chords=False 以加快速度且减少依赖)
            # strict=False 允许处理一些稍微不规范的 MIDI
            rep = InputRepresentation(midi_path, do_extract_chords=False, strict=False)
            events = rep.get_remi_events()
            # 2. 截取逻辑
            bar_count = 0
            cutoff_index = len(events)
            
            # 扫描事件列表，寻找第 (target_bars + 1) 个 Bar 标记
            for i, event in enumerate(events):
                # event 格式通常是 "Bar_1", "Note_Pitch_60" 等
                if event.startswith(f"{BAR_KEY}_"):
                    bar_count += 1
                    if bar_count > target_bars:
                        cutoff_index = i
                        break
            
            # 执行切片
            cropped_events = events[:cutoff_index]
            # print(cropped_events)
            # 补全 EOS (如果切掉了)
            if cropped_events and cropped_events[-1] != EOS_TOKEN:
                cropped_events.append(EOS_TOKEN)
            
            # 3. 还原为 MIDI
            # 尝试获取原曲的时间签名，如果获取不到则默认 4/4
            # InputRepresentation 内部其实处理了 TimeSig 事件，remi2midi 应该能自动识别
            # 但 remi2midi 签名需要一个初始 time_signature
            init_ts = (4, 4)
            if rep.pm.time_signature_changes:

                ts = rep.pm.time_signature_changes[0]
                init_ts = (ts.numerator, ts.denominator)
            pm_new = remi2midi(cropped_events, time_signature=init_ts)
            
            # 4. 保存
            file_name = os.path.basename(midi_path)
            new_path = os.path.join(temp_dir, file_name)
            pm_new.write(new_path)
            new_file_list.append(new_path)
            
        except Exception as e:
            print(f"Warning: Failed to process {midi_path}: {e}")
            continue
            
    return new_file_list
# ===================================================================

def delete_nan(arr):
    arr[np.isnan(arr) | np.isinf(arr)] = 0
    return arr


parser = ArgumentParser()
parser.add_argument('--set1dir', required=True, type=str,
                    help='Path (absolute) to the first dataset (folder)')
parser.add_argument('--set2dir', required=True, type=str,
                    help='Path (absolute) to the second dataset (folder)')
parser.add_argument('--outdir', required=True, type=str,
                    help='Directory where the figures and analysis will be stored')
parser.add_argument('--savename', type=str, default=None,
                    help='savename of the output csv')                    
parser.add_argument('--num_sample', required=False, type=int,
                    help='Number of Samples to be evaluated')
parser.add_argument('--num_runs', required=False, type=int,
                    help='Number of Runs to compute mean and std')
# [新增参数]
parser.add_argument('--crop_bars', required=False, type=int, default=None,
                    help='If set, use InputRepresentation to crop files to this many bars.')

args = parser.parse_args()
print(args.set1dir)
print(args.set2dir)

set1_all = glob.glob(os.path.join(args.set1dir, '*.midi')) + glob.glob(os.path.join(args.set1dir, '*.mid'))
set2_all = glob.glob(os.path.join(args.set2dir, '*.midi')) + glob.glob(os.path.join(args.set2dir, '*.mid'))
# [新增逻辑] 2. 如果指定了 crop_bars，先进行裁剪
if args.crop_bars is not None and args.crop_bars > 0:
    ###################DEBUG#################################
    # target_name = "(Don't Mess With) My Toot Toot.mid" # 替换为你想要测试的文件名
    # set1_all = [f for f in set1_all if target_name in f]
    # set2_all = [f for f in set2_all if target_name in f]
    ###################DEBUG#################################

    print(f"\n[INFO] Cropping mode enabled: All files will be quantized and cropped to {args.crop_bars} bars.")
    # 对 Set1 (GT) 进行裁剪
    set1_all = crop_dataset_with_remi(set1_all, args.crop_bars, label="Set1")
    # 对 Set2 (Gen) 也进行裁剪 (以防万一生成过长)
    set2_all = crop_dataset_with_remi(set2_all, args.crop_bars, label="Set2")
    
    if not set1_all or not set2_all:
        print("Error: Cropping resulted in empty dataset(s). Check input files or log.")
        exit()

set2 = random.sample(set2_all, min(args.num_sample, len(set2_all)))
if not any(set2):
    print("Error: baseline set it empty")
    exit()

current_path = os.getcwd()
save_dir = os.path.join(current_path, args.outdir)
os.makedirs(os.path.expanduser(save_dir), exist_ok=True)


print('Evaluation begins ~')

dfs = []
muspy_dfs = []

for _ in range(args.num_runs):
    # select random part in the gt dataset
    set1 = random.sample(set1_all, min(args.num_sample, len(set1_all)))
    if not any(set1):
      print("Error: sample set it empty")
      exit()

    # Initialize Evaluation Set
    if args.num_sample:
        num_samples = min(args.num_sample, min(len(set2), len(set1)))
    else:
        num_samples = min(len(set2), len(set1))
    # num_samples = 10 # small number for faster testing and debugging purpose

    print("Number of samples in use: ", num_samples)
    evalset = {
                'total_used_pitch': np.zeros((num_samples, 1))
              , 'pitch_range': np.zeros((num_samples, 1))
              , 'avg_IOI': np.zeros((num_samples, 1))
              , 'total_pitch_class_histogram': np.zeros((num_samples, 12))
              , 'mean_note_velocity':np.zeros((num_samples, 1))
              , 'mean_note_duration':np.zeros((num_samples, 1))
              , 'note_density':np.zeros((num_samples, 1))
    }



    metrics_list = list(evalset.keys())

    single_arg_metrics = (
        [ 'total_used_pitch'
        , 'avg_IOI'
        , 'total_pitch_class_histogram'
        , 'pitch_range'
        , 'mean_note_velocity'
        , 'mean_note_duration'
        , 'note_density'
        , 'pitch_class_transition_matrix'
        ])

    set1_eval = copy.deepcopy(evalset)
    set2_eval = copy.deepcopy(evalset)

    sets = [ (set1, set1_eval), (set2, set2_eval) ]


    # Extract Features
    for _set, _set_eval in sets:
        for i in range(0, num_samples):
            feature = core.extract_feature(_set[i])
            for metric in metrics_list:
                evaluator = getattr(core.metrics(), metric)
                if metric in single_arg_metrics:
                    tmp = evaluator(feature)
                else:
                    tmp = evaluator(feature, 0)
                _set_eval[metric][i] = tmp

    loo = LeaveOneOut()
    loo.get_n_splits(np.arange(num_samples))
    set1_intra = np.zeros((num_samples, len(metrics_list), num_samples - 1))
    set2_intra = np.zeros((num_samples, len(metrics_list), num_samples - 1))


    # Calculate Intra-set Metrics
    for i, metric in enumerate(metrics_list):
        for train_index, test_index in loo.split(np.arange(num_samples)):
            set1_intra[test_index[0]][i] = utils.c_dist(
                set1_eval[metrics_list[i]][test_index], set1_eval[metrics_list[i]][train_index])
            set2_intra[test_index[0]][i] = utils.c_dist(
                set2_eval[metrics_list[i]][test_index], set2_eval[metrics_list[i]][train_index])

    loo = LeaveOneOut()
    loo.get_n_splits(np.arange(num_samples))
    sets_inter = np.zeros((num_samples, len(metrics_list), num_samples))

    # Calculate Inter-set Metrics
    for i, metric in enumerate(metrics_list):
        for train_index, test_index in loo.split(np.arange(num_samples)):
            sets_inter[test_index[0]][i] = utils.c_dist(set1_eval[metric][test_index], set2_eval[metric])


    plot_set1_intra = np.transpose(
        set1_intra, (1, 0, 2)).reshape(len(metrics_list), -1)
    plot_set2_intra = np.transpose(
        set2_intra, (1, 0, 2)).reshape(len(metrics_list), -1)
    plot_sets_inter = np.transpose(
        sets_inter, (1, 0, 2)).reshape(len(metrics_list), -1)

    plot_set1_intra = delete_nan(plot_set1_intra)
    plot_set2_intra = delete_nan(plot_set2_intra)
    plot_sets_inter = delete_nan(plot_sets_inter)

    # output = {}
    df_output = defaultdict(list)
    for i, metric in enumerate(metrics_list):
        print("-----------------------------")
        print('calculating KL and OA of: {}'.format(metric))

        mean = np.mean(set1_eval[metric], axis=0).tolist()
        std = np.std(set1_eval[metric], axis=0).tolist()
        filename = metric+".png"

        plt.figure()
        sns.kdeplot(plot_set1_intra[i], label='intra_set1')
        sns.kdeplot(plot_sets_inter[i], label='inter')
        sns.kdeplot(plot_set2_intra[i], label='intra_set2')

        plt.title(metrics_list[i])
        plt.xlabel('Euclidean distance')
        plt.legend()
        plt.savefig(f'{save_dir}/{filename}')
        # plt.show()
        plt.close()

        kl1 = utils.kl_dist(plot_set1_intra[i], plot_sets_inter[i])
        ol1 = utils.overlap_area(plot_set1_intra[i], plot_sets_inter[i])
        # kl2 = utils.kl_dist(plot_set2_intra[i], plot_sets_inter[i])
        # ol2 = utils.overlap_area(plot_set2_intra[i], plot_sets_inter[i])

        print("KL(set1 || inter_set): ", kl1)
        print("OA(set1, inter1): ", ol1)
        # print("KL(set2 || inter_set): ", kl2)
        # print("OA(set2, inter1): ", ol2)
        # output[metric] = [mean, std, kl1, ol1, kl2, ol2]
        df_output['attribute'].append(metric)
        df_output['KL'].append(kl1)
        df_output['OA'].append(ol1)
    df = pd.DataFrame(df_output)
    avg = {'attribute': 'avg', 'KL': df['KL'].mean(), 'OA': df['OA'].mean()}
    avg_df = pd.DataFrame([avg])
    df = pd.concat([df, avg_df], ignore_index=True)
    dfs.append(df)
    print("-----------------------------")
    print("Calculating MusPy Absolute Metrics (Separately)...")
    # 使用与 mgeval 相同的采样数据 set1 和 set2
    muspy_run_df = get_muspy_metrics_df(set1, set2)
    muspy_dfs.append(muspy_run_df)

# Concatenate the values of the 'KL' and 'OA' columns across all dataframes
kl_values = np.column_stack([df['KL'] for df in dfs])
oa_values = np.column_stack([df['OA'] for df in dfs])

# Calculate the mean and standard deviation for 'KL' and 'OA' columns
kl_mean = np.mean(kl_values, axis=1)
kl_std = np.std(kl_values, axis=1)
oa_mean = np.mean(oa_values, axis=1)
oa_std = np.std(oa_values, axis=1)

# Create new dataframes for mean and standard deviation
mean_df = pd.DataFrame({'attribute': df['attribute'], 'KL': kl_mean, 'OA': oa_mean})
std_df = pd.DataFrame({'attribute': df['attribute'], 'KL': kl_std, 'OA': oa_std})

# save statistics into a .txt file for reading
if args.savename is None:
    dataset = args.set1dir.split('/')[-1]
    model = args.set2dir.split('/')[-1]
    print(f'结果保存至：{save_dir}/mean.csv')
    mean_output_csv = f'{save_dir}/mean.csv'
    std_output_csv = f'{save_dir}/std.csv'
    muspy_output_csv = f'{save_dir}/muspy.csv'
else:
    mean_output_csv = f'{save_dir}/{args.savename}_mean.csv'
    std_output_csv = f'{save_dir}/{args.savename}_std.csv'
    muspy_output_csv = f'{save_dir}/{args.savename}_muspy.csv'

mean_df.to_csv(mean_output_csv, index=False)
std_df.to_csv(std_output_csv, index=False)

if muspy_dfs:
    # 堆叠多次 Run 的结果
    # muspy_dfs[0] 是第一次 run 的 DataFrame，包含 attribute, Set1_Mean, Set2_Mean
    # 我们提取所有 runs 的 Set1_Mean 和 Set2_Mean 列
    set1_means_values = np.column_stack([d['Set1_Mean'].values for d in muspy_dfs])
    set2_means_values = np.column_stack([d['Set2_Mean'].values for d in muspy_dfs])

    # 计算多次 Run 的平均值和标准差
    final_muspy_df = pd.DataFrame({
        'attribute': muspy_dfs[0]['attribute'],
        'Set1_Mean': np.mean(set1_means_values, axis=1),
        'Set1_Std': np.std(set1_means_values, axis=1),
        'Set2_Mean': np.mean(set2_means_values, axis=1),
        'Set2_Std': np.std(set2_means_values, axis=1)
    })
    
    # 保存到新的 CSV 文件
    print(f"Saving MusPy results to: {muspy_output_csv}")
    final_muspy_df.to_csv(muspy_output_csv, index=False)

print('Saved output to {} folder.'.format(args.outdir))
print('Evaluation Complete.')