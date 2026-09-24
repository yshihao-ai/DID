import os
import shutil
from datetime import datetime

def collect_and_zip_midis(source_folder_path):
    """
    1. 扫描 source_folder_path 下所有 midi 文件
    2. 提取路径中的关键参数名作为压缩包文件名
    3. 复制到同级的文件夹并压缩
    """
    
    # 1. 路径处理
    # 去除末尾可能存在的斜杠，确保 basename 能正确获取
    source_path = os.path.abspath(source_folder_path).rstrip(os.sep)
    
    if not os.path.exists(source_path):
        print(f"❌ 错误：找不到路径 '{source_path}'")
        return

    # --- 核心修改：智能提取实验名 ---
    # 获取当前文件夹名 (例如: song_predict)
    current_folder_name = os.path.basename(source_path)
    # 获取父文件夹名 (例如: did_Pop1k7_1024_preChord_...)
    parent_folder_name = os.path.basename(os.path.dirname(source_path))
    
    # 构造更具描述性的名字
    # 如果当前文件夹是常见的输出名（如 song_predict），我们主要使用父文件夹名（包含参数的那层）
    if current_folder_name in ['song_predict', 'midi', 'output', 'results']:
        experiment_name = parent_folder_name
    else:
        # 否则把两层名字拼起来，防止信息丢失
        experiment_name = f"{parent_folder_name}_{current_folder_name}"
    
    # 加上时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M") # 精确到分即可
    
    # 最终文件夹/压缩包名字：[实验参数名]_MidiCollection_[时间]
    target_folder_name = f"{experiment_name}_Midi_{timestamp}"
    
    # 目标路径：放在源文件夹的同级目录 (或者你可以改成放在源文件夹的上一级 os.path.dirname(source_path))
    # 这里我选择放在源文件夹的上一级目录，这样压缩包会在 `song_predict` 的旁边，而不是里面
    target_parent_dir = os.path.dirname(source_path)
    target_path = os.path.join(target_parent_dir, target_folder_name)

    # 创建目标文件夹
    if not os.path.exists(target_path):
        os.makedirs(target_path)
        print(f"📂 目标文件夹名已生成: {target_folder_name}")

    # 2. 遍历并复制文件
    midi_count = 0
    print(f"🚀 正在扫描: {source_path}")

    for root, dirs, files in os.walk(source_path):
        # 避免递归扫描到我们正在创建的目标文件夹
        if target_path in root:
            continue
            
        for file in files:
            # 检查后缀名 (不区分大小写)
            if file.lower().endswith(('.mid', '.midi')):
                source_file = os.path.join(root, file)
                
                # 保持文件名，处理重名
                target_file = os.path.join(target_path, file)
                if os.path.exists(target_file):
                    # 如果重名，加上子文件夹名作为前缀
                    sub_folder = os.path.basename(root)
                    if sub_folder != current_folder_name:
                        new_filename = f"{sub_folder}_{file}"
                    else:
                        new_filename = f"copy_{file}"
                    target_file = os.path.join(target_path, new_filename)
                
                try:
                    shutil.copy2(source_file, target_file)
                    midi_count += 1
                except Exception as e:
                    print(f"⚠️ 复制失败: {file} - {e}")

    print(f"✅ 提取完成: 共 {midi_count} 个 MIDI 文件。")

    if midi_count == 0:
        print("⚠️ 未找到任何 MIDI 文件，取消压缩。")
        os.rmdir(target_path)
        return

    # 3. 压缩文件夹
    print("📦 正在压缩...")
    zip_output_path = target_path  # shutil会自动加上 .zip
    
    try:
        shutil.make_archive(base_name=zip_output_path, format='zip', root_dir=target_path)
        print(f"\n🎉 成功！压缩包路径如下：")
        print(f"👉 {zip_output_path}.zip")
    except Exception as e:
        print(f"❌ 压缩失败: {e}")

    # 可选：压缩完删除临时文件夹 (取消注释以启用)
    # shutil.rmtree(target_path)

# --- 使用方法 ---
if __name__ == "__main__":
    # 输入你的长路径
    input_folder = r"/home/xxx/copy/xxx/DID/output/inference/did_Pop1k7_1024_preChord_attnPool_cho64_cpt_4804_msl2048_big_1024_1024_type512/song_predict/" 
    
    collect_and_zip_midis(input_folder)
