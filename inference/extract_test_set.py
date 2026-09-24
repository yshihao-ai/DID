import argparse
import os
import sys
import json
import shutil
from pathlib import Path
from loguru import logger
from tqdm import tqdm

# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description="根据 metadata.json 直接提取测试集 MIDI 文件 (不进行任何处理)")
    
    # 必需参数
    parser.add_argument("--metadata_path", type=str, required=True, 
                        help="metadata.json 文件的路径，用于读取 'test' 列表")
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="包含原始 MIDI 文件的根目录 (Ground Truth 源目录)")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="提取出的测试集保存目录")
    
    args = parser.parse_args()

    # 1. 加载 Metadata 并获取测试集列表
    logger.info(f"正在加载 Metadata: {args.metadata_path}")
    try:
        with open(args.metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        # 提取测试集文件名列表
        if 'test' in metadata:
            test_filenames = set(metadata['test'])  # 转为集合，查询更快
            logger.info(f"Metadata 中包含 {len(test_filenames)} 个测试样本")
        else:
            logger.error("Metadata 中未找到 'test' 键，无法提取测试集！")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"加载 Metadata 失败: {e}")
        sys.exit(1)

    # 2. 扫描输入目录下的所有 MIDI 文件
    logger.info(f"正在扫描输入目录: {args.input_dir}")
    input_path = Path(args.input_dir)
    
    # 递归查找所有 .mid 和 .midi 文件
    all_files = list(input_path.rglob("*.mid")) + list(input_path.rglob("*.midi"))
    logger.info(f"源目录中找到 {len(all_files)} 个 MIDI 文件")

    # 3. 筛选文件
    filtered_files = []
    for f in all_files:
        # 获取文件名（不含后缀），例如 "path/to/909.mid" -> "909"
        # 逻辑必须与 metadata 中的命名规则一致
        # logger.info(f.stem)
        print(f)
        print(str(f.stem))

        if f.stem in test_filenames:
            test_filenames.remove(f.stem)
            filtered_files.append(f)
    logger.info(test_filenames)
    if len(filtered_files) == 0:
        logger.error("未在输入目录中找到 Metadata 指定的测试集文件！请检查路径或文件名匹配规则。")
        sys.exit(1)

    logger.info(f"筛选后匹配到 {len(filtered_files)} 个测试集文件")

    # 4. 执行复制操作
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"开始复制文件到: {args.output_dir}")
    
    success_count = 0
    error_count = 0

    # 使用 tqdm 显示进度
    for src_file in tqdm(filtered_files, unit="file"):
        try:
            # 计算相对路径，保持原始目录结构
            # 例如: input_dir/pop909/001.mid -> output_dir/pop909/001.mid
            # 这样可以防止如果不同子文件夹下有同名文件发生冲突
            rel_path = src_file.relative_to(input_path)
            dest_file = output_path / rel_path
            dest_file = output_path / src_file.name
            # 确保目标文件的父目录存在
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 复制文件 (copy2 保留文件元数据，如时间戳)
            shutil.copy2(src_file, dest_file)
            success_count += 1
            
        except Exception as e:
            logger.error(f"复制失败: {src_file} -> {e}")
            error_count += 1

    # 5. 总结
    logger.info("=" * 30)
    logger.info(f"处理完成！")
    logger.info(f"总计匹配: {len(filtered_files)}")
    logger.info(f"成功复制: {success_count}")
    if error_count > 0:
        logger.warning(f"复制失败: {error_count}")
    logger.info(f"文件已保存在: {args.output_dir}")

if __name__ == "__main__":
    main()
