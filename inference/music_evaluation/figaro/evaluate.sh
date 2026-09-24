#!/bin/bash

# ==========================================
# Figaro 评测一键运行脚本
# ==========================================

# 1. 检查参数输入 (修改为需要3个参数)
if [ "$#" -ne 3 ]; then
    echo "使用方法: $0 <预测MIDI文件夹路径> <答案(GT)MIDI文件夹路径> <输出CSV文件路径>"
    echo "示例: $0 /path/to/generated /path/to/ground_truth ./my_result.csv"
    exit 1
fi

GT_DIR=$1
PRED_DIR=$2
OUT_CSV=$3   # 将第三个参数设为输出的 CSV 路径
PYTHON_SCRIPT="evaluate.py"  # 请确保这个名字和你的 Python 脚本文件名一致

# 检查输入文件夹是否存在
if [ ! -d "$PRED_DIR" ] || [ ! -d "$GT_DIR" ]; then
    echo "错误: 指定的预测文件夹或答案文件夹不存在！"
    exit 1
fi

# 2. 构建临时评测目录结构
TEMP_WORKSPACE="./tmp_figaro_workspace"

echo "==== 正在构建评测目录结构 ===="
# 清理可能残留的旧文件夹并创建新结构
rm -rf "$TEMP_WORKSPACE"
mkdir -p "$TEMP_WORKSPACE/gt"

# 复制预测的 MIDI 到根目录 (忽略找不到特定后缀的报错)
cp "$PRED_DIR"/*.mid "$PRED_DIR"/*.midi "$TEMP_WORKSPACE/" 2>/dev/null || true

# 复制答案(GT)的 MIDI 到 gt 子目录
cp "$GT_DIR"/*.mid "$GT_DIR"/*.midi "$TEMP_WORKSPACE/gt/" 2>/dev/null || true

echo "文件复制完成！"

# 3. 设置环境变量并执行 Python 脚本
echo "==== 开始运行评测 ===="
export SAMPLE_DIR="$TEMP_WORKSPACE"
export OUT_FILE="$OUT_CSV"

# 如果你的预测文件名和真实文件名不完全一样（比如生成的是 song1_pred.mid，GT是 song1.mid）
# 你可以取消下面这两行的注释，并根据实际情况修改 SPLIT_STR
# export SPLIT_STR="_pred"
# export POST_STR="" 

# 执行 Python 脚本
python "$PYTHON_SCRIPT"

# 4. 扫尾工作
echo "==== 评测结束 ===="
if [ -f "$OUT_CSV" ]; then
    echo "✅ 结果已成功保存至: $OUT_CSV"
else
    echo "❌ 似乎没有生成结果，请检查 Python 脚本的输出日志。"
fi

# 清理临时文件夹 (如果想保留排查问题，可以注释掉这一行)
echo "正在清理临时文件夹..."
rm -rf "$TEMP_WORKSPACE"
echo "全部完成！"