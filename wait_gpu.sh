#!/bin/bash

########################################
# 配置区（只需要修改这里）
########################################

# 要监控的 GPU 编号
# 方式1：直接在这里写
GPU_IDS=(1 2 3)

# 方式2：也可以运行时传入，例如：
# bash wait_gpu.sh 0,1,2
if [ -n "$1" ]; then
    IFS=',' read -ra GPU_IDS <<< "$1"
fi

# 当已使用显存(MiB) <= 该值时开始执行
MEMORY_THRESHOLD=500

# 检查间隔（秒）
CHECK_INTERVAL=30

# 是否执行完后退出（true/false）
EXIT_AFTER_RUN=true

# ==========================
# 要执行的命令（支持多条）
# 注意：这里不要再写 CUDA_VISIBLE_DEVICES
# ==========================

COMMANDS=(
    #'python train.py --train_args_file train_args/did_Pop909_tiny.json'
    #'python train.py --train_args_file train_args/did_Pop1k7_tiny.json'
    #'python train.py --train_args_file train_args/did_Pop909_tiny_chord_plan.json'
    'python train.py --train_args_file train_args/did_Pop909_tiny_acc.json'
    # 'python train.py --train_args_file train_args/did_Pop909_tiny_acc_plan.json'
)

########################################


echo "Monitoring GPUs: ${GPU_IDS[*]}"
echo "Memory threshold: ${MEMORY_THRESHOLD} MiB"
echo

while true
do
    AVAILABLE_GPU=""

    NOW=$(date "+%F %T")

    for GPU_ID in "${GPU_IDS[@]}"
    do
        # 获取指定 GPU 已使用显存
        MEM_USED=$(nvidia-smi \
            --id="${GPU_ID}" \
            --query-gpu=memory.used \
            --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ')

        if [ -z "$MEM_USED" ]; then
            echo "${NOW} | GPU ${GPU_ID}: failed to query"
            continue
        fi

        echo "${NOW} | GPU ${GPU_ID}: ${MEM_USED} MiB used"

        # 找到第一张空闲卡就用它
        if [ -z "$AVAILABLE_GPU" ] && [ "$MEM_USED" -le "$MEMORY_THRESHOLD" ]; then
            AVAILABLE_GPU="$GPU_ID"
        fi
    done

    if [ -n "$AVAILABLE_GPU" ]; then
        echo
        echo "GPU ${AVAILABLE_GPU} is available!"
        echo "Using CUDA_VISIBLE_DEVICES=${AVAILABLE_GPU}"
        echo

        for cmd in "${COMMANDS[@]}"
        do
            echo "=================================================="
            echo "CUDA_VISIBLE_DEVICES=${AVAILABLE_GPU} $cmd"
            echo "=================================================="

            (
                export CUDA_VISIBLE_DEVICES="${AVAILABLE_GPU}"
                eval "$cmd"
            )

            STATUS=$?

            if [ $STATUS -ne 0 ]; then
                echo "Command failed! Exit code: $STATUS"
                exit $STATUS
            fi
        done

        echo
        echo "All commands finished."

        if [ "$EXIT_AFTER_RUN" = true ]; then
            exit 0
        fi
    fi

    echo
    sleep "$CHECK_INTERVAL"
done