#!/bin/bash
# Ablation Study Script: 10 Epochs Quick Test
# Usage: bash run_ablation_10_epochs.sh [mode]
# Modes: baseline, gate, mlp, both

set -e # Exit on error

if [ -z "$1" ]; then
    echo "錯誤! 請提供要執行的測試組別。"
    echo "用法: bash $0 [baseline | gate | mlp | both]"
    exit 1
fi

MODE=$1

# Ensure conda env is active
source /home/hank/miniconda3/bin/activate torch_env

BASE_CMD="python pm-vit/finetune.py"
MODEL="tmvit_tiny_patch16_224"
RESUME="/home/hank/prune_and_merge/deit_tiny_patch16_224-a1311bcf.pth"
DATA_PATH="/home/hank/imagenet"
BATCH_SIZE=256
PRUNE_RATE=0.25
KEEP_RATE=0.4
EPOCHS=10
WARMUP=1

echo "========================================"
echo " 準備執行消融實驗: $MODE"
echo "========================================"

if [ "$MODE" == "baseline" ]; then
    DIR="output/ablation_baseline"
    mkdir -p $DIR
    echo ">> 執行剪枝..."
    $BASE_CMD --prune --model $MODEL --resume $RESUME --data-path $DATA_PATH --batch-size $BATCH_SIZE --prune_mode attn --prune_rate $PRUNE_RATE --keep_rate $KEEP_RATE --output_dir $DIR --no_recover_mlp
    echo ">> 執行微調..."
    $BASE_CMD --model $MODEL --resume $DIR/pruned_${PRUNE_RATE}_${KEEP_RATE}.pth --teacher-path $RESUME --teacher-model deit_tiny_patch16_224 --data-path $DATA_PATH --batch-size $BATCH_SIZE --distillation-type soft --prune_mode attn --final_finetune $EPOCHS --warmup-epochs $WARMUP --lr 1e-4 --output_dir $DIR --no_recover_mlp

elif [ "$MODE" == "gate" ]; then
    DIR="output/ablation_gate_only"
    mkdir -p $DIR
    echo ">> 執行剪枝..."
    $BASE_CMD --prune --model $MODEL --resume $RESUME --data-path $DATA_PATH --batch-size $BATCH_SIZE --prune_mode attn --prune_rate $PRUNE_RATE --keep_rate $KEEP_RATE --output_dir $DIR --use_adaptive_gate --no_recover_mlp
    echo ">> 執行微調..."
    $BASE_CMD --model $MODEL --resume $DIR/pruned_${PRUNE_RATE}_${KEEP_RATE}.pth --teacher-path $RESUME --teacher-model deit_tiny_patch16_224 --data-path $DATA_PATH --batch-size $BATCH_SIZE --distillation-type soft --prune_mode attn --final_finetune $EPOCHS --warmup-epochs $WARMUP --lr 1e-4 --output_dir $DIR --use_adaptive_gate --no_recover_mlp

elif [ "$MODE" == "mlp" ]; then
    DIR="output/ablation_mlp_only"
    mkdir -p $DIR
    echo ">> 執行剪枝..."
    $BASE_CMD --prune --model $MODEL --resume $RESUME --data-path $DATA_PATH --batch-size $BATCH_SIZE --prune_mode attn --prune_rate $PRUNE_RATE --keep_rate $KEEP_RATE --output_dir $DIR --use_recover_mlp
    echo ">> 執行微調..."
    $BASE_CMD --model $MODEL --resume $DIR/pruned_${PRUNE_RATE}_${KEEP_RATE}.pth --teacher-path $RESUME --teacher-model deit_tiny_patch16_224 --data-path $DATA_PATH --batch-size $BATCH_SIZE --distillation-type soft --prune_mode attn --final_finetune $EPOCHS --warmup-epochs $WARMUP --lr 1e-4 --output_dir $DIR --use_recover_mlp

elif [ "$MODE" == "both" ]; then
    DIR="output/ablation_both"
    mkdir -p $DIR
    echo ">> 執行剪枝..."
    $BASE_CMD --prune --model $MODEL --resume $RESUME --data-path $DATA_PATH --batch-size $BATCH_SIZE --prune_mode attn --prune_rate $PRUNE_RATE --keep_rate $KEEP_RATE --output_dir $DIR --use_adaptive_gate --use_recover_mlp
    echo ">> 執行微調..."
    $BASE_CMD --model $MODEL --resume $DIR/pruned_${PRUNE_RATE}_${KEEP_RATE}.pth --teacher-path $RESUME --teacher-model deit_tiny_patch16_224 --data-path $DATA_PATH --batch-size $BATCH_SIZE --distillation-type soft --prune_mode attn --final_finetune $EPOCHS --warmup-epochs $WARMUP --lr 1e-4 --output_dir $DIR --use_adaptive_gate --use_recover_mlp

else
    echo "未知的模式: $MODE"
    echo "可用模式: baseline, gate, mlp, both"
    exit 1
fi

echo "========================================"
echo " $MODE 實驗完成！"
echo " 結果已儲存於: $DIR"
echo "========================================"
