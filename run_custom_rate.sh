#!/bin/bash

# ====================================================================
# 手動指定 Prune Rate 與 Keep Rate 進行模型剪枝訓練與評估的腳本
# ====================================================================

# 確保啟動 conda 環境
source /home/hank/miniconda3/bin/activate torch_env

# 檢查是否輸入足夠的參數
if [ $# -lt 2 ]; then
    echo "❌ 錯誤：請提供 prune_rate 和 keep_rate 參數！"
    echo "👉 用法: ./run_custom_rate.sh <prune_rate> <keep_rate> [epochs] [use_mlp] [use_gate]"
    echo "💡 範例 (跑10個Epoch, 預設開啟 MLP 與 Gate): ./run_custom_rate.sh 0.25 0.4 10"
    exit 1
fi

P_RATE=$1
K_RATE=$2

# 如果沒有輸入第三個參數 Epochs，預設為 10
EPOCHS=${3:-10}

# 支援開關 MLP 或 Gate (可選參數，預設都是開啟)
USE_MLP=${4:-"true"}
USE_GATE=${5:-"true"}

# 構建資料夾名稱 (例如 0.25 會變成 25；0.1 會變成 10)
P_NAME=$(echo "$P_RATE * 100" | bc | cut -d'.' -f1)
K_NAME=$(echo "$K_RATE * 100" | bc | cut -d'.' -f1)

# 若有特殊開關，也可以標註在檔名上
EXP_NAME="exp_P${P_NAME}_K${K_NAME}"
if [ "$USE_MLP" != "true" ]; then EXP_NAME="${EXP_NAME}_NoMLP"; fi
if [ "$USE_GATE" != "true" ]; then EXP_NAME="${EXP_NAME}_NoGate"; fi

OUT_DIR="/home/hank/prune_and_merge/output/custom_search/$EXP_NAME"

BASE_CMD="python pm-vit/finetune.py --prune --model tmvit_tiny_patch16_224"
RESUME_MODEL="/home/hank/prune_and_merge/deit_tiny_patch16_224-a1311bcf.pth"
DATA_PATH="/home/hank/imagenet"

echo "========================================================="
echo " 🚀 開始執行自訂實驗: $EXP_NAME"
echo " 📌 Prune Rate = $P_RATE "
echo " 📌 Keep Rate  = $K_RATE "
echo " ⏳ Epochs     = $EPOCHS"
echo " ⚙️  輸出路徑   = $OUT_DIR"
echo "========================================================="

mkdir -p $OUT_DIR

# 動態計算合理 Warmup (約佔總 Epoch 的 10~20%)
WARMUP_EPOCHS=$((EPOCHS / 10))
if [ $WARMUP_EPOCHS -lt 1 ]; then WARMUP_EPOCHS=1; fi

# 1. 準備剪枝指令
PRUNE_CMD="$BASE_CMD --prune --resume $RESUME_MODEL --data-path $DATA_PATH --batch-size 256 --prune_rate $P_RATE --keep_rate $K_RATE --output_dir $OUT_DIR"

if [ "$USE_MLP" == "true" ]; then PRUNE_CMD="$PRUNE_CMD --use_recover_mlp"; else PRUNE_CMD="$PRUNE_CMD --no_recover_mlp"; fi
if [ "$USE_GATE" == "true" ]; then PRUNE_CMD="$PRUNE_CMD --use_adaptive_gate"; fi

# 2. 準備微調指令 (載入剛剪枝完的模型繼續訓練，需要 Teacher Model 做 Distillation)
FINETUNE_CMD="python pm-vit/finetune.py --model tmvit_tiny_patch16_224 --resume $OUT_DIR/pruned_${P_RATE}_${K_RATE}.pth --teacher-path $RESUME_MODEL --teacher-model deit_tiny_patch16_224 --distillation-type soft --data-path $DATA_PATH --batch-size 256 --final_finetune $EPOCHS --warmup-epochs $WARMUP_EPOCHS --lr 1e-4 --output_dir $OUT_DIR"

if [ "$USE_MLP" == "true" ]; then FINETUNE_CMD="$FINETUNE_CMD --use_recover_mlp"; else FINETUNE_CMD="$FINETUNE_CMD --no_recover_mlp"; fi
if [ "$USE_GATE" == "true" ]; then FINETUNE_CMD="$FINETUNE_CMD --use_adaptive_gate"; fi

echo "[1/3] 正在執行剪枝 (Pruning Calibration)..."
eval $PRUNE_CMD

echo "[2/3] 正在執行微調 (Fine-tuning for $EPOCHS Epochs)..."
eval $FINETUNE_CMD

echo ""
echo " 實驗 $EXP_NAME 訓練結束！開始執行自動 Eval 評估..."

# 組裝 Eval 參數
EVAL_CMD="python pm-vit/finetune.py --eval --model tmvit_tiny_patch16_224 --resume $OUT_DIR/best.pth --data-path $DATA_PATH --batch-size 256"
if [ "$USE_MLP" == "true" ]; then EVAL_CMD="$EVAL_CMD --use_recover_mlp"; else EVAL_CMD="$EVAL_CMD --no_recover_mlp"; fi
if [ "$USE_GATE" == "true" ]; then  EVAL_CMD="$EVAL_CMD --use_adaptive_gate"; fi

# 2. 開始評估並輸出到 report
echo "[2/2] 正在執行評估與計算複雜度..."
eval $EVAL_CMD > "$OUT_DIR/eval_report.txt"

echo " $EXP_NAME 測試完畢！評估報告已儲存至: $OUT_DIR/eval_report.txt"
