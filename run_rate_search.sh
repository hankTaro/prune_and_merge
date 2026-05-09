#!/bin/bash

# 小規模 6 組 keep_rate 與 prune_rate 的探索實驗
# 統一使用: --use_adaptive_gate 與 --use_recover_mlp (終極完全體)
# Epoch: 10 快速測試

source /home/hank/miniconda3/bin/activate torch_env

# 定義 6 組參數 (格式: "prune_rate keep_rate 實驗名稱")
# 這 6 組涵蓋了從保守到極限壓縮的各個象限
experiments=(
    "0.1 0.6 exp_P10_K60_Conservative"  # 1. 保守派：僅刪除 10% 廢物，保留高達 60% 中心 (高 FLOPs)
    "0.25 0.5 exp_P25_K50_Moderate"     # 2. 溫和派：標準刪除 25%，保留 50% 中心
    "0.25 0.4 exp_P25_K40_Standard"     # 3. 標準派：您先前的配置基準
    "0.1 0.4 exp_P10_K40_HeavyMerge"    # 4. 融合派：低剪枝 (10%)，但中心少 (40%) -> 迫使高達 50% 特徵進行軟融合
    "0.4 0.4 exp_P40_K40_HeavyPrune"    # 5. 激進派：高剪枝 (40%)，中心少 (40%) -> 放棄大量資訊，僅融合 20%
    "0.4 0.25 exp_P40_K25_Extreme"      # 6. 極限壓縮：超狂配置，保留中心僅 25% (預計 FLOPs 大崩盤)
)

BASE_CMD="python pm-vit/finetune.py --prune --model tmvit_tiny_patch16_224"
RESUME_MODEL="/home/hank/prune_and_merge/deit_tiny_patch16_224-a1311bcf.pth"
DATA_PATH="/home/hank/imagenet"
EPOCHS=10

echo "開始 P_rate 與 K_rate 的 6 組網格搜索 (Grid Search)..."

for exp in "${experiments[@]}"; do
    read -r P_RATE K_RATE NAME <<< "$exp"
    echo "========================================================="
    echo " 開始執行實驗: $NAME "
    echo " 參數: Prune Rate = $P_RATE, Keep Rate = $K_RATE"
    echo "========================================================="
    
    OUT_DIR="/home/hank/prune_and_merge/output/rate_search/$NAME"
    mkdir -p $OUT_DIR
    
    # 動態計算合理 Warmup
    WARMUP_EPOCHS=$((EPOCHS / 5))
    if [ $WARMUP_EPOCHS -lt 1 ]; then WARMUP_EPOCHS=1; fi

    # 1. 執行剪枝 Calibration
    echo "  [1/3] 正在執行剪枝 (Pruning Calibration)..."
    $BASE_CMD --prune \
        --resume $RESUME_MODEL \
        --data-path $DATA_PATH \
        --batch-size 256 \
        --prune_rate $P_RATE --keep_rate $K_RATE \
        --use_recover_mlp --use_adaptive_gate \
        --output_dir $OUT_DIR 
        
    # 2. 執行微調 Fine-tuning
    echo "  [2/3] 正在執行微調 (Fine-tuning for $EPOCHS Epochs)..."
    python pm-vit/finetune.py \
        --model tmvit_tiny_patch16_224 \
        --resume $OUT_DIR/pruned_${P_RATE}_${K_RATE}.pth \
        --teacher-path $RESUME_MODEL --teacher-model deit_tiny_patch16_224 --distillation-type soft \
        --data-path $DATA_PATH \
        --batch-size 256 \
        --final_finetune $EPOCHS \
        --warmup-epochs $WARMUP_EPOCHS \
        --prune_rate $P_RATE --keep_rate $K_RATE \
        --use_recover_mlp --use_adaptive_gate \
        --lr 1e-4 \
        --output_dir $OUT_DIR 
    
    echo "實驗 $NAME 訓練結束！開始執行自動 Eval 評估並紀錄報告..."
    
    # 自動使用 best.pth 打 eval 來取得最終最準確的 FLOPs 與 Acc 跑分
    python pm-vit/finetune.py --eval --model tmvit_tiny_patch16_224 \
        --resume $OUT_DIR/best.pth \
        --data-path $DATA_PATH \
        --batch-size 256 \
        --use_recover_mlp --use_adaptive_gate > $OUT_DIR/eval_report.txt
    
    echo "$NAME 的評估報告已儲存至 $OUT_DIR/eval_report.txt"
    echo ""
done

echo "全數 6 組測試已跑完！您可以逐一檢視 output/rate_search/ 下的 eval_report.txt 來比較算力與準確率了。"
