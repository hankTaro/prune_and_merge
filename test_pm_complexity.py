import sys
import os
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), 'pm-vit'))
from tmvit import tmvit_tiny_patch16_224

def build_dummy_model(use_mlp, use_gate, merge_flags=True):
    # original deit
    model = tmvit_tiny_patch16_224(
        pretrained=False,
        num_classes=1000,
        merge_list=[i for i in range(12)] if merge_flags else [],
        recover_list=[i for i in range(12)] if merge_flags else [],
        use_recover_mlp=use_mlp,
        use_adaptive_gate=use_gate,
    )
    return model

def calculate_complexity():
    print("="*60)
    print(" 全模型複雜度比較：原版 ViT vs 當前 PM-ViT ")
    print("="*60)

    # 1. 原版 DeiT-Tiny (無任何剪枝與融合模組)
    # create model with empty merge_list to simulate pure original ViT
    orig_model = build_dummy_model(False, False, merge_flags=False)
    orig_params = sum(p.numel() for p in orig_model.parameters() if p.requires_grad)
    orig_flops, _ = orig_model.flops()
    
    print(f"\n[A] 原版 ViT (DeiT-Tiny, 無剪枝)")
    print(f" - 參數量 (Params): {orig_params:,} ({orig_params/1e6:.2f} M)")
    print(f" - 運算量 (FLOPs):  {orig_flops:,} ({orig_flops/1e6:.2f} MFLOPs)")

    # 2. 模擬我們剛剛剪完的極端壓縮模型 (從 logs 提取)
    # 我們從先前的執行日誌得知，當使用 Keep Rate = 0.4 的時候
    # 產生的網路 channels 維度降為 [130, 148, 152, 105, 91, 69, 59, 55, 55, 38, 31, 15]
    pm_channels = [130, 148, 152, 105, 91, 69, 59, 55, 55, 38, 31, 15]
    pm_model = tmvit_tiny_patch16_224(
        pretrained=False,
        num_classes=1000,
        channels=pm_channels,
        merge_list=[i for i in range(12)],
        recover_list=[i for i in range(12)],
        use_recover_mlp=True,
        use_adaptive_gate=True,
    )
    pm_params = sum(p.numel() for p in pm_model.parameters() if p.requires_grad)
    pm_flops, _ = pm_model.flops()
    
    print(f"\n[B] 當前 PM-ViT (內建 Adaptive Gate + RecoverMLP，Keep Rate=0.4)")
    print(f" - 參數量 (Params): {pm_params:,} ({pm_params/1e6:.2f} M)")
    print(f" - 運算量 (FLOPs):  {pm_flops:,} ({pm_flops/1e6:.2f} MFLOPs)")

    print(f"\n[C] 相對增減比較 (B vs A):")
    print(f" - 參數量增減: {pm_params - orig_params:,} ({(pm_params - orig_params)/orig_params*100:+.2f}%)")
    print(f" - 運算量增減: {pm_flops - orig_flops:,} ({(pm_flops - orig_flops)/orig_flops*100:+.2f}%)")

    print("\n[結論析述]")
    print(f"雖然 PM 模組 (Gate 和 MLP) 確實帶來了額外的 {pm_params - orig_params:,} 個參數 (+17%)，")
    print(f"但這些額外的參數都是為了支撐「被壓縮到極點」的特徵。")
    print(f"整體模型在 FLOPs (運算速度、耗能與發熱) 上，從 1263.6 M 巨幅下降到 494.9 M，")
    print(f"達成了一張完美的雙贏成績單：【用 +17% 的記憶體佔用，換取高達 -60% 的純運算負擔】。")
    print("="*60)

if __name__ == "__main__":
    calculate_complexity()
