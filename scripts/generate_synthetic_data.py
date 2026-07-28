"""
生成合成多组学数据 (用于演示和单元测试)
==============================================
生成代谢组 + 转录组 + 蛋白组的合成数据,
含已知的潜变量结构和噪声。

输出: data/synthetic_multiblock.npz
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chemocalib.models.mbpls import generate_toy_multiblock_data


def main():
    print("生成合成多组学数据...")

    # 三组数据规模
    blocks, y, feature_names = generate_toy_multiblock_data(
        n_samples=200,
        n_metabolites=80,
        n_transcripts=500,
        n_proteins=120,
        noise=0.08,
    )

    # 保存
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(output_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(output_dir, "synthetic_multiblock.npz"),
        X_metabolome=blocks[0],
        X_transcriptome=blocks[1],
        X_proteome=blocks[2],
        y=y,
    )

    # 特征名
    with open(os.path.join(output_dir, "feature_names.txt"), "w") as f:
        for i, names in enumerate(feature_names):
            f.write(f"Block_{i}: {','.join(names[:10])}... ({len(names)} features)\n")

    print(f"  X1 (代谢组): {blocks[0].shape}")
    print(f"  X2 (转录组): {blocks[1].shape}")
    print(f"  X3 (蛋白组): {blocks[2].shape}")
    print(f"  Y (响应):    {y.shape}")
    print(f"  已保存到 {output_dir}/synthetic_multiblock.npz")


if __name__ == "__main__":
    main()
