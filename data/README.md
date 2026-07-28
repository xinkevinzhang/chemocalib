# Data Directory

合成数据存放目录。

## 生成合成数据

```bash
python scripts/generate_synthetic_data.py
```

生成 `synthetic_multiblock.npz`:
- X_metabolome: (n_samples, n_metabolites)
- X_transcriptome: (n_samples, n_transcripts)
- X_proteome: (n_samples, n_proteins)
- y: (n_samples,) 响应变量

## GEM 模型

COBRApy 的 `test.create_test_model("textbook")` 自动提供 E. coli core model，
无需手动下载。如需 iMM904 酵母模型，代码会自动从 BIGG 下载。

## 真实数据

如有真实代谢组学数据，放入此目录并按如下格式组织：
- 代谢组: metabolome.csv (samples × metabolites)
- 转录组: transcriptome.csv (samples × genes)
- 蛋白组: proteome.csv (samples × proteins)
- 元数据: metadata.csv (samples × conditions)
