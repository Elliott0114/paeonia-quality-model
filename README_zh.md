# 白芍饮片外观与化学质量分析

本仓库配套论文 *Disentangling appearance and multianalyte chemistry in Paeoniae Radix Alba decoction pieces using visible and hyperspectral imaging*，提供**单个真实样本、最终论文汇总结果和核心分析方法**。

研究包含 69 批次、207 份样本。此处仅公开 `SBS-01-G1` 的个体数据；完整原始图片、高光谱数据、全队列个体表及训练模型暂不公开。因此，本仓库用于展示方法、运行单样本处理和重绘汇总图表，不承诺独立复现全队列模型结果。

## 运行

在仓库根目录执行：

```bash
conda env create -f environment.yml
conda activate paeonia-quality-model
python scripts/run_example.py
python scripts/plot_results.py
```

单样本命令计算 12 个图像特征、合并 3 个已有机器视觉指标，并从实测 292 波段平均光谱得到 262 波段 P0–P3，与预期值核对。HPLC 数值和专家等级不参与特征提取，也不使用单样本估计模型性能。

绘图命令直接读取公开汇总 CSV，生成 Figure 2、Figure 3D–F、Figure 4、Figure 5 的 PNG 和 PDF。数值与论文一致，版式适配独立展示。Figure 1 及 Figure 3A–C 的完整重绘仍需要未公开的其他图像、光谱立方体或色谱曲线。生成文件位于 `outputs/`。

## 文件内容

- [论文表格与表注](paper_results/tables/README.md)：4 张主表；补充表 S1–S8，包含 17 个 CSV 分区、200 行数据。
- [汇总图数据](paper_results/figure_data/README.md)：包含 Figure 5D 全部 49 个成分—场次组合，保留负值及低于 −100% 的结果。
- [方法与函数接口](docs/METHODS.md)、[数据字典](docs/DATA_DICTIONARY.md)、[论文对应关系](docs/MANUSCRIPT_MAP.md)。

`paeonia/` 保留论文所用图像和光谱处理、分组建模、跨模态比较、面板统计、置换检验及 LOSO 方法。函数接受内存中的数组和表格；完整建模需要原队列数据。`r/secondary_methods.R` 保留补充表 S3 的 R 统计方法，依赖见 `r/renv.lock`。上述两个公开命令只需 Python。

外观分级、化学预测与跨场次表现分别报告。CAT 排除面板的事后敏感性分析属性及未获支持的结果均予以保留。本仓库不提供总质量评分，未新增许可证。
