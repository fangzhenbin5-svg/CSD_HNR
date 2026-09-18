# E15-neighbor015 实验记录

本实验基于 CSD-HNR 的 PCME++/CLIP ViT-B/32 图文检索模型，使用 COCO 2014
训练数据。代码在本仓库，完整参数见
[`configs/pcmepp_e15_neighbor015.yaml`](../configs/pcmepp_e15_neighbor015.yaml)。

## 相对 E12 的目标改动

`lambda_neighbor` 由 `0.020` 降到 `0.015`（降低 25%）。
`candidate_centrality_weight=0.20`、`neighbor_rho=0.10`、
`soft_label_max=0.30`、`same_modal_threshold=0.47`、
`direct_similarity_threshold=0.18` 和 `negative_min_weight=0.30` 保持不变。
训练共 25 个 epoch，学习率在 epoch 15 切换为原来的 0.1 倍，
SWA 从 epoch 16 开始。

## 结果

下表来自本次运行的 TensorBoard 标量记录。普通模型按 ECCV 平均 mAP@R
选出的最佳 checkpoint 为 epoch 18；SWA 是完成 25 个 epoch 后的单独验证，
不是该 checkpoint 的指标。

| 指标 | 普通模型最佳（epoch 18） | SWA |
|---|---:|---:|
| ECCV mAP@R 平均 | 40.1764 | 40.2812 |
| ECCV R-Precision 平均 | 49.8609 | 49.9836 |
| COCO 5K R@1 平均 | 53.6460 | 53.7680 |
| ECCV I2T mAP@R | 31.7646 | 31.9732 |
| ECCV T2I mAP@R | 48.5883 | 48.5893 |

相对 E12，E15 的普通模型平均 mAP@R 从约 `40.05` 提高到 `40.18`；
SWA 达到 `40.28`。这些是单次运行结果，未固定相同随机种子重复试验，
不能据此把小幅提升解释为稳定的因果增益。

## 复现

在仓库根目录、已安装依赖并准备好 COCO/CxC 标注后运行：

```bash
python train.py ./configs/pcmepp_e15_neighbor015.yaml \
  --dataloader__data_path /path/to/coco \
  --train__expname ./results_csd_hnr_e15_neighbor015
```

配置里的数据目录已与原运行的 TensorBoard `hparams.yaml` 核对：
`/home/fangzb/data/datasets/coco`。迁移到其他机器时，请按实际目录覆盖。
原运行未记录固定随机种子，因此上述命令保证方法与超参数相同，不保证逐位复现。

模型 checkpoint、COCO 图片和 TensorBoard 事件文件没有上传到普通 Git 仓库。
E15 的 checkpoint 单文件约 2.6 GB，超过 GitHub 普通 Git 文件大小限制；
如需分享权重，应另行使用 Git LFS 或模型存储服务。
