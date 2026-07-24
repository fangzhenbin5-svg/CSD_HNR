# CSD-HNR: Soft Label Version for PCME++

本工程基于 PCME++ 修改，用于研究 COCO 图文检索中 mAP@R / R-Precision 提升问题。

当前主线版本是：

```text
原始 PCME++ per-dim σ
+ CSD 概率距离
+ HNR 总方差校准
+ 疑似假负样本 soft label
```

目标不是更换 CLIP backbone，也不加入额外数据集，而是在原始训练集和 ViT-B/32 基座下，让模型更好处理“一图多文 / 一文多图”的语义相关样本。

## 当前版本：CSD-HNR SoftLabel E2

上一版 HNR 只做 soft negative weight：

```text
疑似假负样本仍然是 label = 0，只是降低 BCE 惩罚权重。
```

这可以减少误推远，但不会主动把疑似相关样本拉近。  
当前 E2 改为 soft label：

```text
官方正样本：label = 1.0
普通负样本：label = 0.0
疑似假负样本：label = 0 ~ soft_label_max
```

soft label 的计算方式：

```text
confidence = sqrt(i2t_gate * t2i_gate)
soft_label = soft_label_max * confidence
```

其中 `i2t_gate` 和 `t2i_gate` 分别表示图到文、文到图两个方向的邻域相关证据。  
使用双向一致置信度是为了避免单向高相似样本被过早当成弱正样本。

## 当前架构

```text
Image / Caption
    ↓
CLIP Image Encoder / CLIP Text Encoder
    ↓
Mean Head: μ_img, μ_txt
Uncertainty Head: per-dim logσ²_img, logσ²_txt
    ↓
CSD distance:
    D = ||μ_img - μ_txt||² + Σ exp(logσ²_img) + Σ exp(logσ²_txt)
    logit = shift - negative_scale * D
    ↓
Weighted BCE with optional soft target
```

HNR 分支：

```text
detached μ queue
    ↓
cross-modal centrality
    ↓
target total variance V_target ∈ [0.01, 0.30]
    ↓
SmoothL1(log V_pred, log V_target)
```

HNR 同时生成疑似假负样本的双向置信度：

```text
direct image-text similarity
+ same-modal neighborhood relation
+ candidate centrality penalty
    ↓
bidirectional confidence
    ↓
soft target for BCE
```

## 关键代码位置

| 功能 | 文件 |
|---|---|
| PCME++ CSD 主损失、soft target BCE | `pcmepp/criterions/pcmepp.py` |
| HNR 总方差校准与 soft label 生成 | `pcmepp/hubness_neighbor.py` |
| 训练时把 HNR 输出传入主损失 | `pcmepp/engine.py` |
| 当前实验配置 | `configs/pcmepp.yaml` |

## 当前主要参数

```yaml
criterion:
  prob_distance: csd
  vib_beta: 0
  smoothness_alpha: 0

model:
  scalar_total_variance: false

hubness_neighbor:
  enable: true
  lambda_sigma: 0.01
  lambda_neighbor: 0.02
  neighbor_rho: 0.10
  neighbor_topk: 20

  same_modal_threshold: 0.45
  direct_similarity_threshold: 0.15
  negative_min_weight: 0.30

  soft_label_enable: true
  soft_label_max: 0.30
  soft_label_start_epoch: 8
  soft_label_warmup_epochs: 5

train:
  expname: /home/16t/fangzb/CSD_HNR/results_csd_hnr_softlabel_e2
  train_epochs: 25
```

## Epoch 策略

| Epoch | 训练内容 |
|---:|---|
| 0–4 | 只训练原始 CSD 主损失 |
| 5–7 | 启动 HNR，总方差校准 + soft negative weight |
| 8–12 | soft label 线性启动 |
| 13–24 | CSD + HNR σ 校准 + soft label 全强度训练 |

HNR 强度：

```text
epoch 0-4: 0
epoch 5:   0.2
epoch 6:   0.4
epoch 7:   0.6
epoch 8:   0.8
epoch 9+:  1.0
```

soft label 强度：

```text
epoch 0-7: 0
epoch 8:   0.2
epoch 9:   0.4
epoch 10:  0.6
epoch 11:  0.8
epoch 12+: 1.0
```

## 启动训练

```bash
tmux new-session -d -s csd_hnr_softlabel_e2 'cd /home/16t/fangzb/CSD_HNR && CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 /home/16t/fangzb/miniconda3/envs/coco-omr/bin/python train.py ./configs/pcmepp.yaml'
```

查看训练：

```bash
tmux attach -t csd_hnr_softlabel_e2
```

## 需要重点观察的日志

```text
hnr/soft_label_factor
hnr/soft_label_count
hnr/soft_label_mean
hnr/soft_label_confidence_mean
hnr/pair_weight_mean
hnr/image_sigma_centrality_corr
hnr/text_sigma_centrality_corr
eval_avg/eccv_map_at_r
eval_avg/eccv_rprecision
eval_avg/coco_5k_r1
```

## 上一版结果参考

CSD-HNR per-dim σ + soft negative weight 版本最佳结果：

| 指标 | Best epoch 22 |
|---|---:|
| ECCV mAP@R avg | 39.58 |
| ECCV R-Precision avg | 49.32 |
| COCO 5K R@1 avg | 52.87 |
| COCO 5K R@5 avg | 80.34 |
| COCO 5K R@10 avg | 88.29 |

分方向：

| 指标 | Image→Text | Text→Image |
|---|---:|---:|
| ECCV mAP@R | 31.31 | 47.86 |
| ECCV R-Precision | 42.42 | 56.21 |
| COCO 5K R@1 | 59.76 | 45.98 |

该结果说明上一版主要问题是 Image→Text 方向的相关 caption 没有整体排进前 R，因此当前 E2 重点增强疑似假负样本的软相关监督。

## 当前判断

σ 总方差校准已经能缓解 σ 退化，但它不是 mAP@R / R-Precision 的主要提升来源。  
mAP@R / R-Precision 的核心矛盾是训练标签和 ECCV Caption 测试标签不一致：

```text
训练时多数语义相关样本被当作负样本；
测试时这些样本被认为是相关样本。
```

因此当前版本把 HNR 从“只降低负样本惩罚”改成“为高置信疑似假负样本提供 soft label”，让主 CSD BCE 直接学习语义相关集合排序。

## License

本工程基于 PCME++ 的 MIT License 修改。原始 PCME++ 来自 NAVER Cloud Corp.
