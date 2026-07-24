""" Improved probabilistic embedding loss for cross-modal retrieval

PCME++
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClosedFormSampledDistanceLoss(nn.Module):
    def __init__(
        #初始化对应yaml中的criterion
            self,
            init_shift=5,
            init_negative_scale=5,
            vib_beta=0,
            smoothness_alpha=0,
            prob_distance='csd',
            **kwargs):
        super().__init__()

        shift = init_shift * torch.ones(1) #init_shift：控制匹配 logit 的整体偏移，shift 控制“判定为匹配”的基准线。
        negative_scale = init_negative_scale * torch.ones(1) #negative_scale 也是可学习参数。控制距离对 logit 的影响强度

        shift = nn.Parameter(shift)
        negative_scale = nn.Parameter(negative_scale)

        self.register_parameter('shift', shift)
        self.register_parameter('negative_scale', negative_scale)

        self.vib_beta = vib_beta
        self.smoothness_alpha = smoothness_alpha #控制 pseudo-positive smoothness loss 的权重。

        # XXX Do not specify prob_distance unless for the prob dist ablation study
        self.prob_distance = prob_distance

        self.bceloss = nn.BCEWithLogitsLoss()
        #支持两种概率距离：csd和wdist
        if self.prob_distance not in {'csd', 'wdist'}:
            raise ValueError(f'Invalid prob_distance. Expected ("csd", "wdist"), but {prob_distance=}')

    def max_violation_on(self):
        warnings.warn(
            'PCME loss does not support max violation. Nothing happens')
        return

    def max_violation_off(self):
        warnings.warn(
            'PCME loss does not support max violation. Nothing happens')
        return
    #VIB 正则，使用Gaussian KL,做了限制防止梯度爆炸
    def kl_divergence(self, mu, logsigma):
        kl_loss = -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).mean()
        if kl_loss > 10000:
            # XXX prevent loss exploration
            warnings.warn(f'Detected a VIB loss explosion ({kl_loss=} > 10000). Ignore the VIB loss for stability.')
            return 0
        return kl_loss

    #匹配 loss，matched 是标签矩阵
    def _recompute_matched(self, matched, logits, smoothness=0):
        """ Recompute the `matched` matrix if the smoothness value is given.
        """
        if not smoothness:
            return matched, None
        
        #有 smoothness，如果模型认为某个“负样本”比官方正样本还匹配，那它可能不是真负样本，而是 false negative。

        else:
            logits = logits.view(matched.size())
            # XXX Warning: all negative pairs will return weird results
            gt_labels, gt_indices = torch.max(matched, dim=1) #取官方正样本标签和索引
            gt_vals = logits[:, gt_indices].diag() #取官方正样本 logit
            pseudo_gt_indices = (logits >= gt_vals.unsqueeze(1)) #如果某个负样本的 logit 大于等于官方正样本 logit，则认为它是伪正样本
            new_matched = (gt_labels.unsqueeze(1) * (pseudo_gt_indices)) #如果是伪正样本，则保留官方正样本标签，否则为 0
            _matched = matched.clone() # 复制 matched 矩阵
            _matched[pseudo_gt_indices] = new_matched[pseudo_gt_indices] #将伪正样本的标签更新为官方正样本标签

            return _matched, torch.sum(pseudo_gt_indices).item() - len(gt_indices)

    #
    def _compute_prob_matching_loss(
            self, logits, matched, smoothness=0, pair_weight=None,
            soft_target=None):
        matched, n_pseudo_gts = self._recompute_matched(
            matched, logits, smoothness)
        if soft_target is not None:
            if soft_target.shape != logits.shape:
                raise ValueError(
                    'soft_target and logits must have identical shapes, got '
                    f'{soft_target.shape=} and {logits.shape=}')
            matched = soft_target.to(device=logits.device, dtype=logits.dtype)

        if pair_weight is None:
            loss = self.bceloss(logits, matched)
        else:
            if pair_weight.shape != logits.shape:
                raise ValueError(
                    'pair_weight and logits must have identical shapes, got '
                    f'{pair_weight.shape=} and {logits.shape=}')
            elementwise = F.binary_cross_entropy_with_logits(
                logits, matched.to(logits.dtype), reduction='none')
            pair_weight = pair_weight.to(
                device=logits.device, dtype=logits.dtype)
            loss = (elementwise * pair_weight).sum() / \
                pair_weight.sum().clamp_min(1.0)

        return {
            'loss': loss,
            'n_pseudo_gts': n_pseudo_gts,
        }

    #PCME++ 默认使用的概率距离。D(i, j) = ||μ_i - μ_j||² + sum(σ_i² + σ_j²)
    def _compute_closed_form_loss(
            self, input1, input2, matched, smoothness=0,
            pair_weight=None, soft_target=None):
        """ Closed-form probabilistic matching loss -- See Eq (1) and (2) in the paper.
        """
        mu_pdist = ((input1['mean'].unsqueeze(1) - input2['mean'].unsqueeze(0)) ** 2).sum(-1)
        sigma_pdist = ((torch.exp(input1['std']).unsqueeze(1) + torch.exp(input2['std']).unsqueeze(0))).sum(-1)
        logits = mu_pdist + sigma_pdist
        logits = -self.negative_scale * logits + self.shift
        loss_dict = self._compute_prob_matching_loss(
            logits, matched, smoothness=smoothness,
            pair_weight=pair_weight, soft_target=soft_target)
        loss_dict['loss/mu_pdist'] = mu_pdist.mean()
        loss_dict['loss/sigma_pdist'] = sigma_pdist.mean()
        return loss_dict

    def _compute_wd_loss(
            self, input1, input2, matched, smoothness=0,
            pair_weight=None, soft_target=None):
        """ Wasserstien loss (only used for the ablation study)
        """
        #概率距离
        mu_pdist = ((input1['mean'].unsqueeze(1) - input2['mean'].unsqueeze(0)) ** 2).sum(-1).view(-1)
        #sigma 距离
        sigma_pdist = ((torch.exp(input1['std'] / 2).unsqueeze(1) - torch.exp(input2['std'] / 2).unsqueeze(0)) ** 2).sum(-1).view(-1)
        #混合距离，distance 转成 logit
        logits = mu_pdist + sigma_pdist
        logits = logits.reshape(len(input1['mean']), len(input2['mean']))
        logits = -self.negative_scale * logits + self.shift
        loss_dict = self._compute_prob_matching_loss(
            logits, matched, smoothness=smoothness,
            pair_weight=pair_weight, soft_target=soft_target)
        loss_dict['loss/mu_pdist'] = mu_pdist.mean()
        loss_dict['loss/sigma_pdist'] = sigma_pdist.mean()
        return loss_dict
    #主流程函数
    def forward(
            self, img_emb, cap_emb, matched=None, pair_weight=None,
            soft_target=None):
        if self.prob_distance == 'wdist':
            loss_fn = self._compute_wd_loss
        else:
            loss_fn = self._compute_closed_form_loss
        vib_loss = 0
        
        #计算图像和文本两边的 KL
        if self.vib_beta != 0:
            vib_loss =\
                self.kl_divergence(img_emb['mean'], img_emb['std']) + \
                self.kl_divergence(cap_emb['mean'], cap_emb['std'])

        if matched is None:
            matched = torch.eye(len(img_emb['mean'])).to(img_emb['mean'].device)

        loss = loss_fn(
            img_emb, cap_emb, matched=matched,
            pair_weight=pair_weight, soft_target=soft_target)
        # NOTE: Efficient implementation for
        # when i2t loss and t2i loss are the same (https://github.com/naver-ai/pcme/issues/3)
        loss = 2 * loss['loss'] + self.vib_beta * vib_loss

        loss_dict = {
            'loss/loss': loss,
            'criterion/shift': self.shift,
            'criterion/negative_scale': self.negative_scale,
        }

        if self.vib_beta != 0:
            loss_dict['loss/vib_loss'] = vib_loss

        if self.smoothness_alpha:
            smooth_i2t_loss = loss_fn(img_emb, cap_emb, matched=matched, smoothness=self.smoothness_alpha)
            smooth_t2i_loss = loss_fn(cap_emb, img_emb, matched=matched.T, smoothness=self.smoothness_alpha)
            loss = loss + self.smoothness_alpha * (smooth_i2t_loss['loss'] + smooth_t2i_loss['loss'])
            loss_dict['loss/loss'] = loss
            loss_dict['loss/n_pseudo_gts'] = smooth_i2t_loss['n_pseudo_gts'] + smooth_t2i_loss['n_pseudo_gts']

        return loss, loss_dict
