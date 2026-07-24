"""History-free hubness calibration and neighborhood ranking for CSD."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HubnessNeighborRegularizer(nn.Module):
    """Feature-queue regularizer without persistent pair pseudo-labels.

    The queue contains recent detached embeddings only.  No pair IDs,
    accepted/unknown states, or dataset-sized history tables are maintained.
    """

    def __init__(self, cfg, embed_dim):
        super().__init__()
        self.cfg = cfg or {}
        self.enabled = bool(self.cfg.get('enable', False))
        self.embed_dim = int(embed_dim)
        self.queue_size = int(self.cfg.get('queue_size', 8192))
        if self.enabled and self.queue_size < 1:
            raise ValueError('hubness_neighbor.queue_size must be positive')

        # Queues are deliberately excluded from checkpoints.  Features from an
        # older model state are stale and are rebuilt automatically on resume.
        self.register_buffer(
            'image_queue',
            torch.zeros(self.queue_size, self.embed_dim, dtype=torch.float16),
            persistent=False)
        self.register_buffer(
            'text_queue',
            torch.zeros(self.queue_size, self.embed_dim, dtype=torch.float16),
            persistent=False)
        self.register_buffer(
            'queue_ptr', torch.zeros((), dtype=torch.long), persistent=False)
        self.register_buffer(
            'queue_count', torch.zeros((), dtype=torch.long), persistent=False)

        self.register_buffer(
            'image_centrality_mean', torch.zeros((), dtype=torch.float32))
        self.register_buffer(
            'image_centrality_var', torch.ones((), dtype=torch.float32))
        self.register_buffer(
            'text_centrality_mean', torch.zeros((), dtype=torch.float32))
        self.register_buffer(
            'text_centrality_var', torch.ones((), dtype=torch.float32))
        self.register_buffer(
            'centrality_updates', torch.zeros((), dtype=torch.long))

    def _warmup_factor(self, epoch):
        start_epoch = int(self.cfg.get('start_epoch', 5))
        warmup_epochs = int(self.cfg.get('warmup_epochs', 5))
        if int(epoch) < start_epoch:
            return 0.0
        if warmup_epochs <= 0:
            return 1.0
        return min(1.0, (int(epoch) - start_epoch + 1) / warmup_epochs)

    def _soft_label_factor(self, epoch):
        start_epoch = int(self.cfg.get('soft_label_start_epoch', 8))
        warmup_epochs = int(self.cfg.get('soft_label_warmup_epochs', 5))
        if int(epoch) < start_epoch:
            return 0.0
        if warmup_epochs <= 0:
            return 1.0
        return min(1.0, (int(epoch) - start_epoch + 1) / warmup_epochs)

    def _active_queue(self, queue):
        count = int(self.queue_count.item())
        if count < self.queue_size:
            return queue[:count]
        return queue

    @torch.no_grad()
    def enqueue(self, image_mean, text_mean):
        if not self.enabled:
            return
        image_mean = F.normalize(
            image_mean.detach().float(), dim=-1).to(torch.float16)
        text_mean = F.normalize(
            text_mean.detach().float(), dim=-1).to(torch.float16)
        batch_size = image_mean.size(0)
        if batch_size >= self.queue_size:
            self.image_queue.copy_(image_mean[-self.queue_size:])
            self.text_queue.copy_(text_mean[-self.queue_size:])
            self.queue_ptr.zero_()
            self.queue_count.fill_(self.queue_size)
            return

        ptr = int(self.queue_ptr.item())
        first = min(batch_size, self.queue_size - ptr)
        self.image_queue[ptr:ptr + first] = image_mean[:first]
        self.text_queue[ptr:ptr + first] = text_mean[:first]
        remaining = batch_size - first
        if remaining:
            self.image_queue[:remaining] = image_mean[first:]
            self.text_queue[:remaining] = text_mean[first:]
        self.queue_ptr.fill_((ptr + batch_size) % self.queue_size)
        self.queue_count.fill_(
            min(self.queue_size, int(self.queue_count.item()) + batch_size))

    @torch.no_grad()
    def _centrality(self, image_mean, text_mean):
        image_bank = self._active_queue(self.image_queue)
        text_bank = self._active_queue(self.text_queue)
        image = F.normalize(image_mean.detach().float(), dim=-1)
        text = F.normalize(text_mean.detach().float(), dim=-1)
        # Cross-modal centrality matches the role of candidate variance in CSD:
        # a high value means that the candidate is broadly similar to queries
        # from the other modality.
        image_centrality = (
            image @ text_bank.float().t()).mean(dim=1)
        text_centrality = (
            text @ image_bank.float().t()).mean(dim=1)
        return image_centrality, text_centrality

    @torch.no_grad()
    def _normalize_centrality(self, image_centrality, text_centrality):
        momentum = float(self.cfg.get('centrality_ema_momentum', 0.99))
        if not 0 <= momentum < 1:
            raise ValueError(
                'centrality_ema_momentum must be in [0, 1)')

        image_batch_mean = image_centrality.mean()
        text_batch_mean = text_centrality.mean()
        image_batch_var = image_centrality.var(unbiased=False).clamp_min(1e-6)
        text_batch_var = text_centrality.var(unbiased=False).clamp_min(1e-6)
        if int(self.centrality_updates.item()) == 0:
            self.image_centrality_mean.copy_(image_batch_mean)
            self.text_centrality_mean.copy_(text_batch_mean)
            self.image_centrality_var.copy_(image_batch_var)
            self.text_centrality_var.copy_(text_batch_var)
        else:
            one_minus = 1.0 - momentum
            self.image_centrality_mean.mul_(momentum).add_(
                image_batch_mean, alpha=one_minus)
            self.text_centrality_mean.mul_(momentum).add_(
                text_batch_mean, alpha=one_minus)
            self.image_centrality_var.mul_(momentum).add_(
                image_batch_var, alpha=one_minus)
            self.text_centrality_var.mul_(momentum).add_(
                text_batch_var, alpha=one_minus)
        self.centrality_updates.add_(1)

        z_clip = float(self.cfg.get('centrality_z_clip', 3.0))
        image_z = (
            (image_centrality - self.image_centrality_mean)
            / self.image_centrality_var.sqrt().clamp_min(1e-6)
        ).clamp(-z_clip, z_clip)
        text_z = (
            (text_centrality - self.text_centrality_mean)
            / self.text_centrality_var.sqrt().clamp_min(1e-6)
        ).clamp(-z_clip, z_clip)
        return image_z, text_z

    @staticmethod
    def _log_total_variance(log_variance):
        return torch.logsumexp(log_variance.float(), dim=-1)

    @staticmethod
    def _pearson(x, y):
        x = x.detach().float().flatten()
        y = y.detach().float().flatten()
        x = x - x.mean()
        y = y - y.mean()
        denominator = x.square().sum().sqrt() * y.square().sum().sqrt()
        if denominator <= 1e-12:
            return x.new_tensor(0.0)
        return (x * y).sum() / denominator

    def _variance_calibration(
            self, image_logvar, text_logvar, image_z, text_z):
        var_min = float(self.cfg.get('total_var_min', 0.01))
        var_max = float(self.cfg.get('total_var_max', 0.30))
        temperature = float(self.cfg.get('centrality_temperature', 1.0))
        if not 0 < var_min < var_max:
            raise ValueError('Expected 0 < total_var_min < total_var_max')
        if temperature <= 0:
            raise ValueError('centrality_temperature must be positive')

        with torch.no_grad():
            image_target = var_min + (var_max - var_min) * torch.sigmoid(
                image_z / temperature)
            text_target = var_min + (var_max - var_min) * torch.sigmoid(
                text_z / temperature)
        image_pred_log = self._log_total_variance(image_logvar)
        text_pred_log = self._log_total_variance(text_logvar)
        image_loss = F.smooth_l1_loss(
            image_pred_log, image_target.log())
        text_loss = F.smooth_l1_loss(
            text_pred_log, text_target.log())
        raw_loss = 0.5 * (image_loss + text_loss)
        logs = {
            'hnr/sigma_raw': raw_loss.detach(),
            'hnr/image_sigma_loss': image_loss.detach(),
            'hnr/text_sigma_loss': text_loss.detach(),
            'hnr/image_var_pred': image_pred_log.detach().exp().mean(),
            'hnr/text_var_pred': text_pred_log.detach().exp().mean(),
            'hnr/image_var_target': image_target.mean(),
            'hnr/text_var_target': text_target.mean(),
            'hnr/image_sigma_centrality_corr': self._pearson(
                image_pred_log, image_z),
            'hnr/text_sigma_centrality_corr': self._pearson(
                text_pred_log, text_z),
        }
        return raw_loss, logs

    def _directional_target(
            self, direct_similarity, same_modal_similarity, candidate_z,
            positive_mask, rho):
        batch_size = direct_similarity.size(0)
        negative_mask = ~positive_mask
        if batch_size < 2 or not negative_mask.any():
            empty = direct_similarity.new_zeros(direct_similarity.shape)
            gt = positive_mask.to(direct_similarity.dtype)
            gt = gt / gt.sum(dim=1, keepdim=True).clamp_min(1.0)
            return gt, empty, direct_similarity.new_zeros(batch_size)

        topk = min(
            int(self.cfg.get('neighbor_topk', 20)), batch_size - 1)
        centrality_weight = float(
            self.cfg.get('candidate_centrality_weight', 0.10))
        ranking_score = (
            0.5 * direct_similarity
            + 0.5 * same_modal_similarity
            - centrality_weight * candidate_z.unsqueeze(0)
        )
        ranking_score = ranking_score.masked_fill(
            ~negative_mask, -torch.inf)
        top_indices = ranking_score.topk(topk, dim=1).indices
        candidate_mask = torch.zeros_like(positive_mask)
        candidate_mask.scatter_(1, top_indices, True)
        candidate_mask &= negative_mask

        gate_temperature = float(
            self.cfg.get('evidence_temperature', 0.05))
        if gate_temperature <= 0:
            raise ValueError('evidence_temperature must be positive')
        same_threshold = float(
            self.cfg.get('same_modal_threshold', 0.50))
        direct_threshold = float(
            self.cfg.get('direct_similarity_threshold', 0.20))
        gate = torch.sigmoid(
            (same_modal_similarity - same_threshold) / gate_temperature
        ) * torch.sigmoid(
            (direct_similarity - direct_threshold) / gate_temperature)
        gate = gate * candidate_mask.to(gate.dtype)

        teacher_temperature = float(
            self.cfg.get('teacher_temperature', 0.05))
        if teacher_temperature <= 0:
            raise ValueError('teacher_temperature must be positive')
        stable_score = ranking_score.masked_fill(
            ~candidate_mask, -torch.inf)
        row_max = stable_score.max(dim=1, keepdim=True).values
        row_max = torch.where(
            torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
        neighbor_weight = torch.exp(
            (ranking_score - row_max) / teacher_temperature
        ) * gate
        neighbor_distribution = neighbor_weight / neighbor_weight.sum(
            dim=1, keepdim=True).clamp_min(1e-8)

        gt_distribution = positive_mask.to(direct_similarity.dtype)
        gt_distribution = gt_distribution / gt_distribution.sum(
            dim=1, keepdim=True).clamp_min(1.0)
        row_confidence = gate.max(dim=1).values
        row_rho = (float(rho) * row_confidence).clamp(0.0, 0.95)
        target = (
            (1.0 - row_rho).unsqueeze(1) * gt_distribution
            + row_rho.unsqueeze(1) * neighbor_distribution
        )
        return target, gate, row_rho

    def _neighborhood_loss(
            self, image_mean, text_mean, image_logvar, text_logvar,
            matched, image_z, text_z, negative_scale, shift, rho,
            soft_label_factor):
        image = F.normalize(image_mean.float(), dim=-1)
        text = F.normalize(text_mean.float(), dim=-1)
        positive_mask = matched.detach() > 0.5

        direct = image @ text.t()
        with torch.no_grad():
            detached_direct = direct.detach()
            image_relation = image.detach() @ image.detach().t()
            text_relation = text.detach() @ text.detach().t()
            i2t_target, i2t_gate, i2t_rho = self._directional_target(
                detached_direct, text_relation, text_z,
                positive_mask, rho)
            t2i_target, t2i_gate, t2i_rho = self._directional_target(
                detached_direct.t(), image_relation, image_z,
                positive_mask.t(), rho)

        image_total_var = self._log_total_variance(image_logvar).exp()
        text_total_var = self._log_total_variance(text_logvar).exp()
        mean_distance = (
            (image_mean.float().unsqueeze(1)
             - text_mean.float().unsqueeze(0)).square().sum(dim=-1)
        )
        csd_distance = (
            mean_distance + image_total_var.unsqueeze(1)
            + text_total_var.unsqueeze(0)
        )
        logits = shift.float() - negative_scale.float() * csd_distance
        student_temperature = float(
            self.cfg.get('student_temperature', 1.0))
        if student_temperature <= 0:
            raise ValueError('student_temperature must be positive')
        i2t_loss = -(
            i2t_target * F.log_softmax(
                logits / student_temperature, dim=1)
        ).sum(dim=1).mean()
        t2i_loss = -(
            t2i_target * F.log_softmax(
                logits.t() / student_temperature, dim=1)
        ).sum(dim=1).mean()
        raw_loss = 0.5 * (i2t_loss + t2i_loss)

        with torch.no_grad():
            # A suspected relation softens BCE if either retrieval direction
            # provides evidence.  It is never relabeled as a hard positive.
            relation_confidence = torch.maximum(
                i2t_gate, t2i_gate.t())
            min_weight = float(
                self.cfg.get('negative_min_weight', 0.20))
            pair_weight = 1.0 - (1.0 - min_weight) * relation_confidence
            pair_weight = torch.where(
                positive_mask, torch.ones_like(pair_weight), pair_weight)

            soft_target = None
            soft_label_enabled = bool(
                self.cfg.get('soft_label_enable', False))
            soft_label_max = float(self.cfg.get('soft_label_max', 0.30))
            if not 0.0 <= soft_label_max <= 1.0:
                raise ValueError('soft_label_max must be in [0, 1]')
            if soft_label_enabled and soft_label_factor > 0:
                # Soft labels are stricter than pair reweighting: they require
                # evidence from both retrieval directions.  A high one-way
                # score can reduce negative punishment, but it cannot become
                # a weak positive unless the reverse direction agrees.
                bidirectional_confidence = torch.sqrt(
                    (i2t_gate * t2i_gate.t()).clamp_min(0.0))
                soft_value = (
                    soft_label_max * float(soft_label_factor)
                    * bidirectional_confidence)
                base_target = matched.detach().to(
                    device=soft_value.device, dtype=soft_value.dtype)
                soft_target = (
                    base_target + (1.0 - base_target) * soft_value
                ).clamp(0.0, 1.0)
                soft_target = torch.where(
                    positive_mask, base_target, soft_target)
            else:
                bidirectional_confidence = torch.zeros_like(
                    relation_confidence)

        logs = {
            'hnr/neighbor_raw': raw_loss.detach(),
            'hnr/i2t_neighbor_loss': i2t_loss.detach(),
            'hnr/t2i_neighbor_loss': t2i_loss.detach(),
            'hnr/i2t_rho_mean': i2t_rho.mean(),
            'hnr/t2i_rho_mean': t2i_rho.mean(),
            'hnr/relation_count': (
                relation_confidence > 0.5).sum(),
            'hnr/relation_confidence_mean':
                relation_confidence.mean(),
            'hnr/pair_weight_mean': pair_weight.mean(),
            'hnr/soft_label_factor': image_mean.new_tensor(
                float(soft_label_factor)),
            'hnr/soft_label_confidence_mean':
                bidirectional_confidence.mean(),
        }
        if soft_target is not None:
            negative_mask = ~positive_mask
            soft_values = soft_target.masked_select(negative_mask)
            logs['hnr/soft_label_mean'] = soft_values.mean()
            logs['hnr/soft_label_count'] = (
                soft_values > matched.detach().masked_select(
                    negative_mask).to(soft_values.dtype)
            ).sum()
        else:
            logs['hnr/soft_label_mean'] = image_mean.new_tensor(0.0)
            logs['hnr/soft_label_count'] = image_mean.new_tensor(0)
        return raw_loss, pair_weight, soft_target, logs

    def forward(
            self, image_emb, text_emb, matched, epoch,
            negative_scale, shift):
        zero = image_emb['mean'].sum() * 0.0
        pair_weight = torch.ones_like(
            matched, dtype=image_emb['mean'].dtype)
        soft_target = None
        logs = {
            'hnr/queue_count': image_emb['mean'].new_tensor(
                int(self.queue_count.item())),
            'hnr/warmup_factor': image_emb['mean'].new_tensor(0.0),
        }
        if not self.enabled:
            return zero, pair_weight, soft_target, logs

        factor = self._warmup_factor(epoch)
        min_queue = int(self.cfg.get('min_queue_size', 1024))
        ready = int(self.queue_count.item()) >= min_queue
        if factor <= 0 or not ready:
            self.enqueue(image_emb['mean'], text_emb['mean'])
            logs['hnr/queue_count'] = image_emb['mean'].new_tensor(
                int(self.queue_count.item()))
            return zero, pair_weight, soft_target, logs

        image_centrality, text_centrality = self._centrality(
            image_emb['mean'], text_emb['mean'])
        image_z, text_z = self._normalize_centrality(
            image_centrality, text_centrality)
        sigma_raw, sigma_logs = self._variance_calibration(
            image_emb['std'], text_emb['std'], image_z, text_z)

        rho = float(self.cfg.get('neighbor_rho', 0.10)) * factor
        soft_factor = self._soft_label_factor(epoch)
        neighbor_raw, pair_weight_full, soft_target, neighbor_logs = \
            self._neighborhood_loss(
                image_emb['mean'], text_emb['mean'],
                image_emb['std'], text_emb['std'],
                matched, image_z, text_z,
                negative_scale, shift, rho, soft_factor)
        # Pair softening follows the same schedule as the auxiliary losses.
        pair_weight = 1.0 - factor * (1.0 - pair_weight_full)

        sigma_weight = float(
            self.cfg.get('lambda_sigma', 0.01)) * factor
        neighbor_weight = float(
            self.cfg.get('lambda_neighbor', 0.05)) * factor
        sigma_loss = sigma_weight * sigma_raw
        neighbor_loss = neighbor_weight * neighbor_raw
        total = sigma_loss + neighbor_loss

        self.enqueue(image_emb['mean'], text_emb['mean'])
        logs.update(sigma_logs)
        logs.update(neighbor_logs)
        logs.update({
            'loss/hnr_sigma': sigma_loss,
            'loss/hnr_neighbor': neighbor_loss,
            'hnr/image_centrality_mean': image_centrality.mean(),
            'hnr/text_centrality_mean': text_centrality.mean(),
            'hnr/image_centrality_std':
                image_centrality.std(unbiased=False),
            'hnr/text_centrality_std':
                text_centrality.std(unbiased=False),
            'hnr/queue_count': image_emb['mean'].new_tensor(
                int(self.queue_count.item())),
            'hnr/warmup_factor': image_emb['mean'].new_tensor(factor),
            'hnr/soft_label_factor': image_emb['mean'].new_tensor(
                soft_factor),
            'hnr/rho_current': image_emb['mean'].new_tensor(rho),
            'hnr/lambda_sigma_current':
                image_emb['mean'].new_tensor(sigma_weight),
            'hnr/lambda_neighbor_current':
                image_emb['mean'].new_tensor(neighbor_weight),
        })
        return total, pair_weight, soft_target, logs
