import math

import torch

from pcmepp.criterions.pcmepp import ClosedFormSampledDistanceLoss
from pcmepp.hubness_neighbor import HubnessNeighborRegularizer
from pcmepp.models.uncertainty import ScalarTotalVarianceHead


def test_scalar_total_variance_is_bounded_and_isotropic():
    head = ScalarTotalVarianceHead(
        input_dim=6,
        embed_dim=8,
        total_var_min=0.01,
        total_var_max=0.30,
        total_var_init=0.10,
    )
    features = torch.randn(5, 6, requires_grad=True)
    logvar = head(features)
    total = logvar.exp().sum(dim=-1)

    assert logvar.shape == (5, 8)
    assert torch.allclose(total, torch.full_like(total, 0.10), atol=1e-6)
    assert torch.allclose(logvar, logvar[:, :1].expand_as(logvar))
    total.sum().backward()
    assert features.grad is not None


def test_weighted_csd_keeps_positive_pairs_and_backpropagates():
    criterion = ClosedFormSampledDistanceLoss(
        vib_beta=0, smoothness_alpha=0, prob_distance='csd')
    image_mean = torch.nn.functional.normalize(
        torch.randn(4, 8), dim=-1).requires_grad_()
    text_mean = torch.nn.functional.normalize(
        torch.randn(4, 8), dim=-1).requires_grad_()
    logvar_value = math.log(0.10 / 8)
    image_logvar = torch.full(
        (4, 8), logvar_value, requires_grad=True)
    text_logvar = torch.full(
        (4, 8), logvar_value, requires_grad=True)
    matched = torch.eye(4)
    pair_weight = torch.ones(4, 4)
    pair_weight[0, 1] = 0.2

    loss, logs = criterion(
        {'mean': image_mean, 'std': image_logvar},
        {'mean': text_mean, 'std': text_logvar},
        matched=matched,
        pair_weight=pair_weight,
    )
    assert torch.isfinite(loss)
    assert torch.isfinite(logs['loss/loss'])
    loss.backward()
    assert image_mean.grad is not None
    assert text_logvar.grad is not None


def test_hnr_uses_feature_queue_without_pair_history():
    cfg = {
        'enable': True,
        'queue_size': 8,
        'min_queue_size': 4,
        'start_epoch': 0,
        'warmup_epochs': 0,
        'lambda_sigma': 0.01,
        'lambda_neighbor': 0.05,
        'neighbor_rho': 0.10,
        'neighbor_topk': 2,
        'total_var_min': 0.01,
        'total_var_max': 0.30,
        'same_modal_threshold': -1.0,
        'direct_similarity_threshold': -1.0,
    }
    regularizer = HubnessNeighborRegularizer(cfg, embed_dim=8)
    initial_image = torch.nn.functional.normalize(
        torch.randn(4, 8), dim=-1)
    initial_text = torch.nn.functional.normalize(
        torch.randn(4, 8), dim=-1)
    regularizer.enqueue(initial_image, initial_text)

    image_mean = torch.nn.functional.normalize(
        torch.randn(4, 8), dim=-1).requires_grad_()
    text_mean = torch.nn.functional.normalize(
        torch.randn(4, 8), dim=-1).requires_grad_()
    logvar_value = math.log(0.10 / 8)
    image_logvar = torch.full(
        (4, 8), logvar_value, requires_grad=True)
    text_logvar = torch.full(
        (4, 8), logvar_value, requires_grad=True)
    matched = torch.eye(4)

    loss, pair_weight, logs = regularizer(
        {'mean': image_mean, 'std': image_logvar},
        {'mean': text_mean, 'std': text_logvar},
        matched,
        epoch=0,
        negative_scale=torch.tensor([5.0], requires_grad=True),
        shift=torch.tensor([5.0], requires_grad=True),
    )

    assert torch.isfinite(loss)
    assert pair_weight.shape == matched.shape
    assert torch.allclose(pair_weight.diag(), torch.ones(4))
    assert (pair_weight <= 1.0).all()
    assert int(logs['hnr/queue_count'].item()) == 8
    assert 'history' not in regularizer.__dict__
    assert not any('queue' in key for key in regularizer.state_dict())
    loss.backward()
    assert image_mean.grad is not None
    assert image_logvar.grad is not None
