"""End-to-end training code for PCME++

PCME++
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""
import os
import fire #Google 的命令行工具库。把 Python 函数直接变成命令行接口。

import torch
from transformers import BertTokenizer

import lightning.pytorch as pl #PyTorch Lightning 管理训练流程
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary, TQDMProgressBar #保存模型 checkpoint，打印模型结构摘要，控制训练进度条三个功能
from lightning.pytorch.callbacks.early_stopping import EarlyStopping #导入早停 callback
from lightning.pytorch.strategies import DDPStrategy #DDP 分布式训练策略

from config import parse_config #导入配置解析函数。读取 YAML 配置文件，并支持命令行参数覆盖。
from logger import PCMEPPLogger #继承自 TensorBoardLogger，负责把 loss / 指标写到 TensorBoard。

from pcmepp.datasets import get_loaders, get_test_loader #导入数据加载器函数。get_loaders 用于训练和验证集，get_test_loader 用于测试集。
from pcmepp.engine import PCMEPPModel #导入 PCME++ 模型类。封装了模型结构、前向传播、损失计算和指标评估等功能。

# yaml文件,checkpoint路径，命令行其他参数
def main(config_path, load_from_checkpoint=None, **kwargs):
    """ The training and evaluation script for PCME++.
    This script supports the following features
    (1) loading configurations from the pre-defined configuration file.
    (2) overwriting the configuration file by commandline arguments.
    (3) re-starting the training from the checkpoint.
        (all other arguments will be ignored)

    You can add additional arguments on your configuration file.
    Usage:
        (1) option with value
            --<group_name>__<option_name> <value>
        (2) option with `true` flag
            --<group_name>__<option_name>

    For example, this command will update your lr to 0.0001
    ```
    python train.py ./configs/pcmepp.yaml --optim__lr 0.0001
    ```
    This command will activate early stopping
    ```
    python train.py ./configs/pcmepp.yaml --train__early_stoping
    ```

    Parameters
    ----------
    config_path          : str
                           The path to the configuration file.
                           Ignored if `loader_from_checkpoint` is given.
    load_from_checkpoint : str, optional
                           If given, re-start from the given checkpoint.
                           !Caution! It will ignore all other arguments.
    """
    # Load configuration
    if load_from_checkpoint:
        print(f'Resume from the previous weight {load_from_checkpoint=}')
        ckpt = torch.load(load_from_checkpoint, weights_only=False)
        config = ckpt['hyper_parameters']['opt'] #从 checkpoint 里取出之前保存的配置。
        for arg_key, arg_val in kwargs.items(): #遍历命令行传进来的额外参数。
            keys = arg_key.split('__') #命令行参数格式用双下划线分隔，表示嵌套的配置字典。这里把参数名拆成列表。
            n_keys = len(keys)

            _config = config
            for idx, _key in enumerate(keys):
                if n_keys - 1 == idx:
                    _config[_key] = arg_val
                else:
                    _config = _config[_key]
    
    else:
        config = parse_config(config_path,
                              strict_cast=False,
                              **kwargs)

    # Load Tokenizer and Vocabulary
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab = tokenizer.vocab

    # Data loader
    train_loader, val_loader = get_loaders(
        **config.dataloader, tokenizer=tokenizer, opt=config, vocab_size=len(vocab))
 
    
    #构造测试集 loader
    te_loader = get_test_loader('testall', 'coco', tokenizer,
                                config.dataloader.eval_batch_size, config.dataloader.workers, config, len(vocab))

    #把验证 loader 和测试 loader 放到一个列表里。
    val_loader = [val_loader, te_loader]
    if config.train.get('skip_eval'):
        val_loader = None

    # Define model
    model = PCMEPPModel(config)

    # Model checkpoint options checkpoint 保存目录，如果没有配置 expname，默认，保存在root_dir = "./results"
    root_dir = config.train.get('expname', './results')

    checkpoint_callback_epoch = ModelCheckpoint(
        dirpath=root_dir,
        filename='model-{epoch:02d}-{eval_avg/eccv_map_at_r:.2f}-{eval_avg/coco_5k_r1:.2f}',
        verbose=True,
        save_last=True, #保存最后一个模型，生成last.ckpt
        save_top_k=1, #保存最好的模型
        # 本实验的首要目标是修复漏标正样本的完整排序，因此按
        # ECCV Caption mAP@R 保存最佳模型，而不是按原COCO RSUM保存。
        monitor='eval_avg/eccv_map_at_r',
        mode='max', #最大化验证指标
    )
    
    #训练开始时打印模型结构摘要,最多显示到 2 层模块
    model_summary_callback = ModelSummary(max_depth=2)
    #进度条 callback
    progress_bar = TQDMProgressBar(refresh_rate=config.train.pbar_step)
    #callback 列表
    callbacks = [
        checkpoint_callback_epoch,
        model_summary_callback,
        progress_bar,
    ]

    if config.train.get('early_stopping'):
        callbacks.append(EarlyStopping(monitor='val/rsum', mode='max'))
    #设置训练策略 strategy
    if config.train.get('strategy'):
        strategy = config.train.strategy
    else:
        strategy = DDPStrategy(
            # No way to avoid find_unused_parameters=True. https://github.com/pytorch/pytorch/issues/22049#issuecomment-505617666
            find_unused_parameters=True, #允许模型里有些参数在某些 forward 中没有被用到。
            gradient_as_bucket_view=True, #这是 DDP 的显存存优化参数。它让梯度作为 bucket 的 view 存储，可以减少显存占用
        )
    #创建 Lightning 训练器。
    trainer = pl.Trainer(
        strategy=strategy,
        callbacks=callbacks,
        logger=PCMEPPLogger(
            save_dir=os.path.join(root_dir, 'logs'),
            default_hp_metric=False
        ),
        precision=config.train.precision, #训练精度，支持 16、32、64 位浮点数。
        gradient_clip_val=config.train.grad_clip, #梯度裁剪阈值，防止梯度爆炸。
        log_every_n_steps=config.train.log_step, #每隔多少步记录一次日志。
        max_epochs=config.train.train_epochs, #训练轮数
        num_nodes=int(config.train.get('world_size', 1)), #设置训练节点数量。
        accumulate_grad_batches=1, #梯度累积步数。这里的1指不做梯度累积。
        check_val_every_n_epoch=1, #每隔多少个 epoch 验证一次。
        benchmark=True, #启用 cudnn benchmark 模式，自动寻找最优卷积算法。
        default_root_dir=root_dir, #默认的根目录，用于保存日志和模型。
        num_sanity_val_steps=0, #验证集的样本数量，用于训练前的快速验证。
    )
    #调用 Lightning 的训练入口。
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=load_from_checkpoint,
        weights_only=False if load_from_checkpoint else None, #如果不是从 checkpoint 恢复，就传 None
    )
    #SWA 模型验证，一般用于训练后期平均多个 epoch 的权重，提高泛化性能。
    if model.swa_enabled:
        model.print('evaluate by SWA')
        model.eval_by_swa = True
        trainer.validate(model, val_loader)


if __name__ == '__main__':
    fire.Fire(main)
