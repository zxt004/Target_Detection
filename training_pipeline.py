# training_pipeline.py
import os
import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from train import LargeTargetTrainer
from model import build_model
from loss import MultiModalLoss
from dataset_module import create_dataloader, collate_fn
import argparse
import yaml
from datetime import datetime


def setup_distributed():
    """设置分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(gpu)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        return rank, world_size, gpu
    else:
        return 0, 1, 0


def main():
    # 设置分布式训练
    rank, world_size, gpu = setup_distributed()

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='大目标检测训练')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 设置输出目录 (添加时间戳)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config['output_dir'] = f"{config['output_dir']}_{timestamp}"

    if rank == 0:
        print(f"训练配置: {config}")

    # 初始化模型
    model = build_model(
        num_classes=config['model']['num_classes'],
        backbone_name=config['model']['backbone'],
        text_encoder_name=config['model']['text_encoder']
    )

    # 分布式训练
    if world_size > 1:
        model = model.to(gpu)
        model = DDP(model, device_ids=[gpu])
    else:
        model = model.to(config['training']['device'])

    # 初始化数据加载器
    train_loader = create_dataloader(
        data_path=config['data']['path'],
        annotations_path=config['data']['train_annotations'],
        text_descriptions_path=config['data']['text_descriptions'],
        batch_size=config['training']['batch_size'] // world_size,
        is_train=True,
        num_workers=config['training']['num_workers'],
        distributed=(world_size > 1)
    )

    val_loader = create_dataloader(
        data_path=config['data']['path'],
        annotations_path=config['data']['val_annotations'],
        text_descriptions_path=config['data']['text_descriptions'],
        batch_size=config['training']['batch_size'] // world_size,
        is_train=False,
        num_workers=config['training']['num_workers'],
        distributed=(world_size > 1)
    )

    # 初始化优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )

    # 学习率调度器
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config['training']['lr_milestones'],
        gamma=config['training']['lr_gamma']
    )

    # 损失函数
    criterion = MultiModalLoss({
        'num_classes': config['model']['num_classes'],
        'focal_alpha': config['training'].get('focal_alpha', 0.25),
        'focal_gamma': config['training'].get('focal_gamma', 2.0),
        'det_weight': config['training']['loss_weights']['det'],
        'align_weight': config['training']['loss_weights']['align'],
        'consistency_weight': config['training']['loss_weights']['consistency'],
        'prompt_weight': config['training']['loss_weights']['prompt']
    })

    # 初始化训练器
    trainer = LargeTargetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        criterion=criterion,
        config=config,
        device=gpu if world_size > 1 else config['training']['device'],
        rank=rank,
        world_size=world_size
    )

    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()