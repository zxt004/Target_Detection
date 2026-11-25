import os
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from dataset_module import LargeTargetDataset, create_dataloader, collate_fn
from model import build_model
from loss import MultiModalLoss  # 导入完整的损失函数
from evaluation import LargeTargetEvaluator  # 导入评估器
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
import argparse
import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class LargeTargetTrainer:
    """大目标检测训练器"""

    def __init__(self, model, train_loader, val_loader, optimizer, lr_scheduler,
                 criterion, config, device, rank=0, world_size=1):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.config = config
        self.device = device
        self.rank = rank
        self.world_size = world_size

        # 创建输出目录
        self.output_dir = Path(config['output_dir'])
        if self.rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化TensorBoard（仅在主进程）
        if self.rank == 0:
            self.writer = SummaryWriter(log_dir=self.output_dir / 'logs')

        # 训练状态
        self.current_epoch = 0
        self.best_map = 0.0
        self.global_step = 0

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0

        # 只在主进程显示进度条
        if self.rank == 0:
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        else:
            progress_bar = self.train_loader

        for batch_idx, batch in enumerate(progress_bar):
            # 移动到设备
            images = {k: v.to(self.device) for k, v in batch['images'].items()}
            text_inputs = {
                level: {k: v.to(self.device) for k, v in inputs.items()}
                for level, inputs in batch['text_inputs'].items()
            }
            targets = batch['targets']

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images, text_inputs)

            # 计算损失
            loss_dict = self.criterion(outputs, targets)
            loss = loss_dict['total_loss']

            # 反向传播
            loss.backward()

            # 梯度裁剪
            if self.config['training'].get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['grad_clip']
                )

            # 更新参数
            self.optimizer.step()

            # 更新进度条
            total_loss += loss.item()
            if self.rank == 0:
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'avg_loss': f"{total_loss / (batch_idx + 1):.4f}"
                })

            # 记录到TensorBoard（仅在主进程）
            if self.rank == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                for k, v in loss_dict.items():
                    self.writer.add_scalar(f'train/{k}', v.item(), self.global_step)

            self.global_step += 1

        return total_loss / len(self.train_loader)

    def validate(self):
        """在验证集上评估模型"""
        if self.rank != 0:
            return 0.0, {}

        self.model.eval()

        # 初始化评估器
        evaluator = LargeTargetEvaluator(
            config=self.config,
            model_path=None,  # 不使用检查点，直接使用当前模型
            device=self.device
        )

        # 替换评估器的模型为当前模型
        evaluator.model = self.model

        # 执行评估
        results = evaluator.evaluate()

        # 获取mAP
        mAP = results['overall']['mAP']

        return mAP, results

    def save_checkpoint(self, is_best=False):
        """保存检查点"""
        if self.rank != 0:
            return

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.module.state_dict() if hasattr(self.model,
                                                                          'module') else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'best_map': self.best_map,
            'config': self.config
        }

        # 保存最新检查点
        torch.save(checkpoint, self.output_dir / 'checkpoint_latest.pth')

        # 保存最佳检查点
        if is_best:
            torch.save(checkpoint, self.output_dir / 'checkpoint_best.pth')

    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_map = checkpoint['best_map']

    def train(self):
        """完整训练流程"""
        if self.rank == 0:
            print(f"Starting training for {self.config['training']['epochs']} epochs")

        for epoch in range(self.current_epoch, self.config['training']['epochs']):
            self.current_epoch = epoch

            # 设置数据加载器的epoch（用于分布式训练）
            if hasattr(self.train_loader, 'sampler') and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            # 训练一个epoch
            train_loss = self.train_epoch()

            # 更新学习率
            self.lr_scheduler.step()

            # 在验证集上评估
            if epoch % self.config['training']['eval_interval'] == 0:
                val_map, val_results = self.validate()

                # 记录到TensorBoard（仅在主进程）
                if self.rank == 0:
                    self.writer.add_scalar('val/mAP', val_map, epoch)

                    # 保存最佳模型
                    if val_map > self.best_map:
                        self.best_map = val_map
                        self.save_checkpoint(is_best=True)
                        print(f"New best mAP: {val_map:.4f}")

                    # 保存验证结果
                    with open(self.output_dir / f'val_results_epoch_{epoch}.json', 'w') as f:
                        json.dump(val_results, f)

            # 保存检查点
            if epoch % self.config['training']['save_interval'] == 0 and self.rank == 0:
                self.save_checkpoint()

            if self.rank == 0:
                print(f"Epoch {epoch} completed. Train loss: {train_loss:.4f}, Val mAP: {val_map:.4f}")

        # 训练完成
        if self.rank == 0:
            self.writer.close()
            print(f"Training completed. Best mAP: {self.best_map:.4f}")


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
    parser.add_argument('--config', type=str, required=True,
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
    criterion = MultiModalLoss(config)

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