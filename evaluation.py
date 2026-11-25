# evaluation.py
import torch
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch.nn.functional as F
from dataset_module import LargeTargetDataset
from model import build_model
import json
from pathlib import Path
from tqdm import tqdm
import argparse
import yaml


class LargeTargetEvaluator:
    """大目标检测评估器"""

    def __init__(self, config, model_path, device='cuda'):
        self.config = config
        self.device = torch.device(device)

        # 加载模型
        self.model = build_model(
            num_classes=config['model']['num_classes'],
            backbone_name=config['model']['backbone'],
            text_encoder_name=config['model']['text_encoder']
        )

        # 加载检查点
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # 加载验证集
        self.dataset = LargeTargetDataset(
            data_path=config['data']['path'],
            annotations_path=config['data']['val_annotations'],
            text_descriptions_path=config['data']['text_descriptions'],
            transform=None,
            is_train=False
        )

        # COCO评估工具
        self.coco_gt = COCO(config['data']['val_annotations'])
        self.cat_ids = self.coco_gt.getCatIds()

    def evaluate(self, output_file=None):
        """执行评估"""
        results = []

        # 遍历数据集
        for idx in tqdm(range(len(self.dataset))):
            data = self.dataset[idx]
            images = {k: v.unsqueeze(0).to(self.device) for k, v in data['images'].items()}
            text_inputs = data['text_inputs']

            # 移动到设备
            for level in text_inputs:
                for key in text_inputs[level]:
                    text_inputs[level][key] = text_inputs[level][key].to(self.device)

            # 执行推理
            with torch.no_grad():
                outputs = self.model(images, text_inputs)

            # 处理输出
            pred_logits = outputs['pred_logits'][0]
            pred_boxes = outputs['pred_boxes'][0]

            # 应用softmax获取类别概率
            probs = F.softmax(pred_logits, dim=-1)

            # 获取最高概率的类别
            scores, labels = torch.max(probs, dim=-1)

            # 过滤背景类
            keep = (labels < self.config['model']['num_classes']) & (scores > 0.05)

            if not keep.any():
                continue

            # 获取有效检测结果
            boxes = pred_boxes[keep].cpu().numpy()
            scores = scores[keep].cpu().numpy()
            labels = labels[keep].cpu().numpy()

            # 转换边界框格式 [x1, y1, x2, y2] -> [x1, y1, w, h]
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

            # 转换为COCO格式
            image_id = data['targets']['image_id'].item()
            for i in range(len(boxes)):
                result = {
                    'image_id': image_id,
                    'category_id': self.cat_ids[labels[i]],
                    'bbox': boxes[i].tolist(),
                    'score': scores[i].item()
                }
                results.append(result)

        # 保存结果
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f)

        # COCO评估
        coco_dt = self.coco_gt.loadRes(results)
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')

        # 执行评估
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # 获取详细指标
        stats = coco_eval.stats
        metrics = {
            'mAP': stats[0],  # AP @ [IoU=0.50:0.95 | area=all | maxDets=100]
            'mAP_50': stats[1],  # AP @ [IoU=0.50 | area=all | maxDets=100]
            'mAP_75': stats[2],  # AP @ [IoU=0.75 | area=all | maxDets=100]
            'mAP_small': stats[3],  # AP @ [IoU=0.50:0.95 | area=small | maxDets=100]
            'mAP_medium': stats[4],  # AP @ [IoU=0.50:0.95 | area=medium | maxDets=100]
            'mAP_large': stats[5],  # AP @ [IoU=0.50:0.95 | area=large | maxDets=100]
            'AR_max1': stats[6],  # AR @ [maxDets=1 | area=all]
            'AR_max10': stats[7],  # AR @ [maxDets=10 | area=all]
            'AR_max100': stats[8],  # AR @ [maxDets=100 | area=all]
            'AR_small': stats[9],  # AR @ [maxDets=100 | area=small]
            'AR_medium': stats[10],  # AR @ [maxDets=100 | area=medium]
            'AR_large': stats[11]  # AR @ [maxDets=100 | area=large]
        }

        # 按类别输出AP
        class_metrics = {}
        for cat_id in self.cat_ids:
            coco_eval.params.catIds = [cat_id]
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            cat_name = self.coco_gt.loadCats(cat_id)[0]['name']
            class_metrics[cat_name] = coco_eval.stats[0]  # mAP

        return {
            'overall': metrics,
            'per_class': class_metrics
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='大目标检测评估')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--model', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--output', type=str, default='eval_results.json', help='评估结果输出文件')

    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 执行评估
    evaluator = LargeTargetEvaluator(config, args.model)
    results = evaluator.evaluate(args.output)

    print("\n===== 评估结果 =====")
    print(f"mAP: {results['overall']['mAP']:.4f}")
    print(f"mAP_50: {results['overall']['mAP_50']:.4f}")
    print(f"mAP_75: {results['overall']['mAP_75']:.4f}")
    print(f"大目标mAP: {results['overall']['mAP_large']:.4f}")

    print("\n===== 类别级结果 =====")
    for class_name, ap in results['per_class'].items():
        print(f"{class_name}: {ap:.4f}")