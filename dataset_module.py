#dataset_module.py
import os
import json
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPTokenizer, AutoTokenizer
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import random


class LargeTargetDataset(Dataset):
    """大目标检测数据集"""

    def __init__(self,
                 data_path: str,
                 annotations_path: str,
                 text_descriptions_path: str,
                 transform=None,
                 is_train: bool = True,
                 max_size: int = 1024):

        self.data_path = Path(data_path)
        self.annotations_path = annotations_path
        self.text_descriptions_path = text_descriptions_path
        self.transform = transform
        self.is_train = is_train
        self.max_size = max_size

        # 加载标注数据
        with open(annotations_path, 'r') as f:
            self.coco_data = json.load(f)

        # 加载文本描述
        with open(text_descriptions_path, 'r', encoding='utf-8') as f:
            text_data = json.load(f)
            self.text_descriptions = text_data['multi_level_descriptions']

        # 创建类别映射
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        self.category_names = list(self.categories.values())

        # 创建图像ID到标注的映射
        self.image_annotations = {}
        for ann in self.coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in self.image_annotations:
                self.image_annotations[image_id] = []
            self.image_annotations[image_id].append(ann)

        # 过滤有标注的图像
        self.images = [img for img in self.coco_data['images']
                       if img['id'] in self.image_annotations]

        # CLIP处理器
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        print(f"Dataset loaded: {len(self.images)} images")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict:
        image_info = self.images[idx]
        image_id = image_info['id']

        # 加载图像
        image_path = self.data_path / image_info['file_name']
        image = cv2.imread(str(image_path))

        if image is None:
            # 如果图像加载失败，返回下一个
            return self.__getitem__((idx + 1) % len(self.images))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_h, original_w = image.shape[:2]

        # 获取标注
        annotations = self.image_annotations[image_id]

        # 准备边界框和标签
        boxes = []
        labels = []
        areas = []

        for ann in annotations:
            x, y, w, h = ann['bbox']
            # 转换为 x1, y1, x2, y2 格式
            x1, y1, x2, y2 = x, y, x + w, y + h

            # 确保坐标在图像范围内
            x1 = max(0, min(x1, original_w))
            y1 = max(0, min(y1, original_h))
            x2 = max(0, min(x2, original_w))
            y2 = max(0, min(y2, original_h))

            if x2 > x1 and y2 > y1:  # 有效边界框
                boxes.append([x1, y1, x2, y2])
                labels.append(ann['category_id'])
                areas.append((x2 - x1) * (y2 - y1))

        if not boxes:
            # 如果没有有效的边界框，返回下一个
            return self.__getitem__((idx + 1) % len(self.images))

        # 数据增强
        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                category_ids=labels
            )
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['category_ids']

        # 调整图像大小
        image, boxes, scale_factor = self._resize_image_and_boxes(image, boxes)

        # 准备多平台图像数据（这里简化为单一图像）
        images = {
            'satellite': image.copy(),  # 主要图像
            'uav': image.copy(),  # 可以是同一图像的变体
            'ground': image.copy(),  # 或者添加不同的处理
            'infrared': image.copy()  # 例如颜色空间变换
        }

        # 应用不同的处理到不同平台
        images = self._apply_platform_transforms(images)

        # 转换为tensor
        for platform in images:
            images[platform] = torch.from_numpy(images[platform]).permute(2, 0, 1).float() / 255.0

        # 生成多层次文本描述
        text_inputs = self._generate_text_inputs(labels)

        # 准备目标数据
        targets = {
            'labels': torch.tensor(labels, dtype=torch.long),
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'image_id': torch.tensor(image_id),
            'orig_size': torch.tensor([original_h, original_w]),
            'size': torch.tensor(image.shape[:2])
        }

        return {
            'images': images,
            'text_inputs': text_inputs,
            'targets': targets
        }

    def _resize_image_and_boxes(self, image: np.ndarray, boxes: List) -> Tuple:
        """调整图像大小并相应调整边界框"""
        h, w = image.shape[:2]

        # 计算缩放因子
        scale = min(self.max_size / w, self.max_size / h)

        if scale < 1:
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h))

            # 调整边界框
            scaled_boxes = []
            for box in boxes:
                x1, y1, x2, y2 = box
                scaled_boxes.append([
                    x1 * scale, y1 * scale, x2 * scale, y2 * scale
                ])
            boxes = scaled_boxes

        return image, boxes, scale

    def _apply_platform_transforms(self, images: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """为不同平台应用不同的变换"""
        # 卫星图像 - 标准处理
        # images['satellite'] 保持不变

        # UAV图像 - 轻微的对比度调整
        images['uav'] = cv2.convertScaleAbs(images['uav'], alpha=1.1, beta=10)

        # 地面图像 - 添加噪声模拟
        noise = np.random.normal(0, 5, images['ground'].shape).astype(np.uint8)
        images['ground'] = np.clip(images['ground'] + noise, 0, 255).astype(np.uint8)

        # 红外图像 - 转换为灰度再转回RGB模拟热成像
        gray = cv2.cvtColor(images['infrared'], cv2.COLOR_RGB2GRAY)
        images['infrared'] = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        images['infrared'] = cv2.cvtColor(images['infrared'], cv2.COLOR_BGR2RGB)

        return images

    def _generate_text_inputs(self, labels: List[int]) -> Dict[str, Dict]:
        """生成多层次文本输入"""
        # 确定主要类别
        main_categories = []
        for label in labels:
            category_name = self.categories[label]

            # 映射到大目标类别
            if any(keyword in category_name.lower()
                   for keyword in ['airport', 'runway', 'airplane']):
                main_categories.append('airport')
            elif any(keyword in category_name.lower()
                     for keyword in ['oil', 'tank', 'refinery']):
                main_categories.append('oil_field')
            elif any(keyword in category_name.lower()
                     for keyword in ['port', 'harbor', 'dock', 'ship']):
                main_categories.append('port')
            else:
                main_categories.append('other')

        # 选择主要类别
        primary_category = max(set(main_categories), key=main_categories.count) if main_categories else 'other'

        # 获取对应的文本描述
        if primary_category in self.text_descriptions:
            descriptions = self.text_descriptions[primary_category]
        else:
            # 默认描述
            descriptions = {
                'global': "Large infrastructure facility visible from satellite imagery",
                'local': "Complex structure with multiple components and buildings",
                'detail': "Detailed features include various functional areas and equipment"
            }

        # 使用CLIP tokenizer处理文本
        text_inputs = {}
        for level, text in descriptions.items():
            if level in ['global', 'local', 'detail']:
                # 限制文本长度
                if len(text) > 200:
                    text = text[:200]

                inputs = self.tokenizer(
                    text,
                    padding='max_length',
                    truncation=True,
                    max_length=77,  # CLIP的最大长度
                    return_tensors='pt'
                )

                text_inputs[level] = {
                    'input_ids': inputs['input_ids'].squeeze(0),
                    'attention_mask': inputs['attention_mask'].squeeze(0)
                }

        return text_inputs


def create_transforms(is_train: bool = True, max_size: int = 1024) -> A.Compose:
    """创建数据变换"""
    if is_train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=30, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
            A.GaussNoise(var_limit=(10, 50), p=0.2),
            A.GaussianBlur(blur_limit=3, p=0.1),
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['category_ids'],
            min_visibility=0.3
        ))
    else:
        return A.Compose([
            # 验证时只做标准化
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['category_ids']
        ))


def collate_fn(batch: List[Dict]) -> Dict:
    """自定义批处理函数"""
    # 过滤掉None值
    batch = [item for item in batch if item is not None]

    if not batch:
        return None

    # 分别处理每个组件
    images_batch = {}
    text_inputs_batch = {}
    targets_batch = []

    # 获取所有平台
    platforms = list(batch[0]['images'].keys())
    text_levels = list(batch[0]['text_inputs'].keys())

    # 处理图像数据
    for platform in platforms:
        platform_images = [item['images'][platform] for item in batch]
        images_batch[platform] = torch.stack(platform_images, dim=0)

    # 处理文本数据
    for level in text_levels:
        level_inputs = {}
        for key in ['input_ids', 'attention_mask']:
            level_data = [item['text_inputs'][level][key] for item in batch]
            level_inputs[key] = torch.stack(level_data, dim=0)
        text_inputs_batch[level] = level_inputs

    # 处理目标数据
    targets_batch = [item['targets'] for item in batch]

    return {
        'images': images_batch,
        'text_inputs': text_inputs_batch,
        'targets': targets_batch
    }


def create_dataloader(data_path: str,
                      annotations_path: str,
                      text_descriptions_path: str,
                      batch_size: int,
                      is_train: bool = True,
                      num_workers: int = 4,
                      distributed: bool = False) -> DataLoader:
    """创建数据加载器"""

    # 创建变换
    transform = create_transforms(is_train)

    # 创建数据集
    dataset = LargeTargetDataset(
        data_path=data_path,
        annotations_path=annotations_path,
        text_descriptions_path=text_descriptions_path,
        transform=transform,
        is_train=is_train
    )

    # 分布式采样器
    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=is_train)

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(is_train and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=is_train
    )

    return dataloader