#process.py
import os
import json
import cv2
import numpy as np
import random
import pandas as pd
from pathlib import Path
import yaml
from typing import Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from transformers import CLIPProcessor, CLIPModel
import logging
from tqdm import tqdm
import h5py
import scipy.io as sio
from collections import defaultdict
import shutil

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LargeTargetDataPreprocessor:
    """大目标检测数据预处理系统"""

    def __init__(self, config_path: str):
        """初始化预处理器

        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.large_target_classes = {
            'airport': ['airport', 'airfield', 'runway', 'terminal', 'hangar'],
            'oil_field': ['oil_field', 'oil_well', 'oil_tank', 'refinery'],
            'port': ['port', 'harbor', 'dock', 'wharf', 'shipping_yard']
        }
        self.setup_directories()

    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def setup_directories(self):
        """设置输出目录结构"""
        self.output_dir = Path(self.config['output']['base_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 创建子目录
        subdirs = [
            'images/satellite', 'images/uav', 'images/ground', 'images/infrared',
            'annotations/coco', 'annotations/yolo', 'annotations/xml',
            'text_descriptions', 'processed_data', 'splits', 'statistics'
        ]

        for subdir in subdirs:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)


class XViewProcessor:
    """xView数据集处理器"""

    def __init__(self, data_dir: str, output_dir: str, large_target_classes: Dict):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.large_target_classes = large_target_classes
        self.xview_class_mapping = self._load_xview_classes()

    def _load_xview_classes(self) -> Dict:
        """加载xView类别映射"""
        # xView数据集的类别映射（部分大目标相关）
        return {
            13: 'airport',
            15: 'runway',
            21: 'oil_tank',
            23: 'cargo_plane',
            24: 'helicopter',
            25: 'passenger_vehicle',
            26: 'small_car',
            27: 'bus',
            28: 'pickup_truck',
            29: 'utility_truck',
            30: 'truck',
            31: 'cargo_truck',
            32: 'truck_w/_box',
            33: 'truck_tractor',
            34: 'trailer',
            35: 'truck_w/_flatbed',
            36: 'truck_w/_liquid',
            37: 'crane_truck',
            38: 'railway_vehicle',
            39: 'passenger_car',
            40: 'cargo_car',
            41: 'flat_car',
            42: 'tank_car',
            43: 'locomotive',
            44: 'maritime_vessel',
            45: 'motorboat',
            46: 'sailboat',
            47: 'tugboat',
            48: 'barge',
            49: 'fishing_vessel',
            50: 'ferry',
            51: 'yacht',
            52: 'container_ship',
            53: 'oil_tanker',
            54: 'engineering_vehicle',
            55: 'tower_crane',
            56: 'container_crane',
            57: 'reach_stacker',
            58: 'straddle_carrier',
            59: 'mobile_crane',
            60: 'dump_truck'
        }

    def process_annotations(self, geojson_path: str) -> List[Dict]:
        """处理xView的GeoJSON标注文件"""
        try:
            with open(geojson_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load GeoJSON file: {geojson_path}. Error: {str(e)}")
            return []

        processed_annotations = []
        for feature in data['features']:
            properties = feature['properties']
            geometry = feature['geometry']

            # 确保有image_id属性
            image_id = str(properties.get('image_id', ''))
            if not image_id:
                continue

            # 提取类别信息
            type_id = properties.get('type_id')
            class_name = self.xview_class_mapping.get(type_id, 'unknown')

            # 检查是否为大目标相关类别
            if self._is_large_target_related(class_name):
                # 提取边界框坐标
                if geometry['type'] == 'Polygon':
                    coords = np.array(geometry['coordinates'][0])
                    x_coords = coords[:, 0]
                    y_coords = coords[:, 1]

                    bbox = {
                        'x_min': min(x_coords),
                        'y_min': min(y_coords),
                        'x_max': max(x_coords),
                        'y_max': max(y_coords),
                        'class_name': class_name,
                        'type_id': type_id,
                        'image_id': image_id,  # 添加image_id
                        'properties': properties
                    }
                    processed_annotations.append(bbox)

        return processed_annotations

    def _is_large_target_related(self, class_name: str) -> bool:
        """判断是否为大目标相关类别"""
        for target_type, keywords in self.large_target_classes.items():
            if any(keyword in class_name.lower() for keyword in keywords):
                return True
        return False

    def extract_large_target_images(self, image_dir: str, geojson_path: str) -> List[Dict]:
        """提取包含大目标的图像"""
        annotations = self.process_annotations(geojson_path)
        image_files = list(Path(image_dir).glob('*.tif'))

        # 创建image_id到标注的映射
        annotation_map = defaultdict(list)
        for ann in annotations:
            annotation_map[ann['image_id']].append(ann)

        large_target_data = []
        for image_file in tqdm(image_files, desc="Processing xView images"):
            image_id = image_file.stem
            image_annotations = annotation_map.get(image_id, [])

            if image_annotations:
                try:
                    image = cv2.imread(str(image_file))
                    if image is not None:
                        output_image_path = self.output_dir / 'images' / 'satellite' / f"{image_id}.jpg"
                        cv2.imwrite(str(output_image_path), image)

                        large_target_data.append({
                            'image_path': str(output_image_path),
                            'annotations': image_annotations,
                            'platform': 'satellite',
                            'dataset': 'xview'
                        })
                except Exception as e:
                    logger.error(f"Error processing image {image_file}: {str(e)}")

        logger.info(f"Extracted {len(large_target_data)} large target images from xView")
        return large_target_data


class UCASAODProcessor:
    """UCAS-AOD数据集处理器"""

    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)

    def process_annotations(self, annotation_file: str) -> List[Dict]:
        """处理UCAS-AOD标注文件"""
        annotations = []

        with open(annotation_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                x1, y1, x2, y2 = map(float, parts[:4])
                class_name = parts[4] if len(parts) > 4 else 'airplane'

                annotations.append({
                    'x_min': min(x1, x2),
                    'y_min': min(y1, y2),
                    'x_max': max(x1, x2),
                    'y_max': max(y1, y2),
                    'class_name': class_name
                })

        return annotations

    def extract_airport_images(self) -> List[Dict]:
        """提取机场相关图像"""
        airport_data = []

        # UCAS-AOD通常包含CAR和AIRPLANE两个子目录
        for class_dir in ['AIRPLANE']:
            class_path = self.data_dir / class_dir
            if not class_path.exists():
                logger.warning(f"UCAS-AOD directory not found: {class_path}")
                continue

            image_files = list(class_path.glob('*.png')) + list(class_path.glob('*.jpg'))

            for image_file in tqdm(image_files, desc=f"Processing UCAS-AOD {class_dir}"):
                # 查找对应的标注文件
                annotation_file = image_file.with_suffix('.txt')

                if annotation_file.exists():
                    try:
                        image = cv2.imread(str(image_file))
                        if image is not None:
                            annotations = self.process_annotations(str(annotation_file))

                            # 保存到UAV目录
                            output_image_path = self.output_dir / 'images' / 'uav' / image_file.name
                            cv2.imwrite(str(output_image_path), image)

                            airport_data.append({
                                'image_path': str(output_image_path),
                                'annotations': annotations,
                                'platform': 'uav',
                                'dataset': 'ucas-aod'
                            })
                    except Exception as e:
                        logger.error(f"Error processing image {image_file}: {str(e)}")

        logger.info(f"Extracted {len(airport_data)} airport images from UCAS-AOD")
        return airport_data


class AUAIRProcessor:
    """AU-AIR数据集处理器"""

    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)

    def extract_key_frames(self, video_dir: str, interval: int = 30) -> List[Dict]:
        """从视频中提取关键帧"""
        video_path = Path(self.data_dir) / video_dir
        if not video_path.exists():
            logger.error(f"Video directory not found: {video_path}")
            return []

        video_files = list(video_path.glob('*.mp4')) + list(video_path.glob('*.avi'))
        extracted_frames = []

        for video_file in tqdm(video_files, desc="Processing AU-AIR videos"):
            try:
                cap = cv2.VideoCapture(str(video_file))
                if not cap.isOpened():
                    logger.warning(f"Failed to open video: {video_file}")
                    continue

                frame_count = 0
                saved_count = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # 每隔指定帧数保存一帧
                    if frame_count % interval == 0:
                        frame_name = f"{video_file.stem}_frame_{saved_count:06d}.jpg"
                        output_path = self.output_dir / 'images' / 'uav' / frame_name
                        cv2.imwrite(str(output_path), frame)

                        extracted_frames.append({
                            'image_path': str(output_path),
                            'annotations': [],  # AU-AIR可能需要单独的标注处理
                            'platform': 'uav',
                            'dataset': 'au-air',
                            'video_source': str(video_file),
                            'frame_number': frame_count
                        })
                        saved_count += 1

                    frame_count += 1

                cap.release()
            except Exception as e:
                logger.error(f"Error processing video {video_file}: {str(e)}")

        logger.info(f"Extracted {len(extracted_frames)} key frames from AU-AIR")
        return extracted_frames


class TextDescriptionProcessor:
    """文本描述处理器"""

    def __init__(self, data_dirs: Dict[str, str], output_dir: str):
        self.data_dirs = data_dirs
        self.output_dir = Path(output_dir)
        self.large_target_keywords = [
            'airport', 'airfield', 'runway', 'terminal', 'hangar',
            'oil', 'refinery', 'tank', 'petroleum',
            'port', 'harbor', 'dock', 'wharf', 'shipping'
        ]

    def process_rsicd_dataset(self) -> List[Dict]:
        """处理RSICD数据集的文本描述"""
        rsicd_dir = Path(self.data_dirs['rsicd'])
        if not rsicd_dir.exists():
            logger.warning(f"RSICD directory not found: {rsicd_dir}")
            return []

        descriptions = []

        # RSICD通常有一个JSON文件包含所有描述
        json_files = list(rsicd_dir.glob('*.json'))

        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 修正：检查JSON结构是否包含"images"键
            if 'images' in data:
                # 标准RSICD格式
                for item in data['images']:
                    # 检查原始描述和所有句子
                    raw_desc = item.get('raw', '')
                    sentences = item.get('sentences', [])

                    # 检查原始描述和所有句子是否包含关键词
                    if (self._contains_large_target_keywords(raw_desc) or
                            any(self._contains_large_target_keywords(s.get('raw', '')) for s in sentences)):
                        descriptions.append({
                            'image_id': item.get('imgid', ''),
                            'description': raw_desc,
                            'sentences': [s.get('raw', '') for s in sentences],
                            'dataset': 'rsicd',
                            'category': self._categorize_description(raw_desc)
                        })
            else:
                # 尝试处理为列表格式
                for item in data:
                    if isinstance(item, dict):
                        raw_desc = item.get('raw', '')
                        sentences = item.get('sentences', [])

                        if (self._contains_large_target_keywords(raw_desc) or
                                any(self._contains_large_target_keywords(s.get('raw', '')) for s in sentences)):
                            descriptions.append({
                                'image_id': item.get('imgid', ''),
                                'description': raw_desc,
                                'sentences': [s.get('raw', '') for s in sentences],
                                'dataset': 'rsicd',
                                'category': self._categorize_description(raw_desc)
                            })

        logger.info(f"Processed {len(descriptions)} descriptions from RSICD")
        return descriptions

    def process_rsvgd_dataset(self) -> List[Dict]:
        """处理RSVGD数据集的文本描述"""
        rsvgd_dir = Path(self.data_dirs['rsvgd'])
        if not rsvgd_dir.exists():
            logger.warning(f"RSVGD directory not found: {rsvgd_dir}")
            return []

        descriptions = []

        # RSVGD的处理逻辑（根据实际数据格式调整）
        txt_files = list(rsvgd_dir.glob('*.txt'))

        for txt_file in txt_files:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()

            if self._contains_large_target_keywords(content):
                descriptions.append({
                    'image_id': txt_file.stem,
                    'description': content,
                    'dataset': 'rsvgd',
                    'category': self._categorize_description(content)
                })

        logger.info(f"Processed {len(descriptions)} descriptions from RSVGD")
        return descriptions

    def _contains_large_target_keywords(self, text: str) -> bool:
        """检查文本是否包含大目标关键词"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.large_target_keywords)

    def _categorize_description(self, text: str) -> str:
        """根据描述内容分类"""
        text_lower = text.lower()

        if any(word in text_lower for word in ['airport', 'airfield', 'runway']):
            return 'airport'
        elif any(word in text_lower for word in ['oil', 'refinery', 'petroleum']):
            return 'oil_field'
        elif any(word in text_lower for word in ['port', 'harbor', 'dock']):
            return 'port'
        else:
            return 'other'

    def generate_multi_level_descriptions(self, descriptions: List[Dict]) -> Dict:
        """生成多层次描述"""
        categorized = defaultdict(list)

        for desc in descriptions:
            category = desc['category']
            categorized[category].append(desc['description'])

        multi_level = {}
        for category, texts in categorized.items():
            multi_level[category] = {
                'global': self._generate_global_description(category),
                'local': self._generate_local_description(category),
                'detail': self._generate_detail_description(category),
                'examples': texts[:5]  # 保存一些例子
            }

        return multi_level

    def _generate_global_description(self, category: str) -> str:
        """生成全局描述"""
        templates = {
            'airport': "Large aviation infrastructure for aircraft operations and passenger services",
            'oil_field': "Industrial complex for petroleum extraction, processing and storage",
            'port': "Maritime facility for cargo handling and vessel operations"
        }
        return templates.get(category, "Large infrastructure facility")

    def _generate_local_description(self, category: str) -> str:
        """生成局部描述"""
        templates = {
            'airport': "Consisting of runways, terminals, hangars, and support facilities",
            'oil_field': "Including drilling platforms, storage tanks, processing units and pipelines",
            'port': "Comprising docks, warehouses, cranes and shipping channels"
        }
        return templates.get(category, "Complex structural components")

    def _generate_detail_description(self, category: str) -> str:
        """生成细节描述"""
        templates = {
            'airport': "Detailed features include runway markings, aircraft parking areas, ground service equipment",
            'oil_field': "Specific elements such as wellheads, separator units, flare stacks and control systems",
            'port': "Fine details like container yards, loading equipment, navigation aids and security features"
        }
        return templates.get(category, "Specific detailed features")


class FLIRADASProcessor:
    """FLIR ADAS v2红外数据集处理器"""

    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            logger.error(f"FLIR ADAS directory not found: {self.data_dir}")

        self.output_dir = Path(output_dir)
        # 使用小写类别名匹配
        self.large_target_classes = {
            'bus', 'truck', 'cargo truck', 'container ship', 'large vehicle'
        }

    def process_dataset(self) -> List[Dict]:
        """处理FLIR ADAS v2数据集"""
        if not self.data_dir.exists():
            return []

        infrared_data = []

        # FLIR ADAS v2数据集结构
        thermal_dir = self.data_dir / 'thermal_8_bit'
        annotation_dir = self.data_dir / 'annotations'

        # 处理训练集和验证集
        for split in ['train']:
            split_dir = thermal_dir / split
            annotation_file = annotation_dir / f'{split}.json'

            if not annotation_file.exists():
                logger.warning(f"Annotation file not found: {annotation_file}")
                continue

            # 加载COCO格式的标注
            with open(annotation_file, 'r') as f:
                coco_data = json.load(f)

            # 创建图像ID到文件名的映射
            image_id_map = {img['id']: img['file_name'] for img in coco_data['images']}

            # 创建类别ID到类别名称的映射
            category_map = {cat['id']: cat['name'] for cat in coco_data['categories']}

            # 按图像分组标注
            image_annotations = defaultdict(list)
            for ann in coco_data['annotations']:
                image_id = ann['image_id']
                category_id = ann['category_id']
                category_name = category_map.get(category_id, 'unknown').lower()

                # 只处理大目标类别
                if any(target_class in category_name for target_class in self.large_target_classes):
                    image_annotations[image_id].append({
                        'bbox': ann['bbox'],
                        'category_name': category_name,
                        'area': ann['area'],
                        'iscrowd': ann['iscrowd']
                    })

            # 处理每个图像
            for image_id, annotations in tqdm(image_annotations.items(),
                                              desc=f"Processing FLIR ADAS {split}"):
                image_name = image_id_map.get(image_id)
                if not image_name:
                    continue

                image_path = split_dir / image_name
                if not image_path.exists():
                    continue

                # 读取热红外图像
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue

                # 转换为伪彩色（可选）
                # color_image = self._apply_colormap(image)
                # 或者直接转换为RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

                # 保存处理后的图像
                output_path = self.output_dir / 'images' / 'infrared' / image_name
                cv2.imwrite(str(output_path), rgb_image)

                # 转换为标准标注格式
                standard_annotations = []
                for ann in annotations:
                    x, y, w, h = ann['bbox']
                    standard_annotations.append({
                        'x_min': x,
                        'y_min': y,
                        'x_max': x + w,
                        'y_max': y + h,
                        'class_name': ann['category_name']
                    })

                infrared_data.append({
                    'image_path': str(output_path),
                    'annotations': standard_annotations,
                    'platform': 'infrared',
                    'dataset': 'flir-adas'
                })

        logger.info(f"Processed {len(infrared_data)} infrared images from FLIR ADAS")
        return infrared_data

    def _apply_colormap(self, gray_image: np.ndarray) -> np.ndarray:
        """应用伪彩色映射到热红外图像"""
        # 使用OpenCV的COLORMAP_JET
        colored = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)
        return colored


class DataAugmentation:
    """数据增强处理器"""

    def __init__(self, config: Dict):
        self.config = config
        self.augmentation_pipeline = self._create_augmentation_pipeline()

    def _create_augmentation_pipeline(self):
        """创建数据增强管道"""
        return A.Compose([
            # 几何变换
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=30, p=0.7),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=30,
                p=0.5
            ),

            # 颜色变换
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5
            ),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.3),

            # 噪声和模糊
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.2),

            # 天气效果
            A.RandomRain(p=0.1),
            A.RandomShadow(p=0.1),

        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['category_ids']
        ))

    def augment_data(self, data_list: List[Dict], num_augmentations: int = 3) -> List[Dict]:
        """对数据进行增强"""
        augmented_data = []

        for data_item in tqdm(data_list, desc="Augmenting data"):
            image_path = data_item['image_path']
            annotations = data_item['annotations']

            # 读取原始图像
            image = cv2.imread(image_path)
            if image is None:
                continue

            # 准备边界框和类别
            bboxes = []
            category_ids = []

            for ann in annotations:
                bbox = [ann['x_min'], ann['y_min'], ann['x_max'], ann['y_max']]
                bboxes.append(bbox)
                category_ids.append(ann['class_name'])

            # 应用增强
            for aug_idx in range(num_augmentations):
                try:
                    augmented = self.augmentation_pipeline(
                        image=image,
                        bboxes=bboxes,
                        category_ids=category_ids
                    )

                    # 保存增强后的图像
                    aug_image_name = f"{Path(image_path).stem}_aug_{aug_idx}.jpg"
                    aug_image_path = Path(image_path).parent / aug_image_name
                    cv2.imwrite(str(aug_image_path), augmented['image'])

                    # 更新标注
                    aug_annotations = []
                    for bbox, cat_id in zip(augmented['bboxes'], augmented['category_ids']):
                        aug_annotations.append({
                            'x_min': bbox[0],
                            'y_min': bbox[1],
                            'x_max': bbox[2],
                            'y_max': bbox[3],
                            'class_name': cat_id
                        })

                    augmented_data.append({
                        'image_path': str(aug_image_path),
                        'annotations': aug_annotations,
                        'platform': data_item['platform'],
                        'dataset': data_item['dataset'] + '_aug'
                    })

                except Exception as e:
                    logger.warning(f"Augmentation failed for {image_path}: {e}")

        logger.info(f"Generated {len(augmented_data)} augmented samples")
        return augmented_data


class AnnotationConverter:
    """标注格式转换器"""

    def __init__(self, output_dir: str, class_mapping: Dict):
        self.output_dir = Path(output_dir)
        self.class_mapping = class_mapping

    def to_coco_format(self, data_list: List[Dict], split: str = 'train') -> Dict:
        """转换为COCO格式"""
        coco_data = {
            "info": {
                "description": "Large Target Detection Dataset",
                "version": "1.0",
                "year": 2024
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }

        # 添加类别
        for class_id, class_name in self.class_mapping.items():
            coco_data["categories"].append({
                "id": class_id,
                "name": class_name,
                "supercategory": "large_target"
            })

        # 处理图像和标注
        annotation_id = 1
        for image_id, data_item in enumerate(data_list):
            image_path = data_item['image_path']

            # 读取图像获取尺寸
            image = cv2.imread(image_path)
            if image is None:
                continue

            height, width = image.shape[:2]

            # 添加图像信息
            coco_data["images"].append({
                "id": image_id,
                "file_name": Path(image_path).name,
                "width": width,
                "height": height,
                "platform": data_item['platform'],
                "dataset": data_item['dataset']
            })

            # 添加标注
            for ann in data_item['annotations']:
                x_min, y_min = ann['x_min'], ann['y_min']
                x_max, y_max = ann['x_max'], ann['y_max']

                # 计算COCO格式的bbox和area
                bbox_width = x_max - x_min
                bbox_height = y_max - y_min
                area = bbox_width * bbox_height

                # 获取类别ID
                class_name = ann['class_name']
                category_id = next((cid for cid, cname in self.class_mapping.items()
                                    if cname == class_name), 1)

                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x_min, y_min, bbox_width, bbox_height],
                    "area": area,
                    "iscrowd": 0
                })
                annotation_id += 1

        # 保存COCO格式文件
        output_file = self.output_dir / 'annotations' / 'coco' / f'{split}.json'
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)

        logger.info(f"Saved COCO format annotations to {output_file}")
        return coco_data

    def to_yolo_format(self, data_list: List[Dict], split: str = 'train'):
        """转换为YOLO格式"""
        yolo_dir = self.output_dir / 'annotations' / 'yolo' / split
        yolo_dir.mkdir(parents=True, exist_ok=True)

        for data_item in data_list:
            image_path = data_item['image_path']

            # 读取图像获取尺寸
            image = cv2.imread(image_path)
            if image is None:
                continue

            height, width = image.shape[:2]

            # 创建YOLO标注文件
            yolo_file = yolo_dir / f"{Path(image_path).stem}.txt"

            with open(yolo_file, 'w') as f:
                for ann in data_item['annotations']:
                    x_min, y_min = ann['x_min'], ann['y_min']
                    x_max, y_max = ann['x_max'], ann['y_max']

                    # 转换为YOLO格式 (中心点坐标和相对尺寸)
                    x_center = (x_min + x_max) / (2 * width)
                    y_center = (y_min + y_max) / (2 * height)
                    bbox_width = (x_max - x_min) / width
                    bbox_height = (y_max - y_min) / height

                    # 获取类别ID
                    class_name = ann['class_name']
                    class_id = next((cid for cid, cname in self.class_mapping.items()
                                     if cname == class_name), 0)

                    f.write(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")

        logger.info(f"Saved YOLO format annotations to {yolo_dir}")


class QualityController:
    """数据质量控制器"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.statistics = defaultdict(int)

    def validate_annotations(self, data_list: List[Dict]) -> List[Dict]:
        """验证标注数据质量"""
        valid_data = []

        # 标准化类别名称
        for data_item in data_list:
            for ann in data_item['annotations']:
                ann['class_name'] = self._normalize_class_name(ann['class_name'])

        for data_item in tqdm(data_list, desc="Validating annotations"):
            image_path = data_item['image_path']
            annotations = data_item['annotations']

            # 检查图像是否存在且可读
            if not Path(image_path).exists():
                self.statistics['missing_images'] += 1
                continue

            image = cv2.imread(image_path)
            if image is None:
                self.statistics['corrupted_images'] += 1
                continue

            height, width = image.shape[:2]
            valid_annotations = []

            for ann in annotations:
                # 验证边界框坐标
                x_min, y_min = ann['x_min'], ann['y_min']
                x_max, y_max = ann['x_max'], ann['y_max']

                # 检查坐标范围
                if (x_min >= 0 and y_min >= 0 and
                        x_max <= width and y_max <= height and
                        x_min < x_max and y_min < y_max):

                    # 检查目标尺寸（过滤过小的目标）
                    bbox_width = x_max - x_min
                    bbox_height = y_max - y_min

                    if bbox_width >= 20 and bbox_height >= 20:
                        valid_annotations.append(ann)
                        self.statistics['valid_annotations'] += 1
                    else:
                        self.statistics['small_annotations'] += 1
                else:
                    self.statistics['invalid_coordinates'] += 1

            if valid_annotations:
                data_item['annotations'] = valid_annotations
                valid_data.append(data_item)
                self.statistics['valid_images'] += 1
            else:
                self.statistics['images_no_valid_annotations'] += 1

        logger.info(f"Quality validation completed. Valid data: {len(valid_data)}")
        return valid_data

    def generate_statistics(self, data_list: List[Dict]) -> Dict:
        """生成数据统计信息"""
        stats = {
            'total_images': len(data_list),
            'total_annotations': sum(len(item['annotations']) for item in data_list),
            'platform_distribution': defaultdict(int),
            'class_distribution': defaultdict(int),
            'dataset_distribution': defaultdict(int),
            'image_size_distribution': defaultdict(int)
        }

        for data_item in data_list:
            platform = data_item['platform']
            dataset = data_item['dataset']

            stats['platform_distribution'][platform] += 1
            stats['dataset_distribution'][dataset] += 1

            # 分析图像尺寸
            image = cv2.imread(data_item['image_path'])
            if image is not None:
                height, width = image.shape[:2]
                size_category = self._categorize_image_size(width, height)
                stats['image_size_distribution'][size_category] += 1

            # 统计类别分布
            for ann in data_item['annotations']:
                class_name = ann['class_name']
                stats['class_distribution'][class_name] += 1

        # 保存统计信息
        stats_file = self.output_dir / 'statistics' / 'dataset_statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(dict(stats), f, indent=2, default=str)

        logger.info(f"Dataset statistics saved to {stats_file}")
        return stats

    def _categorize_image_size(self, width: int, height: int) -> str:
        """根据图像尺寸分类"""
        if width <= 512 and height <= 512:
            return 'small'
        elif width <= 1024 and height <= 1024:
            return 'medium'
        elif width <= 2048 and height <= 2048:
            return 'large'
        else:
            return 'extra_large'

    def _normalize_class_name(self, class_name: str) -> str:
        """标准化类别名称"""
        # 处理数值型类别名
        try:
            num = float(class_name)
            return f"class_{int(num)}"
        except ValueError:
            return class_name.lower().replace(' ', '_')

    def visualize_samples(self, data_list: List[Dict], num_samples: int = 20):
        """可视化样本"""
        import random
        samples = random.sample(data_list, min(num_samples, len(data_list)))

        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        axes = axes.flatten()

        for idx, data_item in enumerate(samples):
            if idx >= len(axes):
                break

            image = cv2.imread(data_item['image_path'])
            if image is None:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 绘制边界框
            for ann in data_item['annotations']:
                x_min = int(ann['x_min'])
                y_min = int(ann['y_min'])
                x_max = int(ann['x_max'])
                y_max = int(ann['y_max'])

                cv2.rectangle(image_rgb, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                cv2.putText(image_rgb, ann['class_name'], (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            axes[idx].imshow(image_rgb)
            axes[idx].set_title(f"{data_item['platform']} - {data_item['dataset']}")
            axes[idx].axis('off')

        # 隐藏多余的子图
        for idx in range(len(samples), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'statistics' / 'sample_visualization.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("Sample visualization saved")


class DatasetSplitter:
    """数据集分割器"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)

    def create_splits(self, data_list: List[Dict],
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.15,
                      test_ratio: float = 0.15) -> Dict[str, List[Dict]]:
        """创建训练/验证/测试集划分"""

        # 按类别分层抽样
        class_data = defaultdict(list)
        for item in data_list:
            # 使用第一个标注的类别作为主类别
            if item['annotations']:
                main_class = self._normalize_class_name(item['annotations'][0]['class_name'])
                class_data[main_class].append(item)

        train_data, val_data, test_data = [], [], []

        for class_name, class_items in class_data.items():
            n_samples = len(class_items)

            # 根据样本数量调整分割策略
            if n_samples == 1:
                # 只有一个样本，全部放入训练集
                train_data.extend(class_items)
                logger.info(f"Class {class_name} (1 sample) -> train")
            elif n_samples == 2:
                # 两个样本：一个训练，一个验证
                train_data.append(class_items[0])
                val_data.append(class_items[1])
                logger.info(f"Class {class_name} (2 samples) -> 1 train, 1 val")
            elif n_samples == 3:
                # 三个样本：两个训练，一个验证
                train_data.extend(class_items[:2])
                val_data.append(class_items[2])
                logger.info(f"Class {class_name} (3 samples) -> 2 train, 1 val")
            else:
                # 四个或更多样本，按比例分割
                n_train = max(1, int(train_ratio * n_samples))
                n_val = max(1, int(val_ratio * n_samples))
                n_test = n_samples - n_train - n_val

                # 确保测试集至少有一个样本
                if n_test < 1:
                    n_test = 1
                    n_train = n_samples - n_val - n_test

                # 随机打乱样本
                random.shuffle(class_items)

                train_items = class_items[:n_train]
                val_items = class_items[n_train:n_train + n_val]
                test_items = class_items[n_train + n_val:]

                train_data.extend(train_items)
                val_data.extend(val_items)
                test_data.extend(test_items)

                logger.info(
                    f"Class {class_name} ({n_samples} samples): {len(train_items)} train, {len(val_items)} val, {len(test_items)} test")

        splits = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }

        # 保存分割信息
        for split_name, split_data in splits.items():
            split_file = self.output_dir / 'splits' / f'{split_name}.json'
            split_info = [
                {
                    'image_path': item['image_path'],
                    'platform': item['platform'],
                    'dataset': item['dataset'],
                    'num_annotations': len(item['annotations'])
                }
                for item in split_data
            ]

            with open(split_file, 'w') as f:
                json.dump(split_info, f, indent=2)

        logger.info(f"Dataset splits created: Train={len(train_data)}, "
                    f"Val={len(val_data)}, Test={len(test_data)}")

        return splits

    def _normalize_class_name(self, class_name: str) -> str:
        """标准化类别名称"""
        # 处理数值型类别名
        try:
            num = float(class_name)
            return f"class_{int(num)}"
        except ValueError:
            return class_name.lower().replace(' ', '_')


def main():
    """主处理流程"""

    # 配置信息
    config = {
        'datasets': {
            'xview': '.\datasets\\xView',
            'ucas_aod': '.\datasets\\UCAS-AOD',
            'au_air': '.\datasets\\AU-AIR',
            'rsicd': '.\datasets\\RSICD',
            # 'rsvgd': '.\datasets\\RSVGD',
            'flir_adas': '.\datasets\\FLIR_ADAS_v2',
        },
        'output': {
            'base_dir': './processed_large_target_dataset'
        },
        'augmentation': {
            'enabled': True,
            'num_augmentations': 3
        },
        'quality_control': {
            'min_bbox_size': 20,
            'max_aspect_ratio': 10
        }
    }

    # 保存配置
    config_path = 'preprocessing_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    # 初始化预处理器
    preprocessor = LargeTargetDataPreprocessor(config_path)

    # 类别映射
    class_mapping = {
        1: 'airport',
        2: 'oil_field',
        3: 'port',
        4: 'runway',
        5: 'oil_tank',
        6: 'dock',
        7: 'bus',  # 新增FLIR类别
        8: 'truck',  # 新增FLIR类别
        9: 'cargo_truck'  # 新增FLIR类别
    }

    logger.info("Starting large target dataset preprocessing...")
    logger.info(f"Config: {json.dumps(config, indent=2)}")

    # 验证数据集路径
    for dataset_name, dataset_path in config['datasets'].items():
        if not Path(dataset_path).exists():
            logger.warning(f"Dataset path not found: {dataset_name} - {dataset_path}")

    # 1. 处理各个数据集
    all_data = []

    # 处理xView数据
    if Path(config['datasets']['xview']).exists():
        xview_processor = XViewProcessor(
            config['datasets']['xview'],
            config['output']['base_dir'],
            preprocessor.large_target_classes
        )
        xview_data = xview_processor.extract_large_target_images(
            config['datasets']['xview'] + '/train_images',
            config['datasets']['xview'] + '/xView_train.geojson'
        )
        all_data.extend(xview_data)

    # 处理UCAS-AOD数据
    if Path(config['datasets']['ucas_aod']).exists():
        ucas_processor = UCASAODProcessor(
            config['datasets']['ucas_aod'],
            config['output']['base_dir']
        )
        ucas_data = ucas_processor.extract_airport_images()
        all_data.extend(ucas_data)

    # 处理AU-AIR数据
    if Path(config['datasets']['au_air']).exists():
        auair_processor = AUAIRProcessor(
            config['datasets']['au_air'],
            config['output']['base_dir']
        )
        auair_data = auair_processor.extract_key_frames(
            config['datasets']['au_air'] + '/videos'
        )
        all_data.extend(auair_data)

    # 处理红外数据
    if Path(config['datasets']['flir_adas']).exists():
        flir_processor = FLIRADASProcessor(
            config['datasets']['flir_adas'],
            config['output']['base_dir']
        )
        flir_data = flir_processor.process_dataset()
        all_data.extend(flir_data)

    # 2. 处理文本描述
    text_processor = TextDescriptionProcessor(
        {
            'rsicd': config['datasets']['rsicd'],
            #'rsvgd': config['datasets']['rsvgd']
        },
        config['output']['base_dir']
    )

    descriptions = []
    if Path(config['datasets']['rsicd']).exists():
        rsicd_descriptions = text_processor.process_rsicd_dataset()
        descriptions.extend(rsicd_descriptions)

    # if Path(config['datasets']['rsvgd']).exists():
    #     rsvgd_descriptions = text_processor.process_rsvgd_dataset()
    #     descriptions.extend(rsvgd_descriptions)

    # 生成多层次描述
    multi_level_descriptions = text_processor.generate_multi_level_descriptions(descriptions)

    # 保存文本描述
    desc_file = Path(config['output']['base_dir']) / 'text_descriptions' / 'descriptions.json'
    with open(desc_file, 'w', encoding='utf-8') as f:
        json.dump({
            'individual_descriptions': descriptions,
            'multi_level_descriptions': multi_level_descriptions
        }, f, indent=2, ensure_ascii=False)

    # 3. 数据质量控制
    quality_controller = QualityController(config['output']['base_dir'])
    validated_data = quality_controller.validate_annotations(all_data)

    # 4. 数据增强
    if config['augmentation']['enabled']:
        augmentation = DataAugmentation(config['augmentation'])
        augmented_data = augmentation.augment_data(
            validated_data,
            config['augmentation']['num_augmentations']
        )
        validated_data.extend(augmented_data)

    # 5. 数据集分割
    splitter = DatasetSplitter(config['output']['base_dir'])
    splits = splitter.create_splits(validated_data)

    # 6. 转换标注格式
    converter = AnnotationConverter(config['output']['base_dir'], class_mapping)

    for split_name, split_data in splits.items():
        converter.to_coco_format(split_data, split_name)
        converter.to_yolo_format(split_data, split_name)

    # 7. 生成统计信息和可视化
#    stats = quality_controller.generate_statistics(validated_data)
#    quality_controller.visualize_samples(validated_data)

    # 8. 保存最终的处理数据
    final_data_file = Path(config['output']['base_dir']) / 'processed_data' / 'final_dataset.json'
    with open(final_data_file, 'w') as f:
        json.dump(validated_data, f, indent=2, default=str)

    logger.info("Dataset preprocessing completed successfully!")
    logger.info(f"Total processed images: {len(validated_data)}")
    logger.info(f"Total annotations: {sum(len(item['annotations']) for item in validated_data)}")

    # 打印统计信息
    # print("\n=== Dataset Statistics ===")
    # print(f"Platform distribution: {dict(stats['platform_distribution'])}")
    # print(f"Class distribution: {dict(stats['class_distribution'])}")
    # print(f"Dataset distribution: {dict(stats['dataset_distribution'])}")


if __name__ == "__main__":
    main()