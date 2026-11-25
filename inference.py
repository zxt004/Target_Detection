# inference.py
import torch
import numpy as np
from model import build_model
from transformers import CLIPTokenizer
from PIL import Image
import cv2
import json
import argparse
import yaml
from pathlib import Path


class LargeTargetDetectionAPI:
    """大目标检测API接口"""

    def __init__(self, config_path, model_path, device='cuda'):
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device(device)

        # 加载模型
        self.model = build_model(
            num_classes=self.config['model']['num_classes'],
            backbone_name=self.config['model']['backbone'],
            text_encoder_name=self.config['model']['text_encoder']
        )

        # 加载检查点
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # 加载文本描述
        with open(self.config['data']['text_descriptions'], 'r', encoding='utf-8') as f:
            text_data = json.load(f)
            self.text_descriptions = text_data['multi_level_descriptions']

        # 初始化文本处理器
        self.tokenizer = CLIPTokenizer.from_pretrained(self.config['model']['text_encoder'])

        # 类别名称
        self.class_names = self.config['data']['class_names']
        self.platforms = ['satellite', 'uav', 'ground', 'infrared']

    def preprocess_image(self, image, platform='satellite'):
        """预处理输入图像"""
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            if image.shape[2] == 4:  # RGBA
                image = image[:, :, :3]
        elif isinstance(image, Image.Image):
            image = np.array(image)
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[2] == 4:
                image = image[:, :, :3]

        # 调整尺寸
        h, w = image.shape[:2]
        scale = min(self.config['data']['input_size'][0] / w, self.config['data']['input_size'][1] / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h))

        # 填充到固定尺寸
        padded_image = np.zeros((self.config['data']['input_size'][1], self.config['data']['input_size'][0], 3),
                                dtype=np.uint8)
        padded_image[:new_h, :new_w] = image

        # 归一化
        image_tensor = torch.from_numpy(padded_image).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        return image_tensor, scale, (w, h)

    def prepare_text_inputs(self, target_type='airport'):
        """准备文本输入"""
        if target_type not in self.text_descriptions:
            target_type = 'airport'  # 默认

        descriptions = self.text_descriptions[target_type]

        text_inputs = {}
        for level in ['global', 'local', 'detail']:
            text = descriptions.get(level, f"{target_type} description")
            inputs = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=77,  # CLIP的最大长度
                return_tensors='pt'
            )
            text_inputs[level] = {
                'input_ids': inputs['input_ids'].to(self.device),
                'attention_mask': inputs['attention_mask'].to(self.device)
            }

        return text_inputs

    def detect(self, satellite_image, uav_image, ground_image, infrared_image, target_type='airport',
               confidence_threshold=0.5, use_rl=True, max_optimization_steps=3):
        """执行检测（包含RL提示优化）"""
        # RL提示优化迭代
        best_prompt = None
        best_detections = []
        best_score = -float('inf')

        for step in range(max_optimization_steps):
            # 准备文本输入（使用优化的提示）
            if use_rl and step > 0:
                prompt_info = self._optimize_prompt(target_type, 'satellite', detections)
                text_inputs = prompt_info['optimized_inputs']
            else:
                text_inputs = self.prepare_text_inputs(target_type)

            # 执行检测
            detections = self._run_detection(
                satellite_image, uav_image, ground_image, infrared_image, text_inputs
            )

            # 评估检测质量
            score = self._evaluate_detections(detections, target_type)

            # 记录最佳结果
            if score > best_score:
                best_score = score
                best_detections = detections
                best_prompt = text_inputs

        return best_detections, best_prompt

    def _optimize_prompt(self, target_type, platform, previous_detections):
        """执行一步RL提示优化"""
        # 计算当前奖励（检测质量改进）
        reward = self._compute_reward(previous_detections, target_type)

        # 执行RL优化步骤
        with torch.no_grad():
            prompt_outputs = self.model.prompt_module(
                self.prompt_template,
                target_type,
                platform,
                current_state=previous_detections['text_features']
            )

            # 更新RL策略
            self.model.prompt_module.train()
            rl_loss = self.loss_module._compute_prompt_rl_loss(
                {'prompt_outputs': prompt_outputs},
                torch.tensor([reward], device=self.device)
            )
            rl_loss.backward()
            self.optimizer.step()
            self.model.prompt_module.eval()

        return {
            'optimized_inputs': self._format_text_inputs(prompt_outputs['optimized_prompt']),
            'rl_reward': reward
        }

    def _compute_reward(self, detections, target_type):
        """计算RL奖励"""
        # 简化的奖励函数：基于召回率和精确率
        tp = sum(1 for d in detections if d['class_name'] == target_type and d['score'] > 0.5)
        fp = sum(1 for d in detections if d['class_name'] != target_type and d['score'] > 0.5)
        fn = 5 - tp  # 假设每个图像有5个目标

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        return 2 * precision * recall / (precision + recall + 1e-8)  # F1分数

    def visualize(self, image, detections, output_path=None):
        """可视化检测结果"""
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            pass
        elif isinstance(image, Image.Image):
            image = np.array(image)

        # 创建副本
        vis_image = image.copy()

        # 绘制边界框和标签
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            score = det['score']

            # 绘制边界框
            cv2.rectangle(vis_image,
                          (int(x1), int(y1)),
                          (int(x2), int(y2)),
                          (0, 255, 0), 2)

            # 绘制标签
            label = f"{class_name}: {score:.2f}"
            cv2.putText(vis_image, label,
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 转换为PIL格式返回或保存
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            return output_path
        else:
            return Image.fromarray(vis_image)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='大目标检测API')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--model', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--satellite', type=str, required=True, help='卫星图像路径')
    parser.add_argument('--uav', type=str, required=True, help='无人机图像路径')
    parser.add_argument('--ground', type=str, required=True, help='地面图像路径')
    parser.add_argument('--infrared', type=str, required=True, help='红外图像路径')
    parser.add_argument('--target', type=str, default='airport', choices=['airport', 'oil_field', 'port'],
                        help='目标类型: airport, oil_field, port')
    parser.add_argument('--output', type=str, help='输出图像路径')

    args = parser.parse_args()

    # 初始化API
    api = LargeTargetDetectionAPI(args.config, args.model)

    # 生成优化提示
    prompt_info = api.generate_prompt('satellite', args.target)
    print("\n===== 优化提示 =====")
    print(f"提示: {prompt_info['prompt']}")
    print("Token权重分析:")
    for token, weight in zip(prompt_info['tokens'], prompt_info['token_weights']):
        print(f"{token}: {weight:.2f}")

    # 执行检测
    detections = api.detect(
        satellite_image=args.satellite,
        uav_image=args.uav,
        ground_image=args.ground,
        infrared_image=args.infrared,
        target_type=args.target
    )

    print("\n===== 检测结果 =====")
    for det in detections:
        print(f"{det['class_name']}: 置信度={det['score']:.4f}, 位置={det['bbox']}")

    # 可视化结果
    if args.output:
        # 使用卫星图像作为可视化基础
        api.visualize(args.satellite, detections, args.output)
        print(f"\n可视化结果已保存至: {args.output}")
    else:
        # 显示结果
        result_image = api.visualize(args.satellite, detections)
        result_image.show()