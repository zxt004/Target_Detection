# model.py
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from transformers import CLIPModel, CLIPProcessor, AutoTokenizer, AutoModel
import math
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class AdaptiveFPN(nn.Module):
    """自适应特征金字塔网络 (AFPN)"""

    def __init__(self, in_channels_list, out_channels):
        super(AdaptiveFPN, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.attention_weights = nn.Parameter(torch.ones(len(in_channels_list)))

        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
            self.fpn_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        # 处理侧向连接
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, x)]

        # 计算自适应权重
        weights = F.softmax(self.attention_weights, dim=0)

        # 构建特征金字塔
        fused_features = 0
        for i, lateral in enumerate(laterals):
            # 上采样到最大分辨率
            if i < len(laterals) - 1:
                size = laterals[0].shape[2:]
                lateral = F.interpolate(lateral, size=size, mode='nearest')
            fused_features += weights[i] * lateral

        return self.fpn_convs[0](fused_features)


class CrossPlatformAttention(nn.Module):
    """跨平台注意力机制"""

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(CrossPlatformAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "Embedding dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None):
        batch_size, seq_len, _ = query.shape

        # 线性变换并分割头
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, v)

        # 合并头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class TextGuidedFeatureEnhancement(nn.Module):
    """文本引导特征增强模块 (TG-FE)"""

    def __init__(self, visual_dim, text_dim, num_heads, dropout=0.1):
        super(TextGuidedFeatureEnhancement, self).__init__()
        self.cross_attention = CrossPlatformAttention(
            embed_dim=visual_dim, num_heads=num_heads, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(visual_dim)
        self.norm2 = nn.LayerNorm(visual_dim)
        self.ffn = nn.Sequential(
            nn.Linear(visual_dim, visual_dim * 4),
            nn.ReLU(),
            nn.Linear(visual_dim * 4, visual_dim),
            nn.Dropout(dropout)
        )

        # 文本到视觉的投影
        self.text_proj = nn.Linear(text_dim, visual_dim)

    def forward(self, visual_features, text_features):
        # 投影文本特征到视觉空间
        text_projected = self.text_proj(text_features)

        # 文本引导的注意力
        enhanced_features, attn_weights = self.cross_attention(
            visual_features, text_projected, text_projected
        )

        # 残差连接和层归一化
        enhanced_features = self.norm1(visual_features + enhanced_features)

        # 前馈网络
        ff_output = self.ffn(enhanced_features)
        enhanced_features = self.norm2(enhanced_features + ff_output)

        return enhanced_features, attn_weights


class VisualGuidedTextRefinement(nn.Module):
    """视觉引导文本精化模块 (VG-TR)"""

    def __init__(self, visual_dim, text_dim, num_heads, dropout=0.1):
        super(VisualGuidedTextRefinement, self).__init__()
        self.cross_attention = CrossPlatformAttention(
            embed_dim=text_dim, num_heads=num_heads, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(text_dim)
        self.norm2 = nn.LayerNorm(text_dim)
        self.ffn = nn.Sequential(
            nn.Linear(text_dim, text_dim * 4),
            nn.ReLU(),
            nn.Linear(text_dim * 4, text_dim),
            nn.Dropout(dropout)
        )

        # 视觉到文本的投影
        self.visual_proj = nn.Linear(visual_dim, text_dim)

    def forward(self, text_features, visual_features):
        # 投影视觉特征到文本空间
        visual_projected = self.visual_proj(visual_features)

        # 视觉引导的注意力
        refined_text, attn_weights = self.cross_attention(
            text_features, visual_projected, visual_projected
        )

        # 残差连接和层归一化
        refined_text = self.norm1(text_features + refined_text)

        # 前馈网络
        ff_output = self.ffn(refined_text)
        refined_text = self.norm2(refined_text + ff_output)

        return refined_text, attn_weights


class TextGuidedQueryEnhancement(nn.Module):
    """文本引导查询增强模块 (TG-QE)"""

    def __init__(self, query_dim, text_dim, num_queries):
        super(TextGuidedQueryEnhancement, self).__init__()
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, query_dim)
        self.text_proj = nn.Linear(text_dim, query_dim)

    def forward(self, text_features):
        # 获取初始查询
        batch_size = text_features.shape[0]
        query_init = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)

        # 投影文本特征到查询空间
        text_projected = self.text_proj(text_features)

        # 平均池化文本特征
        text_pooled = torch.mean(text_projected, dim=1, keepdim=True)

        # 增强查询
        enhanced_query = query_init + text_pooled.repeat(1, self.num_queries, 1)

        return enhanced_query


class HierarchicalTextEncoder(nn.Module):
    """分层文本编码器"""

    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super(HierarchicalTextEncoder, self).__init__()
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.text_encoder = self.clip_model.text_model
        self.text_projection = self.clip_model.text_projection

        # 可学习的权重参数
        self.alpha = nn.Parameter(torch.tensor(0.4))
        self.beta = nn.Parameter(torch.tensor(0.3))
        self.gamma = nn.Parameter(torch.tensor(0.3))

    def forward(self, global_text, local_text, detail_text):
        # 编码全局描述
        global_features = self._encode_text(global_text)

        # 编码局部描述
        local_features = self._encode_text(local_text)

        # 编码细节描述
        detail_features = self._encode_text(detail_text)

        # 加权融合
        fused_features = (
                self.alpha * global_features +
                self.beta * local_features +
                self.gamma * detail_features
        )

        return fused_features

    def _encode_text(self, text_inputs):
        # 提取文本特征
        text_outputs = self.text_encoder(
            input_ids=text_inputs['input_ids'],
            attention_mask=text_inputs['attention_mask']
        )

        # 使用注意力掩码计算加权平均
        attention_mask = text_inputs['attention_mask'].unsqueeze(-1)
        text_features = text_outputs.last_hidden_state * attention_mask
        text_features = torch.sum(text_features, dim=1) / torch.clamp(attention_mask.sum(dim=1), min=1e-9)

        # 投影到多模态空间
        text_features = self.text_projection(text_features)

        return text_features


class MultiModalDetector(nn.Module):
    """多模态大目标检测器"""

    def __init__(self, backbone, text_encoder, num_classes, num_queries=100, hidden_dim=256):
        super(MultiModalDetector, self).__init__()
        self.backbone = backbone
        self.text_encoder = text_encoder
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # 特征金字塔网络
        self.fpn = AdaptiveFPN([512, 1024, 2048], hidden_dim)

        # 图文协同模块
        self.tgfe = TextGuidedFeatureEnhancement(hidden_dim, hidden_dim, num_heads=8)
        self.vgtr = VisualGuidedTextRefinement(hidden_dim, hidden_dim, num_heads=8)
        self.tgqe = TextGuidedQueryEnhancement(hidden_dim, hidden_dim, num_queries)

        # Transformer检测头
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1
            ),
            num_layers=6
        )

        # 预测头
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for background
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # 查询初始化
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # 平台对齐权重参数
        self.sat_uav_weight = nn.Parameter(torch.tensor(0.5))
        self.uav_ground_weight = nn.Parameter(torch.tensor(0.3))
        self.sat_ground_weight = nn.Parameter(torch.tensor(0.2))

    def forward(self, images, text_inputs):
        # 提取图像特征
        visual_features = self._extract_visual_features(images)

        # 提取文本特征
        text_features = self._extract_text_features(text_inputs)

        # 文本引导特征增强
        enhanced_visual, visual_attn = self.tgfe(visual_features, text_features)

        # 视觉引导文本精化
        refined_text, text_attn = self.vgtr(text_features, enhanced_visual)

        # 文本引导查询增强
        query = self.tgqe(refined_text)

        # Transformer解码
        memory = enhanced_visual.flatten(2).permute(2, 0, 1)  # (H*W, B, C)
        tgt = query.permute(1, 0, 2)  # (N, B, C)

        decoded_features = self.transformer(tgt, memory)
        decoded_features = decoded_features.permute(1, 0, 2)  # (B, N, C)

        # 预测
        outputs_class = self.class_embed(decoded_features)
        outputs_coord = self.bbox_embed(decoded_features).sigmoid()

        # 跨平台一致性预测
        consistency_outputs = self._cross_platform_consistency(images, text_features)

        return {
            'pred_logits': outputs_class,
            'pred_boxes': outputs_coord,
            'visual_attn': visual_attn,
            'text_attn': text_attn,
            'consistency_outputs': consistency_outputs
        }

    def _extract_visual_features(self, images):
        # 使用骨干网络提取多尺度特征
        satellite_features = self.backbone(images['satellite'])
        uav_features = self.backbone(images['uav'])
        ground_features = self.backbone(images['ground'])
        infrared_features = self.backbone(images['infrared'])

        # 通过FPN融合多尺度特征
        satellite_fpn = self.fpn(satellite_features)
        uav_fpn = self.fpn(uav_features)
        ground_fpn = self.fpn(ground_features)
        infrared_fpn = self.fpn(infrared_features)

        # 跨平台对齐
        aligned_features = self._cross_platform_alignment(
            satellite_fpn, uav_fpn, ground_fpn, infrared_fpn
        )

        return aligned_features

    def _cross_platform_alignment(self, sat_feat, uav_feat, ground_feat, infrared_feat):
        """跨平台特征对齐"""
        # 卫星-无人机对齐
        sat_uav_aligned = self._platform_attention(sat_feat, uav_feat)

        # 无人机-地面对齐
        uav_ground_aligned = self._platform_attention(uav_feat, ground_feat)

        # 卫星-地面对齐
        sat_ground_aligned = self._platform_attention(sat_feat, ground_feat)

        # 红外特征处理
        infrared_proj = self._infrared_feature_projection(infrared_feat)

        # 加权融合
        fused_features = (
                self.sat_uav_weight * sat_uav_aligned +
                self.uav_ground_weight * uav_ground_aligned +
                self.sat_ground_weight * sat_ground_aligned +
                0.1 * infrared_proj  # 红外权重较小
        )

        return fused_features

    def _platform_attention(self, source_feat, target_feat):
        """平台间注意力对齐"""
        # 源特征作为查询，目标特征作为键值
        batch_size, channels, height, width = source_feat.shape
        source_flat = source_feat.view(batch_size, channels, -1).permute(0, 2, 1)
        target_flat = target_feat.view(batch_size, channels, -1).permute(0, 2, 1)

        # 计算注意力
        attn_output, _ = CrossPlatformAttention(channels, 8)(
            source_flat, target_flat, target_flat
        )

        # 恢复空间维度
        aligned = attn_output.permute(0, 2, 1).view(batch_size, channels, height, width)

        return aligned

    def _infrared_feature_projection(self, infrared_feat):
        """红外特征投影"""
        # 红外特征与可见光特征空间不同，需要特殊处理
        proj = nn.Sequential(
            nn.Conv2d(infrared_feat.shape[1], self.hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1)
        ).to(infrared_feat.device)

        return proj(infrared_feat)

    def _extract_text_features(self, text_inputs):
        # 提取分层文本特征
        global_text = {
            'input_ids': text_inputs['global']['input_ids'],
            'attention_mask': text_inputs['global']['attention_mask']
        }

        local_text = {
            'input_ids': text_inputs['local']['input_ids'],
            'attention_mask': text_inputs['local']['attention_mask']
        }

        detail_text = {
            'input_ids': text_inputs['detail']['input_ids'],
            'attention_mask': text_inputs['detail']['attention_mask']
        }

        text_features = self.text_encoder(global_text, local_text, detail_text)

        return text_features

    def _cross_platform_consistency(self, images, text_features):
        """跨平台一致性推理机制（完整实现）"""
        # 1. 各平台独立检测
        sat_outputs = self._single_platform_detection(images['satellite'], text_features)
        uav_outputs = self._single_platform_detection(images['uav'], text_features)
        ground_outputs = self._single_platform_detection(images['ground'], text_features)

        # 2. 特征级一致性约束
        feature_consistency_loss = self._feature_level_consistency(images)

        # 3. 预测级一致性约束
        prediction_consistency_loss = self._prediction_level_consistency(sat_outputs, uav_outputs, ground_outputs)

        # 4. 空间对齐约束
        spatial_consistency_loss = self._spatial_alignment_constraint(sat_outputs, uav_outputs, ground_outputs)

        # 总一致性损失
        consistency_loss = (
                0.4 * feature_consistency_loss +
                0.4 * prediction_consistency_loss +
                0.2 * spatial_consistency_loss
        )

        return {
            'sat_outputs': sat_outputs,
            'uav_outputs': uav_outputs,
            'ground_outputs': ground_outputs,
            'consistency_loss': consistency_loss,
            'feature_consistency': feature_consistency_loss,
            'prediction_consistency': prediction_consistency_loss,
            'spatial_consistency': spatial_consistency_loss
        }

    def _feature_level_consistency(self, images):
        """特征级一致性约束"""
        # 提取各平台特征
        sat_features = self.backbone(images['satellite'])
        uav_features = self.backbone(images['uav'])
        ground_features = self.backbone(images['ground'])

        # 特征对齐损失（最大均值差异）
        mmd_loss = 0
        for sat_feat, uav_feat, ground_feat in zip(sat_features, uav_features, ground_features):
            mmd_loss += self._compute_mmd(sat_feat, uav_feat)
            mmd_loss += self._compute_mmd(uav_feat, ground_feat)
            mmd_loss += self._compute_mmd(sat_feat, ground_feat)

        return mmd_loss / (3 * len(sat_features))

    def _compute_mmd(self, x, y):
        """计算最大均值差异（MMD）"""

        # 高斯核函数
        def gaussian_kernel(x, y, sigma=5.0):
            x_norm = torch.sum(x ** 2, dim=1, keepdim=True)
            y_norm = torch.sum(y ** 2, dim=1, keepdim=True)
            dist = x_norm - 2 * torch.matmul(x, y.t()) + y_norm.t()
            return torch.exp(-dist / (2 * sigma ** 2))

        xx = gaussian_kernel(x, x)
        yy = gaussian_kernel(y, y)
        xy = gaussian_kernel(x, y)

        return torch.mean(xx) + torch.mean(yy) - 2 * torch.mean(xy)

    def _prediction_level_consistency(self, sat_outputs, uav_outputs, ground_outputs):
        """预测级一致性约束"""
        # 1. 类别概率一致性
        sat_probs = F.softmax(sat_outputs['pred_logits'], dim=-1)
        uav_probs = F.softmax(uav_outputs['pred_logits'], dim=-1)
        ground_probs = F.softmax(ground_outputs['pred_logits'], dim=-1)

        # 对称KL散度
        kl_loss = (F.kl_div(F.log_softmax(sat_probs, dim=-1), uav_probs, reduction='batchmean') +
                   F.kl_div(F.log_softmax(uav_probs, dim=-1), sat_probs, reduction='batchmean') +
                   F.kl_div(F.log_softmax(sat_probs, dim=-1), ground_probs, reduction='batchmean') +
                   F.kl_div(F.log_softmax(ground_probs, dim=-1), sat_probs, reduction='batchmean') +
                   F.kl_div(F.log_softmax(uav_probs, dim=-1), ground_probs, reduction='batchmean') +
                   F.kl_div(F.log_softmax(ground_probs, dim=-1), uav_probs, reduction='batchmean')) / 6

        # 2. 边界框一致性（仅在有对应目标时）
        box_loss = 0
        count = 0
        for i in range(sat_probs.shape[1]):
            if torch.max(sat_probs[:, i]) > 0.5 and torch.max(uav_probs[:, i]) > 0.5:
                sat_box = sat_outputs['pred_boxes'][:, i, :]
                uav_box = uav_outputs['pred_boxes'][:, i, :]
                box_loss += F.smooth_l1_loss(sat_box, uav_box)
                count += 1

            if torch.max(sat_probs[:, i]) > 0.5 and torch.max(ground_probs[:, i]) > 0.5:
                sat_box = sat_outputs['pred_boxes'][:, i, :]
                ground_box = ground_outputs['pred_boxes'][:, i, :]
                box_loss += F.smooth_l1_loss(sat_box, ground_box)
                count += 1

            if torch.max(uav_probs[:, i]) > 0.5 and torch.max(ground_probs[:, i]) > 0.5:
                uav_box = uav_outputs['pred_boxes'][:, i, :]
                ground_box = ground_outputs['pred_boxes'][:, i, :]
                box_loss += F.smooth_l1_loss(uav_box, ground_box)
                count += 1

        box_loss = box_loss / count if count > 0 else 0

        return 0.6 * kl_loss + 0.4 * box_loss

    def _spatial_alignment_constraint(self, sat_outputs, uav_outputs, ground_outputs):
        """完整的空间对齐约束实现"""
        total_error = 0.0
        valid_pairs = 0

        # 1. 卫星-无人机对齐
        sat_points, uav_points = self._get_corresponding_points(sat_outputs, uav_outputs)
        if len(sat_points) >= 4:  # 至少需要4个点计算单应性
            H_sat2uav, mask = cv2.findHomography(
                np.array(sat_points, dtype=np.float32),
                np.array(uav_points, dtype=np.float32),
                cv2.RANSAC, 5.0
            )
            if H_sat2uav is not None:
                error = self._compute_reprojection_error(sat_points, uav_points, H_sat2uav)
                total_error += error
                valid_pairs += 1

        # 2. 无人机-地面对齐
        uav_points, ground_points = self._get_corresponding_points(uav_outputs, ground_outputs)
        if len(uav_points) >= 4:
            H_uav2ground, mask = cv2.findHomography(
                np.array(uav_points, dtype=np.float32),
                np.array(ground_points, dtype=np.float32),
                cv2.RANSAC, 5.0
            )
            if H_uav2ground is not None:
                error = self._compute_reprojection_error(uav_points, ground_points, H_uav2ground)
                total_error += error
                valid_pairs += 1

        # 如果没有有效的点对，返回一个较大的惩罚值
        if valid_pairs == 0:
            return 10.0
        return total_error / valid_pairs

    def _get_corresponding_points(self, outputs1, outputs2):
        """获取对应目标的中心点 - 完整实现"""
        points1 = []
        points2 = []

        # 获取预测概率
        probs1 = F.softmax(outputs1['pred_logits'], dim=-1)
        probs2 = F.softmax(outputs2['pred_logits'], dim=-1)

        # 假设batch_size=1 (评估时)
        boxes1 = outputs1['pred_boxes'][0].detach().cpu().numpy()
        boxes2 = outputs2['pred_boxes'][0].detach().cpu().numpy()

        # 获取预测类别
        conf1, labels1 = torch.max(probs1[0], dim=-1)
        conf2, labels2 = torch.max(probs2[0], dim=-1)

        # 遍历所有查询
        for i in range(len(labels1)):
            # 只考虑置信度高的检测结果
            if conf1[i] > 0.7 and labels1[i] < self.num_classes:
                # 在第二个平台的输出中寻找相同类别的检测结果
                for j in range(len(labels2)):
                    if conf2[j] > 0.7 and labels2[j] == labels1[i]:
                        # 计算边界框中心点
                        box1 = boxes1[i]
                        box2 = boxes2[j]

                        center1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
                        center2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]

                        points1.append(center1)
                        points2.append(center2)
                        break  # 假设一对一匹配，找到第一个匹配就跳出

        return points1, points2

    def _compute_reprojection_error(self, src_points, dst_points, H):
        """计算重投影误差"""
        src_pts = np.array(src_points, dtype=np.float32).reshape(-1, 1, 2)
        dst_pts = np.array(dst_points, dtype=np.float32).reshape(-1, 1, 2)

        # 使用单应性矩阵变换源点
        projected = cv2.perspectiveTransform(src_pts, H)

        # 计算欧氏距离
        errors = np.linalg.norm(projected - dst_pts, axis=2)
        return np.mean(errors)

    def _single_platform_detection(self, image, text_features):
        """单平台检测"""
        # 提取特征
        features = self.backbone(image)
        fpn_features = self.fpn(features)

        # 文本引导特征增强
        enhanced_features, _ = self.tgfe(fpn_features, text_features)

        # 文本引导查询增强
        query = self.tgqe(text_features)

        # Transformer解码
        memory = enhanced_features.flatten(2).permute(2, 0, 1)
        tgt = query.permute(1, 0, 2)

        decoded_features = self.transformer(tgt, memory)
        decoded_features = decoded_features.permute(1, 0, 2)

        # 预测
        class_pred = self.class_embed(decoded_features)
        box_pred = self.bbox_embed(decoded_features).sigmoid()

        return {
            'pred_logits': class_pred,
            'pred_boxes': box_pred
        }

    def _consistency_constraint(self, sat_outputs, uav_outputs, ground_outputs):
        """跨平台一致性约束"""
        # 获取预测概率
        sat_probs = F.softmax(sat_outputs['pred_logits'], dim=-1)
        uav_probs = F.softmax(uav_outputs['pred_logits'], dim=-1)
        ground_probs = F.softmax(ground_outputs['pred_logits'], dim=-1)

        # 计算KL散度作为一致性损失
        kl_loss = nn.KLDivLoss(reduction='batchmean')

        # 卫星与无人机一致性
        sat_uav_loss = kl_loss(F.log_softmax(sat_probs, dim=-1), uav_probs)

        # 无人机与地面一致性
        uav_ground_loss = kl_loss(F.log_softmax(uav_probs, dim=-1), ground_probs)

        # 卫星与地面一致性
        sat_ground_loss = kl_loss(F.log_softmax(sat_probs, dim=-1), ground_probs)

        # 加权总一致性损失
        total_loss = (
                0.5 * sat_uav_loss +
                0.3 * uav_ground_loss +
                0.2 * sat_ground_loss
        )

        return total_loss


class MLP(nn.Module):
    """简单的多层感知机"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PromptOptimizationModule(nn.Module):
    """完整实现的提示工程优化模块"""

    def __init__(self, llm_model="OpenGVLab/InternVL-78B", action_dim=128, state_dim=768):
        super(PromptOptimizationModule, self).__init__()

        # 加载大语言模型
        # self.llm = AutoModel.from_pretrained(llm_model)
        # self.tokenizer = AutoTokenizer.from_pretrained(llm_model)

        self.llm = AutoModel.from_pretrained("bert-large-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
        self.token_attention = nn.Linear(self.llm.config.hidden_size, 1)

        # 强化学习策略网络
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

        # 价值网络
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        # 动作空间
        self.action_types = ["add_keyword", "remove_keyword", "emphasize_keyword", "add_context"]
        self.action_dim = action_dim
        self.state_dim = state_dim

        # 动作到token的映射
        self.keyword_library = self._initialize_keyword_library()

    def forward(self, prompt_template, category, platform, current_state=None):
        # 填充模板
        initial_prompt = self._fill_template(prompt_template, category, platform)

        # 生成token嵌入
        inputs = self.tokenizer(initial_prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = self.llm(**inputs)
        token_embeddings = outputs.last_hidden_state
        pooled_state = torch.mean(token_embeddings, dim=1)  # 状态表示

        # 强化学习决策
        if current_state is not None:
            state = torch.cat([pooled_state, current_state], dim=1)
        else:
            state = pooled_state

        # 生成动作概率分布
        action_logits = self.policy_net(state)
        action_probs = F.softmax(action_logits, dim=-1)

        # 采样动作
        if self.training:
            dist = Categorical(action_probs)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
        else:
            action = torch.argmax(action_probs, dim=-1)
            action_log_prob = None

        # 执行动作修改提示
        optimized_prompt = self._apply_action(initial_prompt, action.item())

        # 价值估计
        state_value = self.value_net(state)

        return {
            'initial_prompt': initial_prompt,
            'optimized_prompt': optimized_prompt,
            'action_probs': action_probs,
            'action': action,
            'action_log_prob': action_log_prob,
            'state_value': state_value,
            'token_embeddings': token_embeddings
        }

    def _initialize_keyword_library(self):
        """初始化关键词库"""
        return {
            'airport': ["跑道", "航站楼", "塔台", "滑行道", "停机坪", "导航灯", "机库", "控制塔"],
            'oil_field': ["钻井平台", "储油罐", "输油管", "炼油厂", "泵站", "阀门", "分离器", "火炬"],
            'port': ["码头", "起重机", "集装箱", "仓库", "防波堤", "系船柱", "船闸", "堆场"]
        }

    def _apply_action(self, prompt, action_index):
        """应用RL动作修改提示"""
        # 动作类型索引
        action_type_idx = action_index % len(self.action_types)
        action_type = self.action_types[action_type_idx]

        # 类别索引
        cat_idx = (action_index // len(self.action_types)) % len(self.keyword_library)
        category = list(self.keyword_library.keys())[cat_idx]

        if action_type == "add_keyword":
            # 添加关键词
            keywords = self.keyword_library[category]
            keyword_idx = action_index % len(keywords)
            new_keyword = keywords[keyword_idx]
            return prompt + f"，特别是{new_keyword}"

        elif action_type == "remove_keyword":
            # 移除关键词 - 基于语义相似度
            words = prompt.split("，")
            if len(words) > 1:
                # 计算每个词的嵌入相似度
                embeddings = self.tokenizer(words, return_tensors="pt", padding=True)["input_ids"]
                with torch.no_grad():
                    outputs = self.llm(embeddings)
                    word_embeddings = torch.mean(outputs.last_hidden_state, dim=1)

                # 计算与整体提示的相似度
                prompt_embedding = torch.mean(word_embeddings, dim=0)
                similarities = F.cosine_similarity(word_embeddings, prompt_embedding.unsqueeze(0))

                # 移除最不相关的词
                min_idx = torch.argmin(similarities)
                return "，".join([word for i, word in enumerate(words) if i != min_idx])
            return prompt

        elif action_type == "emphasize_keyword":
            # 强调关键词 - 增加权重
            keywords = self.keyword_library[category]
            keyword_idx = action_index % len(keywords)
            keyword = keywords[keyword_idx]
            return prompt.replace(keyword, f"**{keyword}**")

        elif action_type == "add_context":
            # 添加上下文
            contexts = [
                "在复杂背景环境中",
                "考虑不同光照条件下",
                "注意云雾遮挡情况",
                "多尺度特征融合视角",
                "在低能见度场景中",
                "结合热红外特征"
            ]
            context_idx = action_index % len(contexts)
            context = contexts[context_idx]
            return f"{context}下{prompt}"

        return prompt

    def compute_loss(self, batch, rewards):
        """计算强化学习损失"""
        # 策略损失
        policy_loss = -torch.sum(batch['action_log_prob'] * rewards)

        # 价值损失
        value_loss = F.mse_loss(batch['state_value'], rewards.unsqueeze(1))

        # 熵正则化
        entropy = -torch.sum(batch['action_probs'] * torch.log(batch['action_probs'] + 1e-8))
        entropy_bonus = 0.01 * entropy

        total_loss = policy_loss + 0.5 * value_loss - entropy_bonus

        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy
        }

def build_model(num_classes, backbone_name='resnet50', text_encoder_name='openai/clip-vit-base-patch32'):
    """构建多模态检测模型"""
    # 加载预训练骨干网络
    if backbone_name == 'resnet50':
        backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        # 移除最后的全连接层
        backbone = nn.Sequential(*list(backbone.children())[:-2])
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    # 构建文本编码器
    text_encoder = HierarchicalTextEncoder(text_encoder_name)

    # 构建检测模型
    model = MultiModalDetector(backbone, text_encoder, num_classes)


    return model