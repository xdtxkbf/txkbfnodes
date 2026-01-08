"""
MaskToSEGS Advanced - 高级遮罩到SEGS转换节点
提供智能距离聚类合并和自定义输出形状控制功能

作者: Impact Pack Extension
"""

import numpy as np
import torch
import cv2
import logging
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import fcluster, linkage
from nodes import MAX_RESOLUTION


def get_contour_center(contour):
    """获取轮廓的中心点"""
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        # 如果面积为0，使用边界框中心
        x, y, w, h = cv2.boundingRect(contour)
        cx = x + w // 2
        cy = y + h // 2
    return cx, cy


def cluster_by_distance(contours, merge_distance, image_diagonal):
    """
    根据距离对轮廓进行聚类
    
    参数:
        contours: 轮廓列表
        merge_distance: 合并距离阈值(相对于图像对角线的百分比，0-100)
        image_diagonal: 图像对角线长度
    
    返回:
        聚类标签列表
    """
    if len(contours) <= 1:
        return list(range(len(contours)))
    
    # 获取所有轮廓的中心点
    centers = np.array([get_contour_center(c) for c in contours])
    
    # 计算实际合并距离(基于图像对角线的百分比)
    actual_distance = (merge_distance / 100.0) * image_diagonal
    
    if actual_distance <= 0:
        # 不合并，每个轮廓单独一组
        return list(range(len(contours)))
    
    # 使用层次聚类
    if len(centers) >= 2:
        # 计算距离矩阵
        distances = cdist(centers, centers, metric='euclidean')
        
        # 使用condensed distance matrix进行层次聚类
        condensed_dist = distances[np.triu_indices(len(centers), k=1)]
        
        if len(condensed_dist) > 0:
            # 进行层次聚类
            Z = linkage(condensed_dist, method='average')
            # 根据距离阈值切分聚类
            labels = fcluster(Z, t=actual_distance, criterion='distance')
            return (labels - 1).tolist()  # 转换为0-indexed
    
    return list(range(len(contours)))


def adjust_crop_region_for_shape(bbox, crop_factor, img_width, img_height, 
                                  target_aspect_ratio=1.0, 
                                  shape_mode="auto",
                                  min_size=None):
    """
    根据目标形状调整裁剪区域
    
    参数:
        bbox: 原始边界框 (x1, y1, x2, y2)
        crop_factor: 裁剪因子 - 表示裁剪区域边长相对于bbox的放大倍数
        img_width, img_height: 图像尺寸
        target_aspect_ratio: 目标宽高比 (宽/高)
        shape_mode: 形状模式 ("auto", "square", "horizontal", "vertical", "custom")
        min_size: 最小尺寸
    
    返回:
        调整后的裁剪区域 [x1, y1, x2, y2]
    
    注意: crop_factor在所有shape_mode下都保持一致的上下文拓展程度
    """
    x1, y1, x2, y2 = bbox
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    bbox_area = bbox_w * bbox_h
    
    # 计算目标面积（保持crop_factor的语义一致性）
    # crop_factor表示线性尺寸的放大倍数，面积为其平方
    target_area = bbox_area * (crop_factor ** 2)
    
    # 根据形状模式确定目标宽高比
    if shape_mode == "square":
        final_aspect_ratio = 1.0
    elif shape_mode == "horizontal":
        # 横向矩形，使用指定的宽高比，但至少保持原始比例
        final_aspect_ratio = max(bbox_w / bbox_h, target_aspect_ratio)
    elif shape_mode == "vertical":
        # 纵向矩形，使用指定的宽高比的倒数，但至少保持原始比例
        final_aspect_ratio = min(bbox_w / bbox_h, 1.0 / target_aspect_ratio)
    elif shape_mode == "custom":
        final_aspect_ratio = target_aspect_ratio
    else:  # "auto"
        # 保持原始bbox的宽高比
        final_aspect_ratio = bbox_w / bbox_h if bbox_h > 0 else 1.0
    
    # 根据目标面积和宽高比计算最终尺寸
    # area = w * h, ratio = w / h
    # => w = sqrt(area * ratio), h = sqrt(area / ratio)
    crop_w = np.sqrt(target_area * final_aspect_ratio)
    crop_h = np.sqrt(target_area / final_aspect_ratio)
    
    if min_size is not None:
        if crop_w < min_size or crop_h < min_size:
            # 保持宽高比的同时满足最小尺寸
            scale = max(min_size / crop_w, min_size / crop_h)
            crop_w *= scale
            crop_h *= scale
    
    # 计算中心点
    center_x = x1 + bbox_w / 2
    center_y = y1 + bbox_h / 2
    
    # 根据中心点计算新的裁剪区域
    new_x1 = int(center_x - crop_w / 2)
    new_y1 = int(center_y - crop_h / 2)
    new_x2 = int(center_x + crop_w / 2)
    new_y2 = int(center_y + crop_h / 2)
    
    # 确保在图像范围内
    if new_x1 < 0:
        new_x2 -= new_x1
        new_x1 = 0
    if new_y1 < 0:
        new_y2 -= new_y1
        new_y1 = 0
    if new_x2 > img_width:
        new_x1 -= (new_x2 - img_width)
        new_x2 = img_width
    if new_y2 > img_height:
        new_y1 -= (new_y2 - img_height)
        new_y2 = img_height
    
    # 最终边界检查
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(img_width, new_x2)
    new_y2 = min(img_height, new_y2)
    
    return [new_x1, new_y1, new_x2, new_y2]


def merge_contours_to_mask(contours, mask_shape):
    """将多个轮廓合并为单个遮罩"""
    merged_mask = np.zeros(mask_shape, dtype=np.uint8)
    cv2.drawContours(merged_mask, contours, -1, 255, -1)
    return merged_mask


class SEG:
    """SEG数据结构类"""
    def __init__(self, cropped_image, cropped_mask, confidence, crop_region, bbox, label, control_net_wrapper):
        self.cropped_image = cropped_image
        self.cropped_mask = cropped_mask
        self.confidence = confidence
        self.crop_region = crop_region  # (x1, y1, x2, y2)
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.label = label
        self.control_net_wrapper = control_net_wrapper


def mask_to_segs_advanced(mask, merge_distance, crop_factor, bbox_fill, drop_size,
                          shape_mode="auto", target_aspect_ratio=1.0,
                          label='A', crop_min_size=None, detailer_hook=None, 
                          is_contour=True):
    """
    高级遮罩到SEGS转换
    
    参数:
        mask: 输入遮罩
        merge_distance: 合并距离阈值(0-100，基于图像对角线百分比)
        crop_factor: 裁剪因子
        bbox_fill: 是否填充边界框
        drop_size: 最小尺寸过滤
        shape_mode: 形状模式 ("auto", "square", "horizontal", "vertical", "custom")
        target_aspect_ratio: 目标宽高比(仅用于custom模式)
        label: SEGS标签
        crop_min_size: 裁剪最小尺寸
        detailer_hook: detailer钩子
        is_contour: 是否使用轮廓遮罩
    
    返回:
        (shape, segs_list)
    """
    drop_size = max(drop_size, 1)
    
    if mask is None:
        logging.info("[mask_to_segs_advanced] Cannot operate: MASK is empty.")
        return ([],)
    
    if isinstance(mask, np.ndarray):
        pass
    else:
        try:
            mask = mask.numpy()
        except AttributeError:
            logging.info("[mask_to_segs_advanced] Cannot operate: MASK is not a NumPy array or Tensor.")
            return ([],)
    
    if mask is None:
        logging.info("[mask_to_segs_advanced] Cannot operate: MASK is empty.")
        return ([],)
    
    result = []
    
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, axis=0)
    
    for i in range(mask.shape[0]):
        mask_i = mask[i]
        img_height, img_width = mask_i.shape
        image_diagonal = np.sqrt(img_width**2 + img_height**2)
        
        # 提取轮廓
        mask_i_uint8 = (mask_i * 255.0).astype(np.uint8)
        contours, ctree = cv2.findContours(mask_i_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤有效轮廓(只保留顶层轮廓)
        valid_contours = []
        valid_indices = []
        for j, contour in enumerate(contours):
            hierarchy = ctree[0][j]
            if hierarchy[3] != -1:  # 跳过子轮廓
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w > drop_size and h > drop_size:
                valid_contours.append(contour)
                valid_indices.append(j)
        
        if not valid_contours:
            continue
        
        # 根据距离进行聚类
        cluster_labels = cluster_by_distance(valid_contours, merge_distance, image_diagonal)
        
        # 按聚类分组
        clusters = {}
        for idx, label_id in enumerate(cluster_labels):
            if label_id not in clusters:
                clusters[label_id] = []
            clusters[label_id].append(valid_contours[idx])
        
        # 为每个聚类创建SEG
        for cluster_id, cluster_contours in clusters.items():
            # 合并同一聚类中的所有轮廓
            merged_mask_uint8 = merge_contours_to_mask(cluster_contours, mask_i.shape)
            merged_mask_float = (merged_mask_uint8 / 255.0).astype(np.float32)
            
            # 计算合并后的边界框
            all_points = np.vstack(cluster_contours)
            x, y, w, h = cv2.boundingRect(all_points)
            bbox = (x, y, x + w, y + h)
            
            # 根据形状模式调整裁剪区域
            crop_region = adjust_crop_region_for_shape(
                bbox, crop_factor, img_width, img_height,
                target_aspect_ratio, shape_mode, crop_min_size
            )
            
            if detailer_hook is not None:
                crop_region = detailer_hook.post_crop_region(
                    img_width, img_height, bbox, crop_region
                )
            
            x1, y1, x2, y2 = crop_region
            if x2 - x1 > 0 and y2 - y1 > 0:
                if is_contour:
                    mask_src = merged_mask_float
                else:
                    mask_src = mask_i * merged_mask_float
                
                cropped_mask = np.array(mask_src[y1:y2, x1:x2])
                
                if bbox_fill:
                    bx1, by1, bx2, by2 = bbox
                    cx1 = bx1 - x1
                    cx2 = bx2 - x1
                    cy1 = by1 - y1
                    cy2 = by2 - y1
                    # 确保索引在范围内
                    cx1 = max(0, cx1)
                    cy1 = max(0, cy1)
                    cx2 = min(cropped_mask.shape[1], cx2)
                    cy2 = min(cropped_mask.shape[0], cy2)
                    if cx2 > cx1 and cy2 > cy1:
                        cropped_mask[cy1:cy2, cx1:cx2] = 1.0
                
                if cropped_mask is not None:
                    cropped_mask = torch.clip(torch.from_numpy(cropped_mask), 0, 1.0)
                    item = SEG(None, cropped_mask.numpy(), 1.0, crop_region, bbox, label, None)
                    result.append(item)
    
    if not result:
        logging.info("[mask_to_segs_advanced] Empty mask.")
    
    logging.info(f"[mask_to_segs_advanced] # of Detected SEGS: {len(result)}")
    
    return (mask.shape[1], mask.shape[2]), result


class MaskToSEGSAdvanced:
    """
    高级遮罩到SEGS转换节点
    
    功能:
    1. 智能距离聚类 - 根据遮罩区域的距离自动分组合并
       例如: 4个眼睛遮罩会根据距离自动分成2组(每人2个眼睛)
    
    2. 自定义输出形状 - 可以控制输出SEG的裁剪形状
       - auto: 保持原始比例
       - square: 强制正方形
       - horizontal: 横向矩形
       - vertical: 纵向矩形  
       - custom: 自定义宽高比
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "merge_distance": ("FLOAT", {
                    "default": 10.0, 
                    "min": 0.0, 
                    "max": 100.0, 
                    "step": 0.5,
                    "tooltip": "合并距离阈值(图像对角线的百分比)。0=不合并，越大合并范围越大。例如：10表示距离小于图像对角线10%的遮罩会被合并为同一个SEG"
                }),
                "crop_factor": ("FLOAT", {
                    "default": 3.0, 
                    "min": 1.0, 
                    "max": 100, 
                    "step": 0.1,
                    "tooltip": "裁剪区域边长相对于边界框的放大倍数。1.5表示保留1.5倍的上下文，无论shape_mode如何设置，该倍数保持一致"
                }),
                "bbox_fill": ("BOOLEAN", {
                    "default": False, 
                    "label_on": "enabled", 
                    "label_off": "disabled",
                    "tooltip": "是否填充边界框区域"
                }),
                "drop_size": ("INT", {
                    "min": 1, 
                    "max": MAX_RESOLUTION, 
                    "step": 1, 
                    "default": 10,
                    "tooltip": "小于此尺寸的遮罩区域将被过滤掉"
                }),
                "contour_fill": ("BOOLEAN", {
                    "default": False, 
                    "label_on": "enabled", 
                    "label_off": "disabled",
                    "tooltip": "是否使用轮廓填充遮罩"
                }),
                "shape_mode": (["auto", "square", "horizontal", "vertical", "custom"], {
                    "default": "auto",
                    "tooltip": "输出形状模式: auto=保持原比例, square=正方形, horizontal=横向, vertical=纵向, custom=自定义比例"
                }),
            },
            "optional": {
                "target_aspect_ratio": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.1, 
                    "max": 10.0, 
                    "step": 0.1,
                    "tooltip": "目标宽高比(宽/高)，仅在shape_mode为custom或horizontal/vertical时生效"
                }),
            }
        }
    
    RETURN_TYPES = ("SEGS",)
    FUNCTION = "doit"
    CATEGORY = "txkbfnodes/Mask"
    DESCRIPTION = """高级遮罩到SEGS转换节点

特性:
• 智能距离聚类 - 自动根据位置对遮罩进行分组
  例如: 输入4个眼睛遮罩，会自动分成2组(每人一组)
  
• 自定义输出形状 - 控制SEG裁剪区域的形状
  - auto: 保持检测到的自然形状
  - square: 强制正方形输出(适合脸部等)
  - horizontal/vertical: 横向/纵向矩形
  - custom: 自定义宽高比

merge_distance参数说明:
- 0: 不合并，每个独立遮罩区域为单独的SEG
- 10: 距离小于图像对角线10%的遮罩会被合并
- 越大合并范围越广
"""
    
    @staticmethod
    def doit(mask, merge_distance, crop_factor, bbox_fill, drop_size, 
             contour_fill=False, shape_mode="auto", target_aspect_ratio=1.0):
        
        # 转换为2D遮罩
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        
        if len(mask.shape) == 3:
            mask = mask[0]  # 取第一帧
        
        result = mask_to_segs_advanced(
            mask, 
            merge_distance=merge_distance,
            crop_factor=crop_factor, 
            bbox_fill=bbox_fill, 
            drop_size=drop_size,
            shape_mode=shape_mode,
            target_aspect_ratio=target_aspect_ratio,
            is_contour=contour_fill
        )
        
        return (result,)


class MaskToSEGSAdvanced_for_AnimateDiff:
    """
    用于AnimateDiff的高级遮罩到SEGS转换节点
    支持批量遮罩处理
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "merge_distance": ("FLOAT", {
                    "default": 10.0, 
                    "min": 0.0, 
                    "max": 100.0, 
                    "step": 0.5,
                    "tooltip": "合并距离阈值(图像对角线的百分比)"
                }),
                "crop_factor": ("FLOAT", {
                    "default": 3.0, 
                    "min": 1.0, 
                    "max": 100, 
                    "step": 0.1
                }),
                "bbox_fill": ("BOOLEAN", {
                    "default": False, 
                    "label_on": "enabled", 
                    "label_off": "disabled"
                }),
                "drop_size": ("INT", {
                    "min": 1, 
                    "max": MAX_RESOLUTION, 
                    "step": 1, 
                    "default": 10
                }),
                "contour_fill": ("BOOLEAN", {
                    "default": False, 
                    "label_on": "enabled", 
                    "label_off": "disabled"
                }),
                "shape_mode": (["auto", "square", "horizontal", "vertical", "custom"], {
                    "default": "auto"
                }),
            },
            "optional": {
                "target_aspect_ratio": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.1, 
                    "max": 10.0, 
                    "step": 0.1
                }),
            }
        }
    
    RETURN_TYPES = ("SEGS",)
    FUNCTION = "doit"
    CATEGORY = "txkbfnodes/Mask"
    DESCRIPTION = "用于视频/AnimateDiff的高级遮罩到SEGS转换"
    
    @staticmethod
    def doit(mask, merge_distance, crop_factor, bbox_fill, drop_size, 
             contour_fill=False, shape_mode="auto", target_aspect_ratio=1.0):
        
        # 转换为NumPy
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        
        # 检查是否为批量遮罩
        if len(mask.shape) > 2 and mask.shape[0] > 1:
            # 合并所有帧的遮罩
            combined_mask = mask.max(axis=0)
            
            segs = mask_to_segs_advanced(
                combined_mask, 
                merge_distance=merge_distance,
                crop_factor=crop_factor, 
                bbox_fill=bbox_fill, 
                drop_size=drop_size,
                shape_mode=shape_mode,
                target_aspect_ratio=target_aspect_ratio,
                is_contour=False
            )
            
            # 为每个SEG创建批量遮罩
            new_segs = []
            for seg in segs[1]:
                x1, y1, x2, y2 = seg.crop_region
                cropped_mask = mask[:, y1:y2, x1:x2]
                item = SEG(None, cropped_mask, 1.0, seg.crop_region, seg.bbox, 'A', None)
                new_segs.append(item)
            
            return ((segs[0], new_segs),)
        
        # 单帧处理
        if len(mask.shape) == 3:
            mask = mask[0]
        
        result = mask_to_segs_advanced(
            mask, 
            merge_distance=merge_distance,
            crop_factor=crop_factor, 
            bbox_fill=bbox_fill, 
            drop_size=drop_size,
            shape_mode=shape_mode,
            target_aspect_ratio=target_aspect_ratio,
            is_contour=contour_fill
        )
        
        return (result,)


# 导出节点
NODE_CLASS_MAPPINGS = {
    "MaskToSEGSAdvanced": MaskToSEGSAdvanced,
    "MaskToSEGSAdvanced_for_AnimateDiff": MaskToSEGSAdvanced_for_AnimateDiff,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskToSEGSAdvanced": "MASK to SEGS (Advanced)",
    "MaskToSEGSAdvanced_for_AnimateDiff": "MASK to SEGS (Advanced) for Video",
}
