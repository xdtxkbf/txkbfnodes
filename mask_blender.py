import torch

class MaskBlender:
    """
    Mask 混合节点 - 支持多种混合模式
    输入多个 masks，混合成一个 mask
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "blend_mode": ([
                    "average",       # 平均
                    "add",           # 相加
                    "multiply",      # 相乘
                ],),
                "normalize": ("BOOLEAN", {"default": True}),
            },
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("blended_mask",)
    FUNCTION = "blend_masks"
    CATEGORY = "mask"
    
    def blend_masks(self, masks, blend_mode="average", normalize=True):
        """
        混合多个 masks
        
        Args:
            masks: 输入 mask 张量 (B, H, W) 或 (B, 1, H, W)
            blend_mode: 混合模式
            normalize: 是否归一化到 [0, 1]
            
        Returns:
            blended_mask: 混合后的 mask
        """
        # 确保所有 masks 在同一设备上
        device = masks.device
        
        # 确保 masks 是 3D 张量 (B, H, W)
        if len(masks.shape) == 4:
            masks = masks.squeeze(1)
        
        # 如果只有一个 mask，直接返回
        if masks.shape[0] == 1:
            return (masks,)
        
        # 初始化结果为第一个 mask，确保在正确设备上
        result = masks[0].clone().to(device)
        
        # 逐个混合后续的 masks
        for i in range(1, masks.shape[0]):
            current_mask = masks[i].to(device)
            
            if blend_mode == "add":
                # 相加模式
                result = result + current_mask
                
            elif blend_mode == "multiply":
                # 相乘模式
                result = result * current_mask
                
            elif blend_mode == "average":
                # 平均模式 - 累积求和，最后除以总数
                result = result + current_mask
        
        # 如果是平均模式，除以 mask 数量
        if blend_mode == "average":
            result = result / masks.shape[0]
        
        # 归一化到 [0, 1]
        if normalize:
            result = torch.clamp(result, 0.0, 1.0)
        
        # 返回时保持 3D 格式 (1, H, W)，确保在同一设备
        result = result.unsqueeze(0).to(device)
        
        return (result,)


# 注册节点
NODE_CLASS_MAPPINGS = {
    "MaskBlender": MaskBlender,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskBlender": "Mask Blender (Mask混合)",
}
