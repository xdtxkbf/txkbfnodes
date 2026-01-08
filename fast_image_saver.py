import os
import torch
import numpy as np
from PIL import Image
from datetime import datetime
import folder_paths

class FastImageSaver:
    """
    快速图像保存节点 - 支持 WebP 压缩
    输入图像,保存到本地,并返回保存路径
    """
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "format": (["webp", "png", "jpg"],),
                "webp_quality": ("INT", {
                    "default": 90,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "slider"
                }),
            },
            "optional": {
                "subfolder": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("file_path", "filename",)
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "image/io"
    
    def save_images(self, images, filename_prefix="ComfyUI", format="webp", webp_quality=90, subfolder=""):
        """
        保存图像并返回文件路径
        
        Args:
            images: 输入图像张量 (B, H, W, C)
            filename_prefix: 文件名前缀
            format: 保存格式 (webp, png, jpg)
            webp_quality: WebP 压缩质量 (1-100)
            subfolder: 子文件夹名称
            
        Returns:
            file_path: 完整文件路径
            filename: 文件名
        """
        # 确定输出目录
        if subfolder:
            output_path = os.path.join(self.output_dir, subfolder)
        else:
            output_path = self.output_dir
            
        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)
        
        results = []
        filenames = []
        
        # 处理批量图像
        for batch_number, image in enumerate(images):
            # 转换张量到 numpy 数组 (H, W, C)
            # ComfyUI 图像格式: [0, 1] 浮点数
            i = 255. * image.cpu().numpy()
            img = np.clip(i, 0, 255).astype(np.uint8)
            
            # 转换为 PIL Image
            pil_image = Image.fromarray(img)
            
            # 生成文件名 (带时间戳确保唯一性)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            if len(images) > 1:
                filename = f"{filename_prefix}_{timestamp}_{batch_number:04d}.{format}"
            else:
                filename = f"{filename_prefix}_{timestamp}.{format}"
            
            file_path = os.path.join(output_path, filename)
            
            # 保存图像 - 针对不同格式优化
            if format == "webp":
                # WebP 格式 - 高质量压缩,最快速度
                pil_image.save(
                    file_path,
                    format="WEBP",
                    quality=webp_quality,
                    method=6,  # 0=最快,6=默认平衡
                    lossless=False
                )
            elif format == "png":
                # PNG 格式 - 无损压缩
                pil_image.save(
                    file_path,
                    format="PNG",
                    compress_level=6,  # 0-9, 6是平衡
                    optimize=False  # False 更快
                )
            elif format == "jpg":
                # JPG 格式 - 有损压缩
                if pil_image.mode in ("RGBA", "LA", "P"):
                    pil_image = pil_image.convert("RGB")
                pil_image.save(
                    file_path,
                    format="JPEG",
                    quality=webp_quality,  # 复用质量参数
                    optimize=False,
                    subsampling=0  # 0=最佳质量
                )
            
            results.append(file_path)
            filenames.append(filename)
            
            print(f"[FastImageSaver] 已保存: {file_path}")
        
        # 在节点 UI 中直接展示所有路径,便于确认保存位置
        ui_data = {
            "ui": {
                "text": "\n".join(results)
            },
            "result": (results[-1], filenames[-1])
        }

        return ui_data


class FastImageSaverBatch:
    """
    批量快速图像保存节点 - 返回所有文件路径
    """
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "format": (["webp", "png", "jpg"],),
                "webp_quality": ("INT", {
                    "default": 90,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "slider"
                }),
            },
            "optional": {
                "subfolder": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_paths",)
    FUNCTION = "save_images_batch"
    OUTPUT_NODE = True
    CATEGORY = "image/io"
    
    def save_images_batch(self, images, filename_prefix="ComfyUI", format="webp", webp_quality=90, subfolder=""):
        """
        批量保存图像并返回所有文件路径 (用换行符分隔)
        """
        # 确定输出目录
        if subfolder:
            output_path = os.path.join(self.output_dir, subfolder)
        else:
            output_path = self.output_dir
            
        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)
        
        results = []
        
        # 处理批量图像
        for batch_number, image in enumerate(images):
            # 转换张量到 numpy 数组
            i = 255. * image.cpu().numpy()
            img = np.clip(i, 0, 255).astype(np.uint8)
            
            # 转换为 PIL Image
            pil_image = Image.fromarray(img)
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            if len(images) > 1:
                filename = f"{filename_prefix}_{timestamp}_{batch_number:04d}.{format}"
            else:
                filename = f"{filename_prefix}_{timestamp}.{format}"
            
            file_path = os.path.join(output_path, filename)
            
            # 保存图像
            if format == "webp":
                pil_image.save(
                    file_path,
                    format="WEBP",
                    quality=webp_quality,
                    method=6,
                    lossless=False
                )
            elif format == "png":
                pil_image.save(
                    file_path,
                    format="PNG",
                    compress_level=6,
                    optimize=False
                )
            elif format == "jpg":
                if pil_image.mode in ("RGBA", "LA", "P"):
                    pil_image = pil_image.convert("RGB")
                pil_image.save(
                    file_path,
                    format="JPEG",
                    quality=webp_quality,
                    optimize=False,
                    subsampling=0
                )
            
            results.append(file_path)
            print(f"[FastImageSaverBatch] 已保存: {file_path}")
        
        # 返回所有路径(换行分隔)并在 UI 中展示每条路径
        all_paths = "\n".join(results)
        ui_data = {
            "ui": {
                "text": all_paths
            },
            "result": (all_paths,)
        }

        return ui_data


# 注册节点
NODE_CLASS_MAPPINGS = {
    "FastImageSaver": FastImageSaver,
    "FastImageSaverBatch": FastImageSaverBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FastImageSaver": "Fast Image Saver (快速保存图像)",
    "FastImageSaverBatch": "Fast Image Saver Batch (批量快速保存)",
}
