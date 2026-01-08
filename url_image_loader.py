"""
URLLoader ComfyUI 节点：从 URL 加载图像

支持双 URL 容错格式：主URL|||备用URL
- 只传一个 URL：正常加载
- 传两个 URL（用 ||| 分隔）：主 URL 失败时自动尝试备用 URL
"""

import io
import time
import torch
import numpy as np
from PIL import Image
import requests
from typing import Tuple, Optional, List

# 双 URL 分隔符（用 ||| 避免与 URL 中的 | 冲突）
URL_SEPARATOR = "|||"


class URLLoaderLoadImageFromURL:
    """从 URL 加载图像，支持双 URL 容错"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
            },
            "optional": {
                "timeout": ("INT", {
                    "default": 30,
                    "min": 5,
                    "max": 120,
                    "step": 1,
                    "display": "number",
                }),
                "retry_count": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 5,
                    "step": 1,
                    "display": "number",
                }),
                "retry_delay": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 5.0,
                    "step": 0.5,
                    "display": "number",
                }),
                "stop_on_error": (["enabled", "disabled"], {
                    "default": "enabled",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOOLEAN")
    RETURN_NAMES = ("image", "mask", "success")
    FUNCTION = "load_image"
    CATEGORY = "URLLoader/Image"

    def _fetch_image(self, url: str, timeout: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[str]]:
        """尝试从单个 URL 获取图像，返回 (image, mask, error)"""
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            
            img = Image.open(io.BytesIO(response.content))
            
            if img.mode == 'RGBA':
                alpha = np.array(img.split()[-1]).astype(np.float32) / 255.0
                img = img.convert('RGB')
                mask = torch.from_numpy(alpha)
            else:
                img = img.convert('RGB')
                mask = torch.ones((img.size[1], img.size[0]), dtype=torch.float32)
            
            image_np = np.array(img, dtype=np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np).unsqueeze(0)
            
            return image_tensor, mask.unsqueeze(0), None
            
        except requests.exceptions.Timeout:
            return None, None, f"超时({timeout}s)"
        except requests.exceptions.HTTPError as e:
            return None, None, f"HTTP {e.response.status_code}"
        except Exception as e:
            return None, None, f"{type(e).__name__}"

    def load_image(self, url: str, timeout: int = 30, retry_count: int = 2, retry_delay: float = 1.0, stop_on_error: str = "enabled") -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        从 URL 加载图像
        
        URL 格式：
        - 单个 URL：直接加载
        - 双 URL：主URL|||备用URL，交替尝试直到成功
        
        重试策略（双 URL 时）：主→备→主→备... 交替尝试，总共 retry_count 轮
        """
        if not url or not url.strip():
            raise ValueError("URL 不能为空")
        
        url = url.strip()
        
        # 解析 URL（支持 ||| 分隔的双 URL）
        if URL_SEPARATOR in url:
            urls = [u.strip() for u in url.split(URL_SEPARATOR, 1) if u.strip()]
        else:
            urls = [url]
        
        if not urls:
            raise ValueError("URL 不能为空")
        
        errors = {}  # url_label -> last_error
        has_two_urls = len(urls) >= 2
        
        # 交替尝试：主→备→主→备...，共 retry_count 轮
        for attempt in range(retry_count):
            for idx, current_url in enumerate(urls[:2]):
                url_label = "主URL" if idx == 0 else "备用URL"
                
                if attempt == 0 and idx == 0:
                    if has_two_urls:
                        print(f"[URLLoader] 双URL模式，交替尝试")
                    print(f"[URLLoader] 加载 {url_label}: {current_url}")
                else:
                    print(f"[URLLoader] 尝试 {url_label} (第{attempt + 1}轮): {current_url}")
                
                image, mask, error = self._fetch_image(current_url, timeout)
                
                if image is not None:
                    print(f"[URLLoader] ✅ OK ({url_label})")
                    return (image, mask, True)
                
                errors[url_label] = error
                print(f"[URLLoader] ❌ {url_label} 失败: {error}")
                
                # 尝试之间短暂延迟（除了最后一次）
                is_last_attempt = (attempt == retry_count - 1) and (idx == len(urls[:2]) - 1)
                if not is_last_attempt:
                    time.sleep(retry_delay)
        
        # 全部失败
        error_parts = [f"{k}: {v}" for k, v in errors.items()]
        error_msg = f"加载失败: {'; '.join(error_parts)}"
        print(f"[URLLoader] 🛑 {error_msg}")
        
        if stop_on_error == "enabled":
            raise RuntimeError(error_msg)
        else:
            black_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return (black_image, empty_mask, False)


NODE_CLASS_MAPPINGS = {
    "URLLoaderLoadImageFromURL": URLLoaderLoadImageFromURL,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "URLLoaderLoadImageFromURL": "Load Image from URL (URLLoader)",
}
