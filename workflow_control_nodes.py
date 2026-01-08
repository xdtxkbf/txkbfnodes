"""
WorkflowCtrl 工作流控制节点
"""

import sys
from typing import Tuple, Any

class WorkflowCtrlWorkflowControl:
    """根据条件控制工作流执行（失败时停止）"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "condition": ("BOOLEAN", {
                    "default": True,
                }),
                "error_message": ("STRING", {
                    "default": "工作流条件未满足",
                    "multiline": True,
                }),
            },
            "optional": {
                "pass_through": ("*", {}),  # 任意类型，用于传递数据
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output",)
    FUNCTION = "check_condition"
    CATEGORY = "WorkflowCtrl/Control"
    OUTPUT_NODE = False

    def check_condition(self, condition: bool, error_message: str, pass_through: Any = None) -> Tuple[Any]:
        """
        检查条件，如果为 False 则停止工作流
        
        Args:
            condition: 条件（True 继续，False 停止）
            error_message: 条件为 False 时的错误消息
            pass_through: 可选的透传数据
            
        Returns:
            (output,): 透传的数据
        """
        if not condition:
            error_msg = f"[WorkflowCtrl] 🛑 工作流条件检查失败: {error_message}"
            print(error_msg)
            print("[WorkflowCtrl] 工作流将被终止")
            # 抛出异常来停止整个工作流
            raise RuntimeError(error_msg)
        
        print(f"[WorkflowCtrl] ✅ 工作流条件检查通过")
        return (pass_through,)


class WorkflowCtrlImageValidator:
    """验证图像加载状态，失败时停止工作流"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "success": ("BOOLEAN",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("validated_image",)
    FUNCTION = "validate"
    CATEGORY = "WorkflowCtrl/Image"
    OUTPUT_NODE = False

    def validate(self, image, success: bool) -> Tuple:
        """
        验证图像是否成功加载
        
        Args:
            image: 图像张量
            success: 加载成功标志
            
        Returns:
            (validated_image,): 验证通过的图像
        """
        if not success:
            error_msg = "[WorkflowCtrl] 🛑 图像加载失败，无法继续执行工作流"
            print(error_msg)
            print("[WorkflowCtrl] 请检查：")
            print("[WorkflowCtrl]   1. URL 是否有效")
            print("[WorkflowCtrl]   2. 网络连接是否正常")
            print("[WorkflowCtrl]   3. Telegram Bot Token 是否有效")
            print("[WorkflowCtrl]   4. ComfyUI 服务器是否能访问外网")
            raise RuntimeError(error_msg)
        
        print("[WorkflowCtrl] ✅ 图像验证通过，继续执行工作流")
        return (image,)


class WorkflowCtrlConditionalExecution:
    """条件执行节点：根据 success 标志选择不同的分支"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "condition": ("BOOLEAN",),
                "if_true": ("*",),
                "if_false": ("*",),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output",)
    FUNCTION = "execute"
    CATEGORY = "WorkflowCtrl/Control"

    def execute(self, condition: bool, if_true: Any, if_false: Any) -> Tuple[Any]:
        """
        根据条件选择执行分支
        
        Args:
            condition: 条件
            if_true: 条件为 True 时返回的值
            if_false: 条件为 False 时返回的值
            
        Returns:
            (output,): 根据条件选择的输出
        """
        if condition:
            print("[WorkflowCtrl] ✅ 条件为 True，使用 if_true 分支")
            return (if_true,)
        else:
            print("[WorkflowCtrl] ⚠️ 条件为 False，使用 if_false 分支")
            return (if_false,)


# ComfyUI 节点注册
NODE_CLASS_MAPPINGS = {
    "WorkflowCtrlWorkflowControl": WorkflowCtrlWorkflowControl,
    "WorkflowCtrlImageValidator": WorkflowCtrlImageValidator,
    "WorkflowCtrlConditionalExecution": WorkflowCtrlConditionalExecution,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WorkflowCtrlWorkflowControl": "Workflow Control (WorkflowCtrl)",
    "WorkflowCtrlImageValidator": "Image Validator (WorkflowCtrl)",
    "WorkflowCtrlConditionalExecution": "Conditional Execution (WorkflowCtrl)",
}
