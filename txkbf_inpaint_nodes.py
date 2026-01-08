"""
ComfyUI-txkbfnodes - Inpaint Nodes
Enhanced inpaint crop, stitch, and control nodes
"""
import comfy.utils
import math
import nodes
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter, grey_dilation, grey_erosion, binary_closing, binary_fill_holes


# ============================================================================
# Utility Functions
# ============================================================================

def rescale_i(samples, width, height, algorithm: str):
    samples = samples.movedim(-1, 1)
    algorithm = getattr(Image, algorithm.upper())
    samples_pil: Image.Image = F.to_pil_image(samples[0].cpu()).resize((width, height), algorithm)
    samples = F.to_tensor(samples_pil).unsqueeze(0)
    samples = samples.movedim(1, -1)
    return samples


def rescale_m(samples, width, height, algorithm: str):
    samples = samples.unsqueeze(1)
    algorithm = getattr(Image, algorithm.upper())
    samples_pil: Image.Image = F.to_pil_image(samples[0].cpu()).resize((width, height), algorithm)
    samples = F.to_tensor(samples_pil).unsqueeze(0)
    samples = samples.squeeze(1)
    return samples


def expand_m(mask, pixels):
    """Expand mask by exact number of pixels in all directions"""
    if pixels <= 0:
        return mask.clone()
    mask_np = mask.squeeze(0).cpu().numpy()
    kernel_size = pixels * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated_mask = grey_dilation(mask_np, footprint=kernel)
    dilated_mask = dilated_mask.astype(np.float32)
    dilated_mask = torch.from_numpy(dilated_mask)
    dilated_mask = torch.clamp(dilated_mask, 0.0, 1.0)
    return dilated_mask.unsqueeze(0)


def erode_m(mask, pixels):
    """Erode mask by exact number of pixels in all directions (shrink mask)"""
    if pixels <= 0:
        return mask.clone()
    mask_np = mask.squeeze(0).cpu().numpy()
    kernel_size = pixels * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    eroded_mask = grey_erosion(mask_np, footprint=kernel)
    eroded_mask = eroded_mask.astype(np.float32)
    eroded_mask = torch.from_numpy(eroded_mask)
    eroded_mask = torch.clamp(eroded_mask, 0.0, 1.0)
    return eroded_mask.unsqueeze(0)


def invert_m(samples):
    inverted_mask = samples.clone()
    inverted_mask = 1.0 - inverted_mask
    return inverted_mask


def blur_m(samples, pixels):
    mask = samples.squeeze(0)
    sigma = pixels / 4 
    mask_np = mask.cpu().numpy()
    blurred_mask = gaussian_filter(mask_np, sigma=sigma)
    blurred_mask = torch.from_numpy(blurred_mask).float()
    blurred_mask = torch.clamp(blurred_mask, 0.0, 1.0)
    return blurred_mask.unsqueeze(0)


def fillholes_iterative_hipass_fill_m(samples):
    thresholds = [1, 0.99, 0.97, 0.95, 0.93, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    mask_np = samples.squeeze(0).cpu().numpy()
    for threshold in thresholds:
        thresholded_mask = mask_np >= threshold
        closed_mask = binary_closing(thresholded_mask, structure=np.ones((3, 3)), border_value=1)
        filled_mask = binary_fill_holes(closed_mask)
        mask_np = np.maximum(mask_np, np.where(filled_mask != 0, threshold, 0))
    final_mask = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0)
    return final_mask


def hipassfilter_m(samples, threshold):
    filtered_mask = samples.clone()
    filtered_mask[filtered_mask < threshold] = 0
    return filtered_mask


def preresize_imm(image, mask, optional_context_mask, downscale_algorithm, upscale_algorithm, preresize_mode, preresize_min_width, preresize_min_height, preresize_max_width, preresize_max_height):
    current_width, current_height = image.shape[2], image.shape[1]
    
    if preresize_mode == "ensure minimum resolution":
        if current_width >= preresize_min_width and current_height >= preresize_min_height:
            return image, mask, optional_context_mask
        scale_factor_min_width = preresize_min_width / current_width
        scale_factor_min_height = preresize_min_height / current_height
        scale_factor = max(scale_factor_min_width, scale_factor_min_height)
        target_width = int(current_width * scale_factor)
        target_height = int(current_height * scale_factor)
        image = rescale_i(image, target_width, target_height, upscale_algorithm)
        mask = rescale_m(mask, target_width, target_height, 'bilinear')
        optional_context_mask = rescale_m(optional_context_mask, target_width, target_height, 'bilinear')

    elif preresize_mode == "ensure minimum and maximum resolution":
        if preresize_min_width <= current_width <= preresize_max_width and preresize_min_height <= current_height <= preresize_max_height:
            return image, mask, optional_context_mask
        scale_factor_min_width = preresize_min_width / current_width
        scale_factor_min_height = preresize_min_height / current_height
        scale_factor_min = max(scale_factor_min_width, scale_factor_min_height)
        scale_factor_max_width = preresize_max_width / current_width
        scale_factor_max_height = preresize_max_height / current_height
        scale_factor_max = min(scale_factor_max_width, scale_factor_max_height)
        if scale_factor_min > 1 and scale_factor_max < 1:
            assert False, "Cannot meet both minimum and maximum resolution requirements"
        if scale_factor_min > 1:
            scale_factor = scale_factor_min
            rescale_algorithm = upscale_algorithm
        else:
            scale_factor = scale_factor_max
            rescale_algorithm = downscale_algorithm
        target_width = int(current_width * scale_factor)
        target_height = int(current_height * scale_factor)
        image = rescale_i(image, target_width, target_height, rescale_algorithm)
        mask = rescale_m(mask, target_width, target_height, 'nearest')
        optional_context_mask = rescale_m(optional_context_mask, target_width, target_height, 'nearest')

    elif preresize_mode == "ensure maximum resolution":
        if current_width <= preresize_max_width and current_height <= preresize_max_height:
            return image, mask, optional_context_mask
        scale_factor_max_width = preresize_max_width / current_width
        scale_factor_max_height = preresize_max_height / current_height
        scale_factor_max = min(scale_factor_max_width, scale_factor_max_height)
        target_width = int(current_width * scale_factor_max)
        target_height = int(current_height * scale_factor_max)
        image = rescale_i(image, target_width, target_height, downscale_algorithm)
        mask = rescale_m(mask, target_width, target_height, 'nearest')
        optional_context_mask = rescale_m(optional_context_mask, target_width, target_height, 'nearest')

    return image, mask, optional_context_mask


def extend_imm(image, mask, optional_context_mask, extend_up_factor, extend_down_factor, extend_left_factor, extend_right_factor):
    B, H, W, C = image.shape
    new_H = int(H * (1.0 + extend_up_factor - 1.0 + extend_down_factor - 1.0))
    new_W = int(W * (1.0 + extend_left_factor - 1.0 + extend_right_factor - 1.0))
    assert new_H >= 0, f"Error: Trying to crop too much, height ({new_H}) must be >= 0"
    assert new_W >= 0, f"Error: Trying to crop too much, width ({new_W}) must be >= 0"
    
    extended_image = torch.zeros((B, new_H, new_W, C), device=image.device, dtype=image.dtype)
    extended_mask = torch.ones((mask.shape[0], new_H, new_W), device=mask.device, dtype=mask.dtype)
    extended_optional_context_mask = torch.zeros((optional_context_mask.shape[0], new_H, new_W), device=optional_context_mask.device, dtype=optional_context_mask.dtype)
    
    start_y = int(H * (extend_up_factor - 1.0))
    start_x = int(W * (extend_left_factor - 1.0))
    
    extended_image[:, start_y:start_y+H, start_x:start_x+W, :] = image
    extended_mask[:, start_y:start_y+H, start_x:start_x+W] = mask
    extended_optional_context_mask[:, start_y:start_y+H, start_x:start_x+W] = optional_context_mask
    
    return extended_image, extended_mask, extended_optional_context_mask


def findcontextarea_m(mask):
    mask_squeezed = mask[0]
    non_zero_indices = torch.nonzero(mask_squeezed)
    H, W = mask_squeezed.shape
    if non_zero_indices.numel() == 0:
        x, y = -1, -1
        w, h = -1, -1
    else:
        y = torch.min(non_zero_indices[:, 0]).item()
        x = torch.min(non_zero_indices[:, 1]).item()
        y_max = torch.max(non_zero_indices[:, 0]).item()
        x_max = torch.max(non_zero_indices[:, 1]).item()
        w = x_max - x + 1
        h = y_max - y + 1
    context = mask[:, y:y+h, x:x+w]
    return context, x, y, w, h


def growcontextarea_m(context, mask, x, y, w, h, extend_factor):
    img_h, img_w = mask.shape[1], mask.shape[2]
    grow_left = int(round(w * (extend_factor-1.0) / 2.0))
    grow_right = int(round(w * (extend_factor-1.0) / 2.0))
    grow_up = int(round(h * (extend_factor-1.0) / 2.0))
    grow_down = int(round(h * (extend_factor-1.0) / 2.0))
    new_x = max(0, x - grow_left)
    new_y = max(0, y - grow_up)
    new_x2 = min(img_w, x + w + grow_right)
    new_y2 = min(img_h, y + h + grow_down)
    new_w = new_x2 - new_x
    new_h = new_y2 - new_y
    new_context = mask[:, new_y:new_y+new_h, new_x:new_x+new_w]
    if new_h < 0 or new_w < 0:
        new_x = 0
        new_y = 0
        new_w = mask.shape[2]
        new_h = mask.shape[1]
    return new_context, new_x, new_y, new_w, new_h


def combinecontextmask_m(context, mask, x, y, w, h, optional_context_mask):
    _, x_opt, y_opt, w_opt, h_opt = findcontextarea_m(optional_context_mask)
    if x == -1:
        x, y, w, h = x_opt, y_opt, w_opt, h_opt
    if x_opt == -1:
        x_opt, y_opt, w_opt, h_opt = x, y, w, h
    if x == -1:
        return torch.zeros(1, 0, 0, device=mask.device), -1, -1, -1, -1
    new_x = min(x, x_opt)
    new_y = min(y, y_opt)
    new_x_max = max(x + w, x_opt + w_opt)
    new_y_max = max(y + h, y_opt + h_opt)
    new_w = new_x_max - new_x
    new_h = new_y_max - new_y
    combined_context = mask[:, new_y:new_y+new_h, new_x:new_x+new_w]
    return combined_context, new_x, new_y, new_w, new_h


def pad_to_multiple(value, multiple):
    return int(math.ceil(value / multiple) * multiple)


def crop_magic_im(image, mask, x, y, w, h, target_w, target_h, padding, downscale_algorithm, upscale_algorithm):
    image = image.clone()
    mask = mask.clone()
    
    if target_w <= 0 or target_h <= 0 or w == 0 or h == 0:
        return image, 0, 0, image.shape[2], image.shape[1], image, mask, 0, 0, image.shape[2], image.shape[1]

    if padding != 0:
        target_w = pad_to_multiple(target_w, padding)
        target_h = pad_to_multiple(target_h, padding)

    target_aspect_ratio = target_w / target_h
    B, image_h, image_w, C = image.shape
    context_aspect_ratio = w / h
    
    if context_aspect_ratio < target_aspect_ratio:
        new_w = int(h * target_aspect_ratio)
        new_h = h
        new_x = x - (new_w - w) // 2
        new_y = y
        if new_x < 0:
            shift = -new_x
            if new_x + new_w + shift <= image_w:
                new_x += shift
            else:
                overflow = (new_w - image_w) // 2
                new_x = -overflow
        elif new_x + new_w > image_w:
            overflow = new_x + new_w - image_w
            if new_x - overflow >= 0:
                new_x -= overflow
            else:
                overflow = (new_w - image_w) // 2
                new_x = -overflow
    else:
        new_w = w
        new_h = int(w / target_aspect_ratio)
        new_x = x
        new_y = y - (new_h - h) // 2
        if new_y < 0:
            shift = -new_y
            if new_y + new_h + shift <= image_h:
                new_y += shift
            else:
                overflow = (new_h - image_h) // 2
                new_y = -overflow
        elif new_y + new_h > image_h:
            overflow = new_y + new_h - image_h
            if new_y - overflow >= 0:
                new_y -= overflow
            else:
                overflow = (new_h - image_h) // 2
                new_y = -overflow

    up_padding, down_padding, left_padding, right_padding = 0, 0, 0, 0
    expanded_image_w = image_w
    expanded_image_h = image_h

    if new_x < 0:
        left_padding = -new_x
        expanded_image_w += left_padding
    if new_x + new_w > image_w:
        right_padding = (new_x + new_w - image_w)
        expanded_image_w += right_padding
    if new_y < 0:
        up_padding = -new_y
        expanded_image_h += up_padding 
    if new_y + new_h > image_h:
        down_padding = (new_y + new_h - image_h)
        expanded_image_h += down_padding

    expanded_image = torch.zeros((image.shape[0], expanded_image_h, expanded_image_w, image.shape[3]), device=image.device)
    expanded_mask = torch.ones((mask.shape[0], expanded_image_h, expanded_image_w), device=mask.device)

    image = image.permute(0, 3, 1, 2)
    expanded_image = expanded_image.permute(0, 3, 1, 2)
    expanded_image[:, :, up_padding:up_padding + image_h, left_padding:left_padding + image_w] = image

    if up_padding > 0:
        expanded_image[:, :, :up_padding, left_padding:left_padding + image_w] = image[:, :, 0:1, :].repeat(1, 1, up_padding, 1)
    if down_padding > 0:
        expanded_image[:, :, -down_padding:, left_padding:left_padding + image_w] = image[:, :, -1:, :].repeat(1, 1, down_padding, 1)
    if left_padding > 0:
        expanded_image[:, :, up_padding:up_padding + image_h, :left_padding] = expanded_image[:, :, up_padding:up_padding + image_h, left_padding:left_padding+1].repeat(1, 1, 1, left_padding)
    if right_padding > 0:
        expanded_image[:, :, up_padding:up_padding + image_h, -right_padding:] = expanded_image[:, :, up_padding:up_padding + image_h, -right_padding-1:-right_padding].repeat(1, 1, 1, right_padding)

    expanded_image = expanded_image.permute(0, 2, 3, 1)
    image = image.permute(0, 2, 3, 1)
    expanded_mask[:, up_padding:up_padding + image_h, left_padding:left_padding + image_w] = mask

    cto_x = left_padding
    cto_y = up_padding
    cto_w = image_w
    cto_h = image_h

    canvas_image = expanded_image
    canvas_mask = expanded_mask

    ctc_x = new_x+left_padding
    ctc_y = new_y+up_padding
    ctc_w = new_w
    ctc_h = new_h

    cropped_image = canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]
    cropped_mask = canvas_mask[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]

    cropped_image_raw = cropped_image.clone()
    cropped_mask_raw = cropped_mask.clone()

    if target_w > ctc_w or target_h > ctc_h:
        cropped_image = rescale_i(cropped_image, target_w, target_h, upscale_algorithm)
        cropped_mask = rescale_m(cropped_mask, target_w, target_h, upscale_algorithm)
    else:
        cropped_image = rescale_i(cropped_image, target_w, target_h, downscale_algorithm)
        cropped_mask = rescale_m(cropped_mask, target_w, target_h, downscale_algorithm)

    return canvas_image, cto_x, cto_y, cto_w, cto_h, cropped_image, cropped_mask, ctc_x, ctc_y, ctc_w, ctc_h, cropped_image_raw, cropped_mask_raw


def stitch_magic_im(canvas_image, inpainted_image, mask, ctc_x, ctc_y, ctc_w, ctc_h, cto_x, cto_y, cto_w, cto_h, downscale_algorithm, upscale_algorithm):
    canvas_image = canvas_image.clone()
    inpainted_image = inpainted_image.clone()
    mask = mask.clone()

    _, h, w, _ = inpainted_image.shape
    if ctc_w > w or ctc_h > h:
        resized_image = rescale_i(inpainted_image, ctc_w, ctc_h, upscale_algorithm)
        resized_mask = rescale_m(mask, ctc_w, ctc_h, upscale_algorithm)
    else:
        resized_image = rescale_i(inpainted_image, ctc_w, ctc_h, downscale_algorithm)
        resized_mask = rescale_m(mask, ctc_w, ctc_h, downscale_algorithm)

    resized_mask = resized_mask.clamp(0, 1).unsqueeze(-1)
    canvas_crop = canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]
    blended = resized_mask * resized_image + (1.0 - resized_mask) * canvas_crop
    canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w] = blended
    output_image = canvas_image[:, cto_y:cto_y + cto_h, cto_x:cto_x + cto_w]
    return output_image


# ============================================================================
# Node Classes
# ============================================================================

class InpaintCropEnhanced:
    """Enhanced inpaint crop with pixel-based mask expansion and auto aspect ratio"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "downscale_algorithm": (["nearest", "bilinear", "bicubic", "lanczos", "box", "hamming"], {"default": "bilinear"}),
                "upscale_algorithm": (["nearest", "bilinear", "bicubic", "lanczos", "box", "hamming"], {"default": "bicubic"}),
                "preresize": ("BOOLEAN", {"default": False}),
                "preresize_mode": (["ensure minimum resolution", "ensure maximum resolution", "ensure minimum and maximum resolution"], {"default": "ensure minimum resolution"}),
                "preresize_min_width": ("INT", {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "preresize_min_height": ("INT", {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "preresize_max_width": ("INT", {"default": nodes.MAX_RESOLUTION, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "preresize_max_height": ("INT", {"default": nodes.MAX_RESOLUTION, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "mask_fill_holes": ("BOOLEAN", {"default": True}),
                "mask_expand_pixels": ("INT", {"default": 10, "min": 0, "max": 500, "step": 1}),
                "mask_invert": ("BOOLEAN", {"default": False}),
                "mask_blend_pixels": ("INT", {"default": 32, "min": 0, "max": 64, "step": 1}),
                "mask_hipass_filter": ("FLOAT", {"default": 0.1, "min": 0, "max": 1, "step": 0.01}),
                "extend_for_outpainting": ("BOOLEAN", {"default": False}),
                "extend_up_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "extend_down_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "extend_left_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "extend_right_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "context_from_mask_extend_factor": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 100.0, "step": 0.01}),
                "auto_aspect_ratio": ("BOOLEAN", {"default": True}),
                "output_resize_to_target_size": ("BOOLEAN", {"default": True}),
                "output_kilo_pixels": ("INT", {"default": 1000, "min": 64, "max": 16384, "step": 1}),
                "output_target_width": ("INT", {"default": 512, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "output_target_height": ("INT", {"default": 512, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "output_padding": (["0", "8", "16", "32", "64", "128", "256", "512"], {"default": "32"}),
           },
           "optional": {
                "mask": ("MASK",),
                "optional_context_mask": ("MASK",),
           }
        }

    FUNCTION = "inpaint_crop"
    CATEGORY = "inpaint"
    RETURN_TYPES = ("STITCHER", "IMAGE", "MASK", "MASK", "IMAGE", "MASK")
    RETURN_NAMES = ("stitcher", "cropped_image", "cropped_mask", "cropped_mask_expanded", "cropped_image_raw", "cropped_mask_raw")

    def inpaint_crop(self, image, downscale_algorithm, upscale_algorithm, preresize, preresize_mode, preresize_min_width, preresize_min_height, preresize_max_width, preresize_max_height, extend_for_outpainting, extend_up_factor, extend_down_factor, extend_left_factor, extend_right_factor, mask_hipass_filter, mask_fill_holes, mask_expand_pixels, mask_invert, mask_blend_pixels, context_from_mask_extend_factor, auto_aspect_ratio, output_resize_to_target_size, output_kilo_pixels, output_target_width, output_target_height, output_padding, mask=None, optional_context_mask=None):
        image = image.clone()
        if mask is not None:
            mask = mask.clone()
        if optional_context_mask is not None:
            optional_context_mask = optional_context_mask.clone()

        output_padding = int(output_padding)

        if preresize and preresize_mode == "ensure minimum and maximum resolution":
            assert preresize_max_width >= preresize_min_width
            assert preresize_max_height >= preresize_min_height

        if image.shape[0] > 1:
            assert output_resize_to_target_size

        if mask is not None and (image.shape[0] == 1 or mask.shape[0] == 1 or mask.shape[0] == image.shape[0]):
            if mask.shape[1] != image.shape[1] or mask.shape[2] != image.shape[2]:
                if torch.count_nonzero(mask) == 0:
                    mask = torch.zeros((mask.shape[0], image.shape[1], image.shape[2]), device=image.device, dtype=image.dtype)

        if optional_context_mask is not None and (image.shape[0] == 1 or optional_context_mask.shape[0] == 1 or optional_context_mask.shape[0] == image.shape[0]):
            if optional_context_mask.shape[1] != image.shape[1] or optional_context_mask.shape[2] != image.shape[2]:
                if torch.count_nonzero(optional_context_mask) == 0:
                    optional_context_mask = torch.zeros((optional_context_mask.shape[0], image.shape[1], image.shape[2]), device=image.device, dtype=image.dtype)

        if mask is None:
            mask = torch.zeros_like(image[:, :, :, 0])
    
        if mask.shape[0] > 1 and image.shape[0] == 1:
            image = image.expand(mask.shape[0], -1, -1, -1).clone()

        if image.shape[0] > 1 and mask.shape[0] == 1:
            mask = mask.expand(image.shape[0], -1, -1).clone()

        if optional_context_mask is None:
            optional_context_mask = torch.zeros_like(image[:, :, :, 0])

        if image.shape[0] > 1 and optional_context_mask.shape[0] == 1:
            optional_context_mask = optional_context_mask.expand(image.shape[0], -1, -1).clone()

        original_image_aspect_ratio = image.shape[2] / image.shape[1]

        result_stitcher = {
            'downscale_algorithm': downscale_algorithm,
            'upscale_algorithm': upscale_algorithm,
            'blend_pixels': mask_blend_pixels,
            'canvas_to_orig_x': [],
            'canvas_to_orig_y': [],
            'canvas_to_orig_w': [],
            'canvas_to_orig_h': [],
            'canvas_image': [],
            'cropped_to_canvas_x': [],
            'cropped_to_canvas_y': [],
            'cropped_to_canvas_w': [],
            'cropped_to_canvas_h': [],
            'cropped_mask_for_blend': [],
        }
        
        result_image = []
        result_mask = []
        result_mask_expanded = []
        result_image_raw = []
        result_mask_raw = []

        batch_size = image.shape[0]
        for b in range(batch_size):
            one_image = image[b:b+1]
            one_mask = mask[b:b+1]
            one_optional_context_mask = optional_context_mask[b:b+1]

            outputs = self.inpaint_crop_single_image(
                one_image, downscale_algorithm, upscale_algorithm, preresize, preresize_mode,
                preresize_min_width, preresize_min_height, preresize_max_width, preresize_max_height,
                extend_for_outpainting, extend_up_factor, extend_down_factor, extend_left_factor, extend_right_factor,
                mask_hipass_filter, mask_fill_holes, mask_expand_pixels, mask_invert, mask_blend_pixels,
                context_from_mask_extend_factor, auto_aspect_ratio, original_image_aspect_ratio, 
                output_resize_to_target_size, output_kilo_pixels, output_target_width, output_target_height,
                output_padding, one_mask, one_optional_context_mask)

            stitcher, cropped_image, cropped_mask, cropped_mask_expanded, cropped_image_raw, cropped_mask_raw = outputs
            for key in ['canvas_to_orig_x', 'canvas_to_orig_y', 'canvas_to_orig_w', 'canvas_to_orig_h', 'canvas_image', 'cropped_to_canvas_x', 'cropped_to_canvas_y', 'cropped_to_canvas_w', 'cropped_to_canvas_h', 'cropped_mask_for_blend']:
                result_stitcher[key].append(stitcher[key])

            result_image.append(cropped_image.squeeze(0))
            result_mask.append(cropped_mask.squeeze(0))
            result_mask_expanded.append(cropped_mask_expanded.squeeze(0))
            result_image_raw.append(cropped_image_raw.squeeze(0))
            result_mask_raw.append(cropped_mask_raw.squeeze(0))

        result_image = torch.stack(result_image, dim=0)
        result_mask = torch.stack(result_mask, dim=0)
        result_mask_expanded = torch.stack(result_mask_expanded, dim=0)
        result_image_raw = torch.stack(result_image_raw, dim=0)
        result_mask_raw = torch.stack(result_mask_raw, dim=0)

        return result_stitcher, result_image, result_mask, result_mask_expanded, result_image_raw, result_mask_raw

    def inpaint_crop_single_image(self, image, downscale_algorithm, upscale_algorithm, preresize, preresize_mode, preresize_min_width, preresize_min_height, preresize_max_width, preresize_max_height, extend_for_outpainting, extend_up_factor, extend_down_factor, extend_left_factor, extend_right_factor, mask_hipass_filter, mask_fill_holes, mask_expand_pixels, mask_invert, mask_blend_pixels, context_from_mask_extend_factor, auto_aspect_ratio, original_image_aspect_ratio, output_resize_to_target_size, output_kilo_pixels, output_target_width, output_target_height, output_padding, mask, optional_context_mask):
        if preresize:
            image, mask, optional_context_mask = preresize_imm(image, mask, optional_context_mask, downscale_algorithm, upscale_algorithm, preresize_mode, preresize_min_width, preresize_min_height, preresize_max_width, preresize_max_height)
       
        if mask_fill_holes:
           mask = fillholes_iterative_hipass_fill_m(mask)

        if mask_invert:
            mask = invert_m(mask)

        if mask_hipass_filter >= 0.01:
            mask = hipassfilter_m(mask, mask_hipass_filter)
            optional_context_mask = hipassfilter_m(optional_context_mask, mask_hipass_filter)

        if extend_for_outpainting:
            image, mask, optional_context_mask = extend_imm(image, mask, optional_context_mask, extend_up_factor, extend_down_factor, extend_left_factor, extend_right_factor)

        context, x, y, w, h = findcontextarea_m(mask)
        if x == -1 or w == -1 or h == -1 or y == -1:
            x, y, w, h = 0, 0, image.shape[2], image.shape[1]
        else:
            if context_from_mask_extend_factor >= 1.01:
                context, x, y, w, h = growcontextarea_m(context, mask, x, y, w, h, context_from_mask_extend_factor)
                if x == -1 or w == -1 or h == -1 or y == -1:
                    x, y, w, h = 0, 0, image.shape[2], image.shape[1]

            context, x, y, w, h = combinecontextmask_m(context, mask, x, y, w, h, optional_context_mask)
            if x == -1 or w == -1 or h == -1 or y == -1:
                x, y, w, h = 0, 0, image.shape[2], image.shape[1]

        padding = int(output_padding)
        
        if auto_aspect_ratio:
            if output_resize_to_target_size:
                total_pixels = output_kilo_pixels * 1000
                final_target_w = int(math.sqrt(total_pixels * original_image_aspect_ratio))
                final_target_h = int(math.sqrt(total_pixels / original_image_aspect_ratio))
                if padding > 0:
                    final_target_w = int(round(final_target_w / padding) * padding)
                    final_target_h = int(round(final_target_h / padding) * padding)
                    final_target_w = max(padding, final_target_w)
                    final_target_h = max(padding, final_target_h)
            else:
                final_target_w = w
                final_target_h = h
        else:
            if output_resize_to_target_size:
                final_target_w = output_target_width
                final_target_h = output_target_height
                if padding > 0:
                    final_target_w = int(round(final_target_w / padding) * padding)
                    final_target_h = int(round(final_target_h / padding) * padding)
                    final_target_w = max(padding, final_target_w)
                    final_target_h = max(padding, final_target_h)
            else:
                final_target_w = w
                final_target_h = h

        canvas_image, cto_x, cto_y, cto_w, cto_h, cropped_image, cropped_mask, ctc_x, ctc_y, ctc_w, ctc_h, cropped_image_raw, cropped_mask_raw = crop_magic_im(
            image, mask, x, y, w, h, final_target_w, final_target_h, output_padding, downscale_algorithm, upscale_algorithm)

        cropped_mask_original = cropped_mask.clone()
        
        if mask_expand_pixels > 0:
            cropped_mask_expanded = expand_m(cropped_mask, mask_expand_pixels)
        else:
            cropped_mask_expanded = cropped_mask.clone()
        
        if mask_blend_pixels > 0:
            outer_boundary = cropped_mask_expanded.clone()
            blurred = blur_m(cropped_mask_expanded, mask_blend_pixels)
            cropped_mask_blend = torch.minimum(blurred, outer_boundary)
        else:
            cropped_mask_blend = cropped_mask_expanded.clone()

        stitcher = {
            'canvas_to_orig_x': cto_x,
            'canvas_to_orig_y': cto_y,
            'canvas_to_orig_w': cto_w,
            'canvas_to_orig_h': cto_h,
            'canvas_image': canvas_image,
            'cropped_to_canvas_x': ctc_x,
            'cropped_to_canvas_y': ctc_y,
            'cropped_to_canvas_w': ctc_w,
            'cropped_to_canvas_h': ctc_h,
            'cropped_mask_for_blend': cropped_mask_blend,
        }

        return stitcher, cropped_image, cropped_mask_original, cropped_mask_expanded, cropped_image_raw, cropped_mask_raw


class InpaintStitchImproved:
    """Stitch inpainted image back to original"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitcher": ("STITCHER",),
                "inpainted_image": ("IMAGE",),
            }
        }

    CATEGORY = "inpaint"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "inpaint_stitch"

    def inpaint_stitch(self, stitcher, inpainted_image):
        inpainted_image = inpainted_image.clone()
        results = []

        batch_size = inpainted_image.shape[0]
        assert len(stitcher['cropped_to_canvas_x']) == batch_size or len(stitcher['cropped_to_canvas_x']) == 1
        override = False
        if len(stitcher['cropped_to_canvas_x']) != batch_size and len(stitcher['cropped_to_canvas_x']) == 1:
            override = True
            
        for b in range(batch_size):
            one_image = inpainted_image[b]
            one_stitcher = {}
            for key in ['downscale_algorithm', 'upscale_algorithm', 'blend_pixels']:
                one_stitcher[key] = stitcher[key]
            for key in ['canvas_to_orig_x', 'canvas_to_orig_y', 'canvas_to_orig_w', 'canvas_to_orig_h', 'canvas_image', 'cropped_to_canvas_x', 'cropped_to_canvas_y', 'cropped_to_canvas_w', 'cropped_to_canvas_h', 'cropped_mask_for_blend']:
                if override:
                    one_stitcher[key] = stitcher[key][0]
                else:
                    one_stitcher[key] = stitcher[key][b]
            one_image = one_image.unsqueeze(0)
            one_image, = self.inpaint_stitch_single_image(one_stitcher, one_image)
            one_image = one_image.squeeze(0)
            one_image = one_image.clone()
            results.append(one_image)

        result_batch = torch.stack(results, dim=0)
        return (result_batch,)

    def inpaint_stitch_single_image(self, stitcher, inpainted_image):
        downscale_algorithm = stitcher['downscale_algorithm']
        upscale_algorithm = stitcher['upscale_algorithm']
        canvas_image = stitcher['canvas_image']
        ctc_x = stitcher['cropped_to_canvas_x']
        ctc_y = stitcher['cropped_to_canvas_y']
        ctc_w = stitcher['cropped_to_canvas_w']
        ctc_h = stitcher['cropped_to_canvas_h']
        cto_x = stitcher['canvas_to_orig_x']
        cto_y = stitcher['canvas_to_orig_y']
        cto_w = stitcher['canvas_to_orig_w']
        cto_h = stitcher['canvas_to_orig_h']
        mask = stitcher['cropped_mask_for_blend']

        output_image = stitch_magic_im(canvas_image, inpainted_image, mask, ctc_x, ctc_y, ctc_w, ctc_h, cto_x, cto_y, cto_w, cto_h, downscale_algorithm, upscale_algorithm)
        return (output_image,)


class InpaintStitcherControl:
    """Expand mask to cover more area for multi-pass inpainting"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitcher": ("STITCHER",),
                "mask": ("MASK",),
                "expand_pixels": ("INT", {"default": 0, "min": 0, "max": 500, "step": 1}),
                "inpaint_blur_pixels": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "stitch_blur_pixels": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("STITCHER", "MASK")
    RETURN_NAMES = ("stitcher", "mask")
    FUNCTION = "process_mask"
    CATEGORY = "inpaint"

    def process_mask(self, stitcher, mask, expand_pixels, inpaint_blur_pixels, stitch_blur_pixels):
        stitcher = stitcher.copy()
        original_blend_pixels = stitcher.get('blend_pixels', 0)
        
        # Gaussian blur expands mask outward, so account for it
        inpaint_binary_expand = max(0, expand_pixels - inpaint_blur_pixels)
        inpaint_mask = mask.clone()
        if inpaint_binary_expand > 0:
            inpaint_mask = expand_m(inpaint_mask, inpaint_binary_expand)
        if inpaint_blur_pixels > 0:
            inpaint_mask = blur_m(inpaint_mask, inpaint_blur_pixels)
        
        actual_blend_pixels = stitch_blur_pixels if stitch_blur_pixels > 0 else original_blend_pixels
        blend_binary_expand = max(0, expand_pixels - actual_blend_pixels)
        blend_mask = mask.clone()
        if blend_binary_expand > 0:
            blend_mask = expand_m(blend_mask, blend_binary_expand)
        if actual_blend_pixels > 0:
            blend_mask = blur_m(blend_mask, actual_blend_pixels)
        
        if isinstance(stitcher['cropped_mask_for_blend'], list):
            batch_idx = 0
            if batch_idx < len(stitcher['cropped_mask_for_blend']):
                stitcher['cropped_mask_for_blend'][batch_idx] = blend_mask
        else:
            stitcher['cropped_mask_for_blend'] = blend_mask
        
        return (stitcher, inpaint_mask)


# ============================================================================
# Node Mappings
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "InpaintCropEnhanced": InpaintCropEnhanced,
    "InpaintStitchImproved": InpaintStitchImproved,
    "InpaintStitcherControl": InpaintStitcherControl,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InpaintCropEnhanced": "🎨 Inpaint Crop Enhanced",
    "InpaintStitchImproved": "🧵 Inpaint Stitch Improved",
    "InpaintStitcherControl": "🎛️ Inpaint Stitch Control",
}
