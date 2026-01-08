# ComfyUI-txkbfnodes

快速图像保存节点 - 高性能本地保存,支持 WebP 压缩

## 功能特点

- ⚡ **高速保存**: 优化的保存逻辑,最快速度写入磁盘
- 🗜️ **WebP 压缩**: 支持 WebP 高质量压缩,文件更小
- 📁 **路径返回**: 保存后返回完整文件路径,方便后续使用
- 🎨 **多格式支持**: 支持 WebP、PNG、JPG 格式
- 📦 **批量处理**: 支持批量保存多张图像

## 安装

1. 将此文件夹放置到 `ComfyUI/custom_nodes/` 目录下
2. 重启 ComfyUI 或刷新自定义节点

## 节点说明

### 1. Fast Image Saver (快速保存图像)

**输入参数**:
- `images`: 图像输入 (IMAGE 类型)
- `filename_prefix`: 文件名前缀 (默认: "ComfyUI")
- `format`: 保存格式 (webp/png/jpg)
- `webp_quality`: 压缩质量 (1-100,默认: 90)
- `subfolder`: 子文件夹名称 (可选)

**输出**:
- `file_path`: 完整文件路径 (如: `D:\ComfyUI\output\image_20251017_123456.webp`)
- `filename`: 文件名 (如: `ComfyUI_20251017_123456.webp`)

**特点**:
- 自动添加时间戳,避免文件覆盖
- 返回最后一张图像的路径

### 2. Fast Image Saver Batch (批量快速保存)

**输入参数**: 同上

**输出**:
- `file_paths`: 所有文件路径 (用换行符分隔的字符串)

**特点**:
- 批量保存时返回所有文件路径
- 适合需要处理多个文件路径的场景

## 使用示例

### 基础使用

```
[Load Image] → [Your Processing Nodes] → [Fast Image Saver]
                                              ↓
                                         [file_path输出]
```

### 格式选择建议

- **WebP**: 推荐用于日常保存,质量90可获得极佳的压缩比和质量平衡
- **PNG**: 需要无损保存时使用,文件较大
- **JPG**: 不支持透明通道,适合照片类图像

### 性能优化

节点已针对速度进行优化:
- WebP: `method=6` (平衡模式)
- PNG: `compress_level=6`, `optimize=False`
- JPG: `optimize=False`

如需更高压缩率但速度稍慢,可修改源码中的参数。

## 文件保存位置

默认保存到 ComfyUI 的 `output` 目录。
可通过 `subfolder` 参数指定子文件夹,如: `saved_images`

## 注意事项

1. 文件名会自动添加时间戳和微秒,确保唯一性
2. 批量保存时会在文件名后添加序号 (如: `_0001`, `_0002`)
3. JPG 格式会自动将 RGBA 图像转换为 RGB

## 更新日志

### v1.0.0 (2025-10-17)
- 初始版本
- 支持 WebP/PNG/JPG 格式
- 快速保存优化
- 返回文件路径

## 许可证

MIT License

## 作者

txkbfnodes
