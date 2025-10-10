import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def simple_multi_threshold(depth_map, target_size, thresholds=[-0.35, 0.15]):
    """
    简化版本的多阈值分类
    输入: depth_map shape (b, 3, h, w), range [-1, 1]
    输出: shape (b, 3, 244, 244), 分类结果 (离散类别值)
    """
    if thresholds is None:
        # 针对 [-1, 1] 范围的阈值设置
        # 分为5个类别: [-1, -0.5, 0, 0.5, 1]
        thresholds = [-0.5, 0.0, 0.5, 0.8]  # 5个类别: 0,1,2,3,4

    # 使用第一个通道进行分类（假设3个通道相同）
    depth = depth_map[:, 0:1, :, :]  # (b, 1, h, w)

    # 创建分类张量
    classes = torch.zeros_like(depth)

    for i, t in enumerate(thresholds):
        classes = torch.where(depth > t, float(i + 1), classes)

    # 调整大小到 244x244
    classes = F.interpolate(classes, size=(target_size, target_size), mode="nearest")

    # 复制到3个通道以匹配输出要求 (b, 3, 244, 244)
    return classes.repeat(1, 3, 1, 1)


def load_and_normalize_depth_images_batch(image_paths, resize_to=None):
    """
    批量加载并归一化多张深度图到[-1,1]。
    参数:
        image_paths: List[str]
        resize_to: (H, W) 或 None。如果提供则统一 resize 到该尺寸（使用最近邻以保持深度离散特征）。
    返回:
        tensor: shape (B, 3, H, W) (若 resize_to 未提供则 H/W 为各自原尺寸中最大填充后裁剪? 这里采取直接各自读入后若尺寸不同并执行 interpolate 到统一尺寸)
        sizes: List[(width,height)] 原始尺寸列表
    说明:
        为简化，若未传入 resize_to，我们将自动使用第一张图尺寸作为目标尺寸。
    """
    images = []
    sizes = []
    # 首先读取所有图像并转换为单通道numpy
    for p in image_paths:
        img = Image.open(p)
        if img.mode in ["RGB", "RGBA"]:
            arr = np.array(img)[:, :, 0]
        else:
            arr = np.array(img)
        sizes.append(img.size)  # (w,h)
        norm = (arr.astype(np.float32) / 255.0) * 2.0 - 1.0  # [-1,1]
        tensor = torch.from_numpy(norm).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        images.append(tensor)

    # 目标尺寸
    if resize_to is None:
        target_h, target_w = images[0].shape[-2], images[0].shape[-1]
    else:
        target_h, target_w = resize_to

    batch_tensors = []
    for t in images:
        if t.shape[-2:] != (target_h, target_w):
            t = F.interpolate(t, size=(target_h, target_w), mode="nearest")
        batch_tensors.append(t)

    batch = torch.cat(batch_tensors, dim=0)  # (B,1,H,W)
    batch = batch.repeat(1, 3, 1, 1)  # (B,3,H,W)
    return batch, sizes


def load_and_normalize_depth_image(image_path):
    """
    加载深度图像并归一化到 [-1, 1] 范围
    输入: 图像路径 (0-255 范围的深度图)
    输出: tensor shape (1, 3, h, w), range [-1, 1]
    """
    # 加载图像
    image = Image.open(image_path)

    # 转换为numpy数组
    if image.mode in ["RGB", "RGBA"]:
        img_array = np.array(image)[:, :, 0]  # 取第一个通道
    else:
        img_array = np.array(image)  # 灰度图

    # 归一化到 [-1, 1] 范围
    # 假设原始范围是 [0, 255]
    normalized_array = (img_array.astype(np.float32) / 255.0) * 2.0 - 1.0

    # 转换为tensor
    tensor = torch.from_numpy(normalized_array).float()

    # 添加 batch 和 channel 维度: (1, 1, h, w)
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)

    # 复制到3个通道: (1, 3, h, w)
    tensor_3ch = tensor.repeat(1, 3, 1, 1)

    return tensor_3ch, image.size


def save_comparison_images(original_tensor, classified_tensor, save_dir, filename):
    """
    保存原始图像和分类结果的对比图
    original_tensor: (1, 3, h, w) 范围 [-1, 1]
    classified_tensor: (1, 3, 244, 244) 分类结果
    """
    os.makedirs(save_dir, exist_ok=True)

    # 转换为numpy用于显示
    original_np = original_tensor[0, 0].cpu().numpy()  # 取第一个通道
    classified_np = classified_tensor[0, 0].cpu().numpy()  # 取第一个通道

    # 创建对比图
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 原始深度图 (归一化后的 [-1, 1] 范围)
    im1 = axes[0].imshow(original_np, cmap="viridis", vmin=-1, vmax=1)
    axes[0].set_title("Normalized Depth Map [-1, 1]", fontsize=12)
    axes[0].axis("off")
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label("Normalized Depth", fontsize=10)

    # 分类结果
    num_classes = int(classified_np.max()) + 1
    im2 = axes[1].imshow(classified_np, cmap="tab10", vmin=0, vmax=max(4, num_classes - 1))
    axes[1].set_title(f"Multi-threshold Classification ({num_classes} classes)", fontsize=12)
    axes[1].axis("off")
    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, ticks=range(num_classes))
    cbar2.set_label("Class Index", fontsize=10)

    plt.tight_layout()
    comparison_path = os.path.join(save_dir, f"comparison_{filename}.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close()

    # # 保存单独的分类结果（作为彩色图像）
    # classified_color = plt.get_cmap("tab10")(classified_np / (classified_np.max() + 1e-8))
    # classified_pil = Image.fromarray((classified_color[:, :, :3] * 255).astype(np.uint8))
    # classified_path = os.path.join(save_dir, f"classified_{filename}.png")
    # classified_pil.save(classified_path)

    print(f"Saved: comparison_{filename}.png | classified_{filename}.png")


def _set_seed(seed: int = 42):
    """固定随机种子，确保可复现。"""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 尽量保证确定性
    if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def process_images_for_training(
    image_dir, num_samples=10, thresholds=None, resize_to=None, seed: int = 42, target_size=244
):
    """
    处理图像用于训练准备
    将原始 [0,255] 深度图转换为 [-1,1] 范围，然后进行多阈值分类
    """

    # 设置随机种子（控制 random.sample 与后续任何随机行为）
    _set_seed(seed)

    # 获取目录中的图像文件
    image_extensions = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]
    image_files = []
    for file in os.listdir(image_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)

    # 随机选择 num_samples 张图像（受 seed 影响）
    if len(image_files) < num_samples:
        print(f"Warning: Only {len(image_files)} images available, processing all of them.")
        selected_files = image_files
    else:
        selected_files = random.sample(image_files, num_samples)

    print(f"Found {len(image_files)} images, randomly selected {len(selected_files)} for processing.")

    # 处理每张图像
    save_dir = "/home/zhangwh/Projects/mydiff/image/training_depth_classification"
    os.makedirs(save_dir, exist_ok=True)

    results_info = []

    print("\n== Batch mode enabled ==")
    image_paths = [os.path.join(image_dir, f) for f in selected_files]
    batch_tensor, original_sizes = load_and_normalize_depth_images_batch(image_paths, resize_to=resize_to)
    print(
        f"Loaded batch tensor shape: {batch_tensor.shape}, range: [{batch_tensor.min():.3f}, {batch_tensor.max():.3f}]"
    )
    classified_batch = simple_multi_threshold(
        batch_tensor, target_size=target_size, thresholds=thresholds
    )  # (B,3,244,244)
    print(f"Classified batch shape: {classified_batch.shape}")

    for idx, image_file in enumerate(selected_files):
        single_norm = batch_tensor[idx : idx + 1]
        single_cls = classified_batch[idx : idx + 1]
        filename = os.path.splitext(image_file)[0]
        # 保存对比
        save_comparison_images(single_norm, single_cls, save_dir, filename)
        # 记录
        results_info.append(
            {
                "filename": image_file,
                "normalized_shape": tuple(single_norm.shape),
                "normalized_range": [single_norm.min().item(), single_norm.max().item()],
                "output_shape": tuple(single_cls.shape),
                "num_classes": len(torch.unique(single_cls)),
            }
        )

    # 打印总结信息
    print(f"\n{'=' * 60}")
    print(f"Training Data Processing Summary:")
    print(f"Total images processed: {len(results_info)}")
    print(f"Results saved to: {os.path.abspath(save_dir)}")
    print(f"Input format: (b, 3, h, w) with range [-1, 1]")
    print(f"Output format: (b, 3, 244, 244) with discrete class values")
    print(f"Default thresholds used: {[-0.5, 0.0, 0.5, 0.8] if thresholds is None else thresholds}")
    print(f"Seed used: {seed}")
    print(f"{'=' * 60}")

    return results_info


if __name__ == "__main__":
    print("\n=== Processing Real Images for Training ===")

    # 图像路径
    image_dir = "/mnt/data/zwh/data/maxar/flooding/brazil-flood-2024-5/before_depth"

    # 自定义阈值（针对 [-1, 1] 范围）
    custom_thresholds = [-0.35, 0.15]  # 可根据需要调整

    # 处理10张随机图像
    results = process_images_for_training(
        image_dir=image_dir, num_samples=10, thresholds=custom_thresholds, target_size=244
    )
