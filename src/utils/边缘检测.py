# This is an improved version and model of HED edge detection with Apache License, Version 2.0.
# Please use this implementation in your products
# This implementation may produce slightly different results from Saining Xie's official implementations,
# but it generates smoother edges and is more suitable for ControlNet as well as other image-to-image translations.
# Different from official models and other implementations, this is an RGB-input model (rather than BGR)
# and in this way it works better for gradio's RGB protocol

import os
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
from einops import rearrange


class DoubleConvBlock(torch.nn.Module):
    def __init__(self, input_channel, output_channel, layer_number):
        super().__init__()
        self.convs = torch.nn.Sequential()
        self.convs.append(
            torch.nn.Conv2d(
                in_channels=input_channel, out_channels=output_channel, kernel_size=(3, 3), stride=(1, 1), padding=1
            )
        )
        for i in range(1, layer_number):
            self.convs.append(
                torch.nn.Conv2d(
                    in_channels=output_channel,
                    out_channels=output_channel,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=1,
                )
            )
        self.projection = torch.nn.Conv2d(
            in_channels=output_channel, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=0
        )

    def __call__(self, x, down_sampling=False):
        h = x
        if down_sampling:
            h = torch.nn.functional.max_pool2d(h, kernel_size=(2, 2), stride=(2, 2))
        for conv in self.convs:
            h = conv(h)
            h = torch.nn.functional.relu(h)
        return h, self.projection(h)


class ControlNetHED_Apache2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.Parameter(torch.zeros(size=(1, 3, 1, 1)))
        self.block1 = DoubleConvBlock(input_channel=3, output_channel=64, layer_number=2)
        self.block2 = DoubleConvBlock(input_channel=64, output_channel=128, layer_number=2)
        self.block3 = DoubleConvBlock(input_channel=128, output_channel=256, layer_number=3)
        self.block4 = DoubleConvBlock(input_channel=256, output_channel=512, layer_number=3)
        self.block5 = DoubleConvBlock(input_channel=512, output_channel=512, layer_number=3)

    def __call__(self, x):
        h = x - self.norm
        h, projection1 = self.block1(h)
        h, projection2 = self.block2(h, down_sampling=True)
        h, projection3 = self.block3(h, down_sampling=True)
        h, projection4 = self.block4(h, down_sampling=True)
        h, projection5 = self.block5(h, down_sampling=True)
        return projection1, projection2, projection3, projection4, projection5


class HEDDetector(torch.nn.Module):
    def __init__(self, hed_cfg: Optional[Dict[str, Any]] = None):
        """
        HED 边缘检测器封装类

        Args:
            hed_cfg: 配置字典，包含:
                - model_path: str, 模型权重路径
        """
        super().__init__()

        # 构建模型
        self.hed_model = ControlNetHED_Apache2()

        # 加载权重
        sd = torch.load(hed_cfg["model_path"], map_location="cpu")
        self.hed_model.load_state_dict(sd, strict=True)

        # 固定模型
        self.hed_model.eval()
        for p in self.hed_model.parameters():
            p.requires_grad_(False)

        # 注册为子模块，确保 .to(device) 能正确迁移
        self.add_module("hed_model", self.hed_model)

    def forward(self, images_01: torch.Tensor) -> torch.Tensor:
        """
        计算批量图像的 HED 边缘图

        Args:
            images_01: [B, 3, H, W], 范围 [0, 1]

        Returns:
            edges: [B, 1, H, W], 范围 [0, 1]
        """
        assert images_01.dim() == 4 and images_01.size(1) == 3, "images_01 应为 [B,3,H,W]"
        B, C, H, W = images_01.shape

        # 转换为 HED 期望的 [0, 255] 范围
        x = images_01 * 255.0

        # 使用混合精度（可选：根据实际稳定性决定是否保留）
        # with torch.autocast(images_01.device.type, enabled=True):
        p1, p2, p3, p4, p5 = self.hed_model(x)
        ps = [
            torch.nn.functional.interpolate(p, size=(H, W), mode="bilinear", align_corners=False)
            for p in (p1, p2, p3, p4, p5)
        ]
        stacked = torch.stack(ps, dim=1)  # (B, 5, 1, H, W)
        mean_map = stacked.mean(dim=1)  # (B, 1, H, W)
        edges = torch.sigmoid(mean_map)  # (B, 1, H, W), [0,1]

        return edges


def nms(x, t, s):
    x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), s)

    f1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    f2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    f3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    f4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)

    y = np.zeros_like(x)

    for f in [f1, f2, f3, f4]:
        np.putmask(y, cv2.dilate(x, kernel=f) == x, x)

    z = np.zeros_like(y, dtype=np.uint8)
    z[y > t] = 255
    return z


if __name__ == "__main__":
    import PIL.Image

    hed_cfg = {"enabled": True, "model_path": "/mnt/data/zwh/model/HED/ControlNetHED.pth"}
    hed = HEDDetector(hed_cfg)
    image = PIL.Image.open(
        "/mnt/data/zwh/data/maxar/flooding/brazil-flood-2024-5/before/10300100FA870500_213131103110_223_from_10300100EE07F000_213131103110.png"
    ).convert("RGB")
    # 转为 numpy 并再转为 PyTorch 张量 [B,3,H,W], 归一化到 [0,1]
    image_np = np.array(image)
    image_t = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        edge_t = hed(image_t)  # [1,1,H,W], float in [0,1]
    # 转回 numpy 单通道灰度图，uint8，供 OpenCV 使用
    edge = (edge_t[0, 0].cpu().numpy() * 255.0).astype(np.uint8)
    edge_nms = nms(edge, 100, 1.0)
    # 计算传统边缘检测结果（Canny / Sobel / Laplacian）
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Canny 边缘
    canny = cv2.Canny(gray, 100, 200)

    # Sobel 梯度幅值
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(sobelx, sobely)
    if sobel_mag.max() > 0:
        sobel = np.uint8(255.0 * sobel_mag / sobel_mag.max())
    else:
        sobel = np.zeros_like(gray, dtype=np.uint8)

    # Laplacian 二阶边缘
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    lap_abs = np.abs(lap)
    if lap_abs.max() > 0:
        lap_norm = np.uint8(255.0 * lap_abs / lap_abs.max())
    else:
        lap_norm = np.zeros_like(gray, dtype=np.uint8)

    # 转为三通道便于与原图拼接与标注
    def to_color(g):
        return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)

    def label_panel(img_rgb, text):
        panel = img_rgb.copy()
        pad_w = 10 + int(len(text) * 12)
        cv2.rectangle(panel, (6, 6), (6 + pad_w, 34), (0, 0, 0), -1)
        cv2.putText(panel, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return panel

    p_orig = label_panel(image_np, "原图")
    p_hed = label_panel(to_color(edge), "HED")
    p_nms = label_panel(to_color(edge_nms), "HED+NMS")
    p_canny = label_panel(to_color(canny), "Canny")
    p_sobel = label_panel(to_color(sobel), "Sobel")
    p_lap = label_panel(to_color(lap_norm), "Laplacian")

    # 2x3 网格拼接：
    row1 = np.concatenate([p_orig, p_hed, p_nms], axis=1)
    row2 = np.concatenate([p_canny, p_sobel, p_lap], axis=1)
    combined = np.concatenate([row1, row2], axis=0)

    PIL.Image.fromarray(combined).save("hed_results_combined.png")
