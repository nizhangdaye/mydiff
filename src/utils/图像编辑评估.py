# evaluator.py - 面向对象的图像编辑评估脚本

import os

# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HOME"] = "/mnt/data/zwh/cache/huggingface"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


import argparse
import logging
from abc import ABC, abstractmethod
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from datasets import load_from_disk
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModel,
    CLIPImageProcessor,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

# 依赖库
from diffusers import StableDiffusionInstructPix2PixPipeline

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------------------------
# 图像拼接工具函数
# -------------------------------
def create_comparison_image(before_img, edited_img, after_img, prompt, metrics=None):
    """创建对比图像：灾前|编辑|灾后"""
    # 确保所有图像尺寸一致
    size = (512, 512)
    before_img = (
        before_img.resize(size)
        if before_img
        else Image.new("RGB", size, (128, 128, 128))
    )
    edited_img = (
        edited_img.resize(size)
        if edited_img
        else Image.new("RGB", size, (128, 128, 128))
    )
    after_img = (
        after_img.resize(size) if after_img else Image.new("RGB", size, (128, 128, 128))
    )

    # 创建拼接图像 (宽度为3倍，高度增加标题空间)
    title_height = 80
    total_width = size[0] * 3
    total_height = size[1] + title_height * 2  # 上下都留空间

    combined = Image.new("RGB", (total_width, total_height), (255, 255, 255))

    # 粘贴图像
    combined.paste(before_img, (0, title_height))
    combined.paste(edited_img, (size[0], title_height))
    combined.paste(after_img, (size[0] * 2, title_height))

    # 添加标题
    draw = ImageDraw.Draw(combined)
    try:
        # 尝试使用系统字体
        font_large = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24
        )
        font_small = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16
        )
    except:
        # 如果没有找到字体，使用默认字体
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # 图像标题
    titles = ["灾前 (Before)", "编辑结果 (Edited)", "真实灾后 (After)"]
    for i, title in enumerate(titles):
        x = size[0] * i + size[0] // 2
        # 计算文字宽度以居中
        bbox = draw.textbbox((0, 0), title, font=font_large)
        text_width = bbox[2] - bbox[0]
        draw.text((x - text_width // 2, 10), title, fill=(0, 0, 0), font=font_large)

    # 添加编辑提示
    prompt_text = (
        f"编辑提示: {prompt[:100]}..." if len(prompt) > 100 else f"编辑提示: {prompt}"
    )
    bbox = draw.textbbox((0, 0), prompt_text, font=font_small)
    text_width = bbox[2] - bbox[0]
    draw.text(
        (total_width // 2 - text_width // 2, total_height - 60),
        prompt_text,
        fill=(0, 0, 0),
        font=font_small,
    )

    # 添加指标信息
    if metrics:
        metrics_text = f"CLIPim: {metrics.get('CLIPim', 0):.3f} | CLIPout: {metrics.get('CLIPout', 0):.3f} | DINO: {metrics.get('DINO', 0):.3f} | SSIM: {metrics.get('SSIM', 0):.3f}"
        bbox = draw.textbbox((0, 0), metrics_text, font=font_small)
        text_width = bbox[2] - bbox[0]
        draw.text(
            (total_width // 2 - text_width // 2, total_height - 30),
            metrics_text,
            fill=(0, 0, 0),
            font=font_small,
        )

    return combined


# -------------------------------
# 数据集处理类
# -------------------------------
class ImageEditingDataset(Dataset):
    """自定义数据集类，用于图像编辑评估"""

    def __init__(self, dataset, resolution=512):
        self.dataset = dataset
        self.resolution = resolution
        self.transform = transforms.Compose(
            [
                transforms.Resize((resolution, resolution)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # 处理 before 图像
        before_img = sample.get("before", sample.get("input_image"))
        if before_img is not None:
            if not isinstance(before_img, Image.Image):
                before_img = Image.fromarray(before_img)
            before_img = before_img.convert("RGB")

        # 处理 after 图像
        after_img = sample.get("after", sample.get("target_image"))
        if after_img is not None:
            if not isinstance(after_img, Image.Image):
                after_img = Image.fromarray(after_img)
            after_img = after_img.convert("RGB")

        # 获取编辑提示
        edit_prompt = sample.get("edit", sample.get("instruction", ""))
        edited_prompt = sample.get("edited")
        original_prompt = sample.get("original")

        return {
            "before": before_img,
            "after": after_img,
            "edit_prompt": edit_prompt,  # 编辑指令
            "edited_prompt": edited_prompt,  # 灾后图像描述
            "original_prompt": original_prompt,  # 灾前图像描述
            "sample_id": sample.get("id", idx),
        }


def collate_fn(batch):
    """自定义 collate 函数"""
    before_images = [item["before"] for item in batch]
    after_images = [item["after"] for item in batch]
    edit_prompts = [item["edit_prompt"] for item in batch]
    edited_prompts = [item["edited_prompt"] for item in batch]
    original_prompts = [item["original_prompt"] for item in batch]
    sample_ids = [item["sample_id"] for item in batch]

    return {
        "before": before_images,
        "after": after_images,
        "edit_prompt": edit_prompts,
        "edited_prompt": edited_prompts,
        "original_prompt": original_prompts,
        "sample_id": sample_ids,
    }


# -------------------------------
# 抽象基类：BaseEvaluator
# -------------------------------
class BaseEvaluator(ABC):
    def __init__(self, pipeline, dataloader, args, device, accelerator=None):
        self.pipeline = pipeline.to(device)
        self.dataloader = dataloader
        self.args = args
        self.device = device
        self.accelerator = accelerator
        self.results = []
        self.metrics = {
            "CLIPim": [],
            "CLIPout": [],
            "DINO": [],
            "SSIM": [],
            "L1_before": [],
            "L1_after": [],
            "L2_before": [],
            "L2_after": [],
        }

        # 创建图像保存目录
        if self.args.save_images and accelerator and accelerator.is_main_process:
            self.images_dir = os.path.join(args.output_dir, "comparison_images")
            os.makedirs(self.images_dir, exist_ok=True)

    @abstractmethod
    def compute_metrics(
        self, before_img, edited_img, after_img, edit_prompt, edited_prompt
    ):
        """子类实现：计算每张图像的评估指标"""
        pass

    def run(self):
        """主评估流程"""
        logger.info("Starting evaluation...")
        self.pipeline.set_progress_bar_config(disable=True)

        # 根据设备类型选择自动混合精度
        if torch.backends.mps.is_available():
            autocast_ctx = nullcontext()
        else:
            autocast_ctx = torch.autocast(self.accelerator.device.type)

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Evaluating")):
                before_images = batch["before"]
                after_images = batch["after"]
                edit_prompts = batch["edit_prompt"]
                edited_prompts = batch["edited_prompt"]
                original_prompts = batch["original_prompt"]
                sample_ids = batch["sample_id"]

                for i, (
                    before_img,
                    after_img,
                    edit_prompt,
                    edited_prompt,
                    original_prompt,
                    sample_id,
                ) in enumerate(
                    zip(
                        before_images,
                        after_images,
                        edit_prompts,
                        edited_prompts,
                        original_prompts,
                        sample_ids,
                    )
                ):
                    # 预处理图像
                    before_pil = self._preprocess_image(before_img)
                    after_pil = (
                        self._preprocess_image(after_img)
                        if after_img is not None
                        else None
                    )

                    # 生成编辑图像
                    with autocast_ctx:
                        edited_image = self._generate_edited_image(
                            before_pil, edit_prompt, sample_id
                        )

                    # 计算指标
                    metrics = self.compute_metrics(
                        before_pil, edited_image, after_pil, edit_prompt, edited_prompt
                    )

                    # 保存对比图像
                    if (
                        self.args.save_images
                        and self.accelerator
                        and self.accelerator.is_main_process
                    ):
                        comparison_img = create_comparison_image(
                            before_pil, edited_image, after_pil, edit_prompt, metrics
                        )
                        img_filename = f"sample_{sample_id}_{batch_idx}_{i}.jpg"
                        img_path = os.path.join(self.images_dir, img_filename)
                        comparison_img.save(img_path, quality=95)

                    # 保存结果
                    result = {
                        "sample_id": sample_id,
                        "batch_idx": batch_idx,
                        "item_idx": i,
                        "before": before_pil,
                        "edited": edited_image,
                        "after": after_pil,
                        "edit_prompt": edit_prompt,
                        "edited_prompt": edited_prompt,
                        **metrics,
                    }
                    self.results.append(result)

                    # 累加到 metrics
                    for k, v in metrics.items():
                        if k in self.metrics and v is not None:
                            self.metrics[k].append(v)

        self._log_results()
        if self.accelerator:
            self._log_to_wandb()
        return self.metrics

    def _preprocess_image(self, img):
        if img is None:
            return None
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        return img.convert("RGB").resize((self.args.resolution, self.args.resolution))

    def _generate_edited_image(self, before_img, prompt, seed):
        # 使用样本ID作为种子，确保可重现性
        generator = torch.Generator(self.device).manual_seed(hash(str(seed)) % (2**32))
        output = self.pipeline(
            prompt,
            image=before_img,
            num_inference_steps=self.args.num_inference_steps,
            image_guidance_scale=self.args.image_guidance_scale,
            guidance_scale=self.args.guidance_scale,
            generator=generator,
            output_type="pil",  # 默认就为 PIL
        )
        return output.images[0]

    def _log_results(self):
        logger.info("=== Evaluation Results ===")
        for name, values in self.metrics.items():
            if values:  # 只有非空列表才计算统计
                mean_val = np.mean(values)
                std_val = np.std(values)
                logger.info(f"{name:12s}: {mean_val:.4f} ± {std_val:.4f}")

        if self.args.save_images and hasattr(self, "images_dir"):
            logger.info(f"Comparison images saved to: {self.images_dir}")

    def _log_to_wandb(self):
        if not self.accelerator or "wandb" not in [
            tracker.name for tracker in self.accelerator.trackers
        ]:
            return

        columns = [
            "sample_id",
            "prompt",
            "before",
            "edited",
            "after",
            *self.metrics.keys(),
        ]
        table = wandb.Table(columns=columns)

        for res in self.results:
            row = [
                res["sample_id"],
                res["prompt"],
                wandb.Image(res["before"]),
                wandb.Image(res["edited"]),
                wandb.Image(res["after"]) if res["after"] else None,
            ]
            row.extend(res.get(k, None) for k in self.metrics.keys())
            table.add_data(*row)

        for tracker in self.accelerator.trackers:
            if tracker.name == "wandb":
                tracker.log({"evaluation": table})


class ImageEditingEvaluator(BaseEvaluator):
    def __init__(self, pipeline, dataloader, args, device, accelerator=None):
        super().__init__(pipeline, dataloader, args, device, accelerator)
        (
            self.clip_tokenizer,
            self.clip_text_encoder,
            self.clip_image_processor,
            self.clip_image_encoder,
        ) = self._load_clip()
        self.dino_processor, self.dino_model = self._load_dino()

    def _load_clip(self):
        logger.info("Loading CLIP model (openai/clip-vit-large-patch14)...")
        clip_id = "openai/clip-vit-large-patch14"
        tokenizer = CLIPTokenizer.from_pretrained(clip_id)
        text_encoder = (
            CLIPTextModelWithProjection.from_pretrained(clip_id).to(self.device).eval()
        )
        image_processor = CLIPImageProcessor.from_pretrained(clip_id)
        image_encoder = (
            CLIPVisionModelWithProjection.from_pretrained(clip_id)
            .to(self.device)
            .eval()
        )
        return tokenizer, text_encoder, image_processor, image_encoder

    def _load_dino(self):
        logger.info("Loading DINOv2 model (facebook/dinov2-base)...")
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        model = AutoModel.from_pretrained("facebook/dinov2-base").to(self.device).eval()
        return processor, model

    def compute_clip_similarity(self, img: Image.Image = None, text: str = None):
        with torch.no_grad():
            if text is not None:
                text_inputs = self.clip_tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77,
                ).to(self.device)
                text_outputs = self.clip_text_encoder(**text_inputs)
                feat = text_outputs.text_embeds
            else:
                image_inputs = self.clip_image_processor(
                    images=img, return_tensors="pt"
                ).to(self.device)
                image_outputs = self.clip_image_encoder(**image_inputs)
                feat = image_outputs.image_embeds
            return F.normalize(feat, dim=-1)

    def compute_dino_similarity(self, img1: Image.Image, img2: Image.Image):
        with torch.no_grad():
            inputs = self.dino_processor(images=[img1, img2], return_tensors="pt").to(
                self.device
            )
            outputs = self.dino_model(**inputs)
            cls_feats = outputs.last_hidden_state[:, 0, :]  # [2, D]
            sim = F.cosine_similarity(cls_feats[0:1], cls_feats[1:2]).item()
        return sim

    def compute_metrics(
        self, before_img, edited_img, after_img, edit_prompt, edited_prompt
    ):
        # CLIPim: edited vs before
        before_clip_feat = self.compute_clip_similarity(img=before_img)
        edited_clip_feat = self.compute_clip_similarity(img=edited_img)
        clipim = F.cosine_similarity(edited_clip_feat, before_clip_feat).item()

        # CLIPout: edited vs prompt
        text_clip_feat = self.compute_clip_similarity(text=edited_prompt)
        clipout = F.cosine_similarity(edited_clip_feat, text_clip_feat).item()

        # DINO: edited vs before
        dino_sim = self.compute_dino_similarity(before_img, edited_img)

        # SSIM, L1, L2
        def to_float_array(img):
            return np.array(img).astype(np.float32) / 255.0

        before_np = to_float_array(before_img)
        edited_np = to_float_array(edited_img)
        ssim_val = ssim(
            before_np, edited_np, multichannel=True, data_range=1.0, channel_axis=2
        )
        l1_before = np.mean(np.abs(edited_np - before_np))
        l2_before = np.mean((edited_np - before_np) ** 2)

        l1_after = l2_after = None
        if after_img is not None:
            after_np = to_float_array(after_img)
            l1_after = np.mean(np.abs(edited_np - after_np))
            l2_after = np.mean((edited_np - after_np) ** 2)

        return {
            "CLIPim": clipim,
            "CLIPout": clipout,
            "DINO": dino_sim,
            "SSIM": ssim_val,
            "L1_before": l1_before,
            "L1_after": l1_after,
            "L2_before": l2_before,
            "L2_after": l2_after,
        }


# -------------------------------
# 主函数
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/mnt/data/zwh/log/instruct-pix2pix/experiment_0",
        help="预训练模型路径",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/mnt/data/zwh/data/maxar/disaster_dataset",
        help="数据集路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/data/zwh/log/instruct-pix2pix/Experiments/experiment_0",
        help="结果保存路径",
    )
    parser.add_argument("--resolution", type=int, default=512, help="图像分辨率")
    parser.add_argument("--batch_size", type=int, default=8, help="评估 batch size")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="推理步数")
    parser.add_argument(
        "--image_guidance_scale", type=float, default=1.5, help="图像引导系数"
    )
    parser.add_argument("--guidance_scale", type=float, default=7, help="文本引导系数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--save_images", action="store_true", help="是否保存对比图像 (灾前|编辑|灾后)"
    )

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 初始化 Accelerator
    accelerator = Accelerator(
        project_dir=args.output_dir,
    )

    # 加载模型
    logger.info(f"Loading pipeline from {args.pretrained_model_name_or_path}")
    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )

    # 优化模型
    if hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
        pipeline.enable_xformers_memory_efficient_attention()

    # 加载数据集
    logger.info(f"Loading dataset from {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)
    eval_dataset = dataset["test"] if "test" in dataset else dataset["validation"]

    # 创建数据集和 DataLoader
    with accelerator.main_process_first():
        eval_dataset_processed = ImageEditingDataset(
            eval_dataset, resolution=args.resolution
        )

    dataloader = DataLoader(
        eval_dataset_processed,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    # 使用 accelerator 准备 dataloader
    dataloader = accelerator.prepare(dataloader)

    # 创建评估器
    evaluator = ImageEditingEvaluator(
        pipeline=pipeline,
        dataloader=dataloader,
        args=args,
        device=accelerator.device,
        accelerator=accelerator,
    )

    # 运行评估
    metrics = evaluator.run()

    # 保存结果到 JSON
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        import json

        summary = {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}
        summary["num_samples"] = len(evaluator.results)

        with open(
            os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {os.path.join(args.output_dir, 'metrics.json')}")

        if args.save_images:
            logger.info(
                f"Comparison images saved to {os.path.join(args.output_dir, 'comparison_images')}"
            )

    logger.info("✅ Evaluation completed.")


if __name__ == "__main__":
    main()
