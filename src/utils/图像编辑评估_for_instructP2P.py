# evaluator.py - 面向对象的图像编辑评估脚本

import argparse
import logging
import os
import sys
from abc import ABC, abstractmethod
from contextlib import nullcontext

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# 环境变量设置
os.environ["HF_HOME"] = "/mnt/data/zwh/cache/huggingface"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 设置字体为黑体，解决中文显示问题

import lpips
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModel,
    CLIPImageProcessor,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from diffusers import StableDiffusionInstructPix2PixPipeline
from src.data.instruct_pix2pix_dataset import ImageEditDataset

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
    before_img = before_img.resize(size) if before_img else Image.new("RGB", size, (128, 128, 128))
    edited_img = edited_img.resize(size) if edited_img else Image.new("RGB", size, (128, 128, 128))
    after_img = after_img.resize(size) if after_img else Image.new("RGB", size, (128, 128, 128))

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
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
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
    prompt_text = f"编辑提示: {prompt[:100]}..." if len(prompt) > 100 else f"编辑提示: {prompt}"
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
# 评估数据集适配器
# -------------------------------
class EvaluationDataset(Dataset):
    """评估数据集，适配 ImageEditDataset 为评估格式"""

    def __init__(
        self,
        dataset_path: str,
        tokenizer: CLIPTokenizer,
        resolution: int = 512,
        split: str = "test",
        seed: int = 42,
        use_fixed_edit_text: bool = False,
    ):
        self.fixed_edit_mapping = {
            "fire": "add fire to the image",
            "flooding": "add flooding to the image",
            "tornado": "add tornado damage to the image",
            "earthquake": "add earthquake damage to the image",
            "hurricane": "add hurricane damage to the image",
            "wildfire": "add wildfire damage to the image",
            "landslide": "add landslide damage to the image",
            "volcano": "add volcanic damage to the image",
        }
        self.use_fixed_edit_text = use_fixed_edit_text
        self.image_edit_dataset = ImageEditDataset(
            dataset_path=dataset_path,
            tokenizer=tokenizer,
            resolution=resolution,
            split=split,
            center_crop=False,
            random_flip=False,
            seed=seed,
        )

    def __len__(self):
        return len(self.image_edit_dataset)

    def __getitem__(self, idx):
        raw_sample = self.image_edit_dataset.dataset[idx]

        # 处理图像
        before_img = self._process_image(raw_sample.get("before"))
        after_img = self._process_image(raw_sample.get("after"))
        if self.use_fixed_edit_text:
            edit_prompt = self.fixed_edit_mapping.get(raw_sample["disaster_type"])
        else:
            edit_prompt = raw_sample["edit"]

        return {
            "before": before_img,
            "after": after_img,
            "edit_prompt": edit_prompt,
            "edited_prompt": raw_sample.get("edited", ""),
            "original_prompt": raw_sample.get("original", ""),
            "sample_id": raw_sample.get("id", idx),
        }

    def _process_image(self, img):
        if img is None:
            return None
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        return img.convert("RGB")


def collate_fn(batch):
    """批处理函数"""
    return {
        "before": [item["before"] for item in batch],
        "after": [item["after"] for item in batch],
        "edit_prompt": [item["edit_prompt"] for item in batch],
        "edited_prompt": [item["edited_prompt"] for item in batch],
        "original_prompt": [item["original_prompt"] for item in batch],
        "sample_id": [item["sample_id"] for item in batch],
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
            "CLIPdir": [],
            "CLIPafter": [],
            "DINO": [],
            "SSIM": [],
            "LPIPS_before": [],
            "LPIPS_after": [],
            "PSNR_before": [],
            "PSNR_after": [],
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
    def compute_metrics(self, before_img, edited_img, after_img, edit_prompt, edited_prompt):
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

                # 批量预处理图像
                before_pils = [self._preprocess_image(img) for img in before_images]
                after_pils = [self._preprocess_image(img) if img is not None else None for img in after_images]

                # 批量生成编辑图像
                with autocast_ctx:
                    edited_images = self._generate_edited_image(before_pils, edit_prompts, sample_ids)

                # 批量计算指标和保存结果
                for i, (
                    before_pil,
                    after_pil,
                    edited_image,
                    edit_prompt,
                    edited_prompt,
                    original_prompt,
                    sample_id,
                ) in enumerate(
                    zip(
                        before_pils,
                        after_pils,
                        edited_images,
                        edit_prompts,
                        edited_prompts,
                        original_prompts,
                        sample_ids,
                    )
                ):
                    # 检查 edit_prompt 是否为空，如果为空则跳过指标计算
                    if not edit_prompt or edit_prompt.strip() == "" or edit_prompt == "add fire to the image":
                        continue

                    # 计算指标
                    metrics = self.compute_metrics(
                        before_pil,
                        edited_image,
                        after_pil,
                        edit_prompt,
                        edited_prompt,
                        original_prompt,
                    )

                    # 保存对比图像
                    if self.args.save_images and self.accelerator and self.accelerator.is_main_process:
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

    def _generate_edited_image(self, before_imgs, prompts, seeds):
        # 使用样本ID作为种子，确保可重现性
        generators = [torch.Generator(self.device).manual_seed(hash(str(seed)) % (2**32)) for seed in seeds]

        # 使用批量推理
        outputs = self.pipeline(
            prompts,
            image=before_imgs,
            num_inference_steps=self.args.num_inference_steps,
            image_guidance_scale=self.args.image_guidance_scale,
            guidance_scale=self.args.guidance_scale,
            generator=generators,
            output_type="pil",  # 默认输出为 PIL 图片
        )
        return outputs.images

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
        if not self.accelerator or "wandb" not in [tracker.name for tracker in self.accelerator.trackers]:
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
        # 加载评估模型
        logger.info("Loading evaluation models...")

        # CLIP模型
        clip_id = "openai/clip-vit-large-patch14"
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_id)
        self.clip_text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_id).to(self.device).eval()
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(clip_id)
        self.clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_id).to(self.device).eval()

        # DINO模型
        self.dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to(self.device).eval()

        # LPIPS模型
        self.lpips_model = lpips.LPIPS(net="vgg").to(self.device).eval()

    def compute_clip_similarity(self, img: Image.Image = None, text: str = None):
        with torch.no_grad():
            if text is not None:
                inputs = self.clip_tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77,
                ).to(self.device)
                feat = self.clip_text_encoder(**inputs).text_embeds
            else:
                inputs = self.clip_image_processor(images=img, return_tensors="pt").to(self.device)
                feat = self.clip_image_encoder(**inputs).image_embeds
            return F.normalize(feat, dim=-1)

    def compute_dino_similarity(self, img1: Image.Image, img2: Image.Image):
        with torch.no_grad():
            inputs = self.dino_processor(images=[img1, img2], return_tensors="pt").to(self.device)
            outputs = self.dino_model(**inputs)
            cls_feats = outputs.last_hidden_state[:, 0, :]
            return F.cosine_similarity(cls_feats[0:1], cls_feats[1:2]).item()

    def compute_lpips(self, img1: Image.Image, img2: Image.Image):
        if img1 is None or img2 is None:
            return None

        def pil_to_tensor(img):
            arr = np.array(img.resize((224, 224))).astype(np.float32) / 255.0
            arr = arr * 2 - 1  # [0,1] -> [-1,1]
            return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            return self.lpips_model(pil_to_tensor(img1), pil_to_tensor(img2)).item()

    def compute_psnr(self, img1: Image.Image, img2: Image.Image):
        if img1 is None or img2 is None:
            return None
        arr1 = np.array(img1).astype(np.float32)
        arr2 = np.array(img2).astype(np.float32)
        return psnr(arr1, arr2, data_range=255)

    def compute_clipdir(self, before_img, edited_img, original_prompt, edited_prompt):
        """计算CLIPdir指标"""
        # 检查必要的输入是否存在
        if (
            before_img is None
            or edited_img is None
            or original_prompt is None
            or edited_prompt is None
            or original_prompt.strip() == ""
            or edited_prompt.strip() == ""
        ):
            return None

        with torch.no_grad():
            I_in = self.compute_clip_similarity(img=before_img)
            I_out = self.compute_clip_similarity(img=edited_img)
            T_in = self.compute_clip_similarity(text=original_prompt)
            T_out = self.compute_clip_similarity(text=edited_prompt)

            dir_img = F.normalize(I_out - I_in, dim=-1)
            dir_txt = F.normalize(T_out - T_in, dim=-1)
            return F.cosine_similarity(dir_img, dir_txt).item()

    def compute_metrics(
        self,
        before_img,
        edited_img,
        after_img,
        edit_prompt,
        edited_prompt,
        original_prompt=None,
    ):
        """计算所有评估指标"""
        # 检查基本输入
        if before_img is None or edited_img is None:
            return {k: None for k in self.metrics.keys()}

        # CLIP指标
        before_feat = self.compute_clip_similarity(img=before_img)
        edited_feat = self.compute_clip_similarity(img=edited_img)

        clipim = F.cosine_similarity(edited_feat, before_feat).item()

        # CLIPout需要检查edit_prompt是否存在
        clipout = None
        text_feat = self.compute_clip_similarity(text=edit_prompt)
        clipout = F.cosine_similarity(edited_feat, text_feat).item()

        # CLIPdir需要检查原始和编辑后的文本描述
        clipdir = self.compute_clipdir(before_img, edited_img, original_prompt, edited_prompt)

        # 其他指标
        dino_sim = self.compute_dino_similarity(before_img, edited_img)

        # 图像质量指标
        def to_array(img):
            return np.array(img).astype(np.float32) / 255.0

        before_np = to_array(before_img)
        edited_np = to_array(edited_img)

        ssim_val = ssim(before_np, edited_np, multichannel=True, data_range=1.0, channel_axis=2)
        l1_before = np.mean(np.abs(edited_np - before_np))
        l2_before = np.mean((edited_np - before_np) ** 2)
        lpips_before = self.compute_lpips(before_img, edited_img)
        psnr_before = self.compute_psnr(before_img, edited_img)

        # 与目标图像的对比
        l1_after = l2_after = lpips_after = psnr_after = clipafter = None
        if after_img is not None:
            after_np = to_array(after_img)
            l1_after = np.mean(np.abs(edited_np - after_np))
            l2_after = np.mean((edited_np - after_np) ** 2)
            lpips_after = self.compute_lpips(after_img, edited_img)
            psnr_after = self.compute_psnr(after_img, edited_img)
            # 计算与真实灾后图像的CLIP一致性
            after_feat = self.compute_clip_similarity(img=after_img)
            clipafter = F.cosine_similarity(edited_feat, after_feat).item()

        return {
            "CLIPim": clipim,
            "CLIPout": clipout,
            "CLIPdir": clipdir,
            "CLIPafter": clipafter,
            "DINO": dino_sim,
            "SSIM": ssim_val,
            "LPIPS_before": lpips_before,
            "LPIPS_after": lpips_after,
            "PSNR_before": psnr_before,
            "PSNR_after": psnr_after,
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
        default="/mnt/data/zwh/log/instruct-pix2pix/experiment_2",
    )
    parser.add_argument("--dataset_path", type=str, default="/mnt/data/zwh/data/maxar/disaster_dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/data/zwh/log/instruct-pix2pix/Experiments/experiment_2",
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--image_guidance_scale", type=float, default=1.5)
    parser.add_argument("--guidance_scale", type=float, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--use_fixed_edit_text", action="store_true")

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 初始化
    accelerator = Accelerator(project_dir=args.output_dir)

    # 加载模型和数据
    logger.info(f"Loading pipeline from {args.pretrained_model_name_or_path}")
    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )

    if hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
        pipeline.enable_xformers_memory_efficient_attention()

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

    # 创建数据集
    with accelerator.main_process_first():
        eval_dataset = EvaluationDataset(
            dataset_path=args.dataset_path,
            tokenizer=tokenizer,
            resolution=args.resolution,
            split="test",
            seed=args.seed,
            use_fixed_edit_text=args.use_fixed_edit_text,
        )

    dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    dataloader = accelerator.prepare(dataloader)

    # 运行评估
    evaluator = ImageEditingEvaluator(pipeline, dataloader, args, accelerator.device, accelerator)
    metrics = evaluator.run()

    # 保存结果
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        import json

        summary = {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}
        summary["num_samples"] = len(evaluator.results)

        with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {os.path.join(args.output_dir, 'metrics.json')}")
        if args.save_images:
            logger.info(f"Images saved to {os.path.join(args.output_dir, 'comparison_images')}")

    logger.info("✅ Evaluation completed.")


if __name__ == "__main__":
    main()
