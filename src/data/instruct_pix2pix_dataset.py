#!/usr/bin/env python
# coding=utf-8
"""ImageEditDataset class for training."""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import PIL
import torch
from datasets import Dataset, load_from_disk
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms
from transformers import CLIPTokenizer


def convert_to_np(image: PIL.Image.Image, resolution: int) -> np.ndarray:
    """Convert PIL image to numpy array with specified resolution.

    Args:
        image: PIL Image to convert
        resolution: Target resolution for the image

    Returns:
        Numpy array with shape (3, resolution, resolution)
    """
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)


class ImageEditDataset(TorchDataset):
    """Dataset class for image editing training.

    This dataset handles the preprocessing of images and text for training
    image editing models. It supports both HuggingFace datasets and
    custom data formats.
    """

    def __init__(
        self,
        dataset_path: str,
        tokenizer: CLIPTokenizer,
        resolution: int = 256,
        split: str = "train",
        center_crop: bool = False,
        random_flip: bool = False,
        max_samples: Optional[int] = None,
        seed: Optional[int] = None,
        use_fixed_edit_text: bool = False,
    ):
        """Initialize the ImageEditDataset.

        Args:
            dataset_path: Path to the dataset (HuggingFace datasets format)
            tokenizer: CLIP tokenizer for text encoding
            resolution: Image resolution for training
            split: Dataset split to use ("train", "test", "validation")
            center_crop: Whether to center crop images
            random_flip: Whether to apply random horizontal flip
            max_samples: Maximum number of samples to use (for debugging)
            seed: Random seed for reproducibility
            use_fixed_edit_text: Whether to use fixed edit text based on disaster type
        """
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.split = split
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.max_samples = max_samples
        self.seed = seed
        self.use_fixed_edit_text = use_fixed_edit_text

        # Set default fixed edit mapping if none provided
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

        # Load the dataset
        self._load_dataset()

        # # Setup image transforms
        # self._setup_transforms()

        # 新增：image_transforms 和 conditioning_image_transforms
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.conditioning_image_transforms = transforms.Compose(
            [
                transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.resolution),
                transforms.ToTensor(),
            ]
        )

    def _load_dataset(self):
        """Load the dataset from disk."""
        print(f"Loading dataset from {self.dataset_path}")
        full_dataset = load_from_disk(self.dataset_path)

        # Select the appropriate split
        if self.split not in full_dataset:
            raise ValueError(
                f"Split '{self.split}' not found in dataset. Available splits: {list(full_dataset.keys())}"
            )

        self.dataset = full_dataset[self.split]

        # Apply max_samples limit if specified
        if self.max_samples is not None:
            if self.seed is not None:
                self.dataset = self.dataset.shuffle(seed=self.seed)
            self.dataset = self.dataset.select(range(min(self.max_samples, len(self.dataset))))

        print(f"Loaded {len(self.dataset)} samples from {self.split} split")

    # # TODO: 添加新的数据处理逻辑
    # def _setup_transforms(self):
    #     """Setup image transforms based on configuration."""
    #     transform_list = []

    #     if self.center_crop:
    #         transform_list.append(transforms.CenterCrop(self.resolution))
    #     else:
    #         transform_list.append(transforms.RandomCrop(self.resolution))

    #     if self.random_flip:
    #         transform_list.append(transforms.RandomHorizontalFlip())
    #     # else:
    #     #     transform_list.append(transforms.Lambda(lambda x: x))

    #     self.train_transforms = transforms.Compose(transform_list)

    def tokenize_captions(self, captions: List[str]) -> torch.Tensor:
        """Tokenize text captions using the CLIP tokenizer.

        Args:
            captions: List of text captions to tokenize

        Returns:
            Tokenized input_ids tensor
        """
        inputs = self.tokenizer(
            captions,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids

    def preprocess_images(
        self,
        before_images: List[PIL.Image.Image],
        after_images: List[PIL.Image.Image],
        before_depth_images: List[PIL.Image.Image],
        before_seg_images: List[PIL.Image.Image],
        after_depth_images: List[PIL.Image.Image],
        after_seg_images: List[PIL.Image.Image],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Preprocess before and after images, depth and seg images.

        Args:
            before_images: List of original images
            after_images: List of edited images
            before_depth_images: List of depth images
            before_seg_images: List of seg images
            after_depth_images: List of after depth images
            after_seg_images: List of after seg images

        Returns:
            Tuple of (original_images_tensor, edited_images_tensor, before_depth_tensor, before_seg_tensor, after_depth_tensor, after_seg_tensor)
        """
        # 原图和编辑图像
        original_images = torch.stack([self.image_transforms(img) for img in before_images])
        edited_images = torch.stack([self.image_transforms(img) for img in after_images])
        # depth 和 seg 作为条件图像
        before_depth_images = torch.stack([self.conditioning_image_transforms(img) for img in before_depth_images])
        before_seg_images = torch.stack([self.conditioning_image_transforms(img) for img in before_seg_images])
        after_depth_images = torch.stack([self.conditioning_image_transforms(img) for img in after_depth_images])
        after_seg_images = torch.stack([self.conditioning_image_transforms(img) for img in after_seg_images])
        return (
            original_images,  # 数值范围 [-1, 1]
            edited_images,  # 数值范围 [-1, 1]
            before_depth_images,  # 数值范围 [0, 1]
            before_seg_images,  # 数值范围 [0, 1]
            after_depth_images,  # 数值范围 [0, 1]
            after_seg_images,  # 数值范围 [0, 1]
        )

    def _random_augment(self, images: List[PIL.Image.Image]) -> List[PIL.Image.Image]:
        """对一组图像做一致的随机上下翻转、左右翻转、90度倍数旋转."""

        # 随机决定是否上下翻转
        if random.random() < 0.5:
            images = [img.transpose(PIL.Image.FLIP_TOP_BOTTOM) for img in images]
        # 随机决定是否左右翻转
        if random.random() < 0.5:
            images = [img.transpose(PIL.Image.FLIP_LEFT_RIGHT) for img in images]
        # 随机旋转 0, 90, 180, 270 度
        k = random.randint(0, 3)
        if k > 0:
            images = [img.rotate(90 * k) for img in images]
        return images

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset.

        Args:
            idx: Index of the item to retrieve

        Returns:
            Dictionary containing processed data for training
        """
        sample = self.dataset[idx]

        # Get the images
        before_image = sample["before"]
        before_depth_image = sample["before_depth"]
        before_seg_image = sample["before_seg"]
        after_image = sample["after"]
        after_depth_image = sample["after_depth"]
        after_seg_image = sample["after_seg"]

        # Get edit instruction based on configuration
        if self.use_fixed_edit_text:
            # Use fixed edit text based on disaster type
            edit_instruction = self.fixed_edit_mapping[sample["disaster_type"]]
        else:
            # Use edit text from dataset
            edit_instruction = sample["edit"]

        # Ensure images are PIL Images
        if not isinstance(before_image, PIL.Image.Image):
            before_image = PIL.Image.fromarray(before_image)
        if not isinstance(after_image, PIL.Image.Image):
            after_image = PIL.Image.fromarray(after_image)
        if not isinstance(before_depth_image, PIL.Image.Image):
            before_depth_image = PIL.Image.fromarray(before_depth_image)
        if not isinstance(before_seg_image, PIL.Image.Image):
            before_seg_image = PIL.Image.fromarray(before_seg_image)
        if not isinstance(after_depth_image, PIL.Image.Image):
            after_depth_image = PIL.Image.fromarray(after_depth_image)
        if not isinstance(after_seg_image, PIL.Image.Image):
            after_seg_image = PIL.Image.fromarray(after_seg_image)

        # 如果是训练集，做一致的数据增强
        if self.split == "train":
            imgs = [before_image, after_image, before_depth_image, before_seg_image, after_depth_image, after_seg_image]
            before_image, after_image, before_depth_image, before_seg_image, after_depth_image, after_seg_image = (
                self._random_augment(imgs)
            )

        # Preprocess images
        (
            original_pixel_values,
            edited_pixel_values,
            before_depth_pixel_values,
            before_seg_pixel_values,
            after_depth_pixel_values,
            after_seg_pixel_values,
        ) = self.preprocess_images(
            [before_image],
            [after_image],
            [before_depth_image],
            [before_seg_image],
            [after_depth_image],
            [after_seg_image],
        )

        # Tokenize the edit instruction
        input_ids = self.tokenize_captions([edit_instruction])

        return {
            "original_pixel_values": original_pixel_values.squeeze(0),
            "edited_pixel_values": edited_pixel_values.squeeze(0),
            "input_ids": input_ids.squeeze(0),
            "before_depth_pixel_values": before_depth_pixel_values.squeeze(0),
            "before_seg_pixel_values": before_seg_pixel_values.squeeze(0),
            "after_depth_pixel_values": after_depth_pixel_values.squeeze(0),
            "after_seg_pixel_values": after_seg_pixel_values.squeeze(0),
        }


def collate_fn(examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader.

    Args:
        examples: List of samples from the dataset

    Returns:
        Batched data dictionary
    """
    original_pixel_values = torch.stack([example["original_pixel_values"] for example in examples])
    original_pixel_values = original_pixel_values.to(memory_format=torch.contiguous_format).float()

    edited_pixel_values = torch.stack([example["edited_pixel_values"] for example in examples])
    edited_pixel_values = edited_pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.stack([example["input_ids"] for example in examples])

    before_depth_pixel_values = torch.stack([example["before_depth_pixel_values"] for example in examples]).float()
    before_depth_pixel_values = before_depth_pixel_values.to(memory_format=torch.contiguous_format).float()

    before_seg_pixel_values = torch.stack([example["before_seg_pixel_values"] for example in examples]).float()
    before_seg_pixel_values = before_seg_pixel_values.to(memory_format=torch.contiguous_format).float()

    after_depth_pixel_values = torch.stack([example["after_depth_pixel_values"] for example in examples]).float()
    after_depth_pixel_values = after_depth_pixel_values.to(memory_format=torch.contiguous_format).float()

    after_seg_pixel_values = torch.stack([example["after_seg_pixel_values"] for example in examples]).float()
    after_seg_pixel_values = after_seg_pixel_values.to(memory_format=torch.contiguous_format).float()

    return {
        "original_pixel_values": original_pixel_values,
        "edited_pixel_values": edited_pixel_values,
        "input_ids": input_ids,
        "before_depth_pixel_values": before_depth_pixel_values,
        "before_seg_pixel_values": before_seg_pixel_values,
        "after_depth_pixel_values": after_depth_pixel_values,
        "after_seg_pixel_values": after_seg_pixel_values,
    }


if __name__ == "__main__":
    # 示例

    dataset_path = "/mnt/data/zwh/data/maxar/disaster_dataset"
    # 加载 clip tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        subfolder="tokenizer",
    )
    # 打印 tokenizer 配置
    print(tokenizer)

    train_dataset = ImageEditDataset(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        resolution=512,
        split="train",
        max_samples=10,
        seed=42,
        use_fixed_edit_text=True,
    )
    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)

    # 4. 遍历示例
    for batch in train_loader:
        print("Original pixel values shape:", batch["original_pixel_values"].shape)
        print("Edited pixel values shape:", batch["edited_pixel_values"].shape)
        print("Before depth pixel values shape:", batch["before_depth_pixel_values"].shape)
        print("Before seg pixel values shape:", batch["before_seg_pixel_values"].shape)
        print("After depth pixel values shape:", batch["after_depth_pixel_values"].shape)
        print("After seg pixel values shape:", batch["after_seg_pixel_values"].shape)
        print("Input IDs shape:", batch["input_ids"].shape)
        break
