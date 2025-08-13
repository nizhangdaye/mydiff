#!/usr/bin/env python
# coding=utf-8
"""ImageEditDataset class for training."""

# import os

# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_DATASETS_OFFLINE"] = "1"

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

        # Setup image transforms
        self._setup_transforms()

    def _load_dataset(self):
        """Load the dataset from disk."""
        print(f"Loading dataset from {self.dataset_path}")
        full_dataset = load_from_disk(self.dataset_path)

        # Select the appropriate split
        if self.split not in full_dataset:
            raise ValueError(
                f"Split '{self.split}' not found in dataset. "
                f"Available splits: {list(full_dataset.keys())}"
            )

        self.dataset = full_dataset[self.split]

        # Apply max_samples limit if specified
        if self.max_samples is not None:
            if self.seed is not None:
                self.dataset = self.dataset.shuffle(seed=self.seed)
            self.dataset = self.dataset.select(
                range(min(self.max_samples, len(self.dataset)))
            )

        print(f"Loaded {len(self.dataset)} samples from {self.split} split")

    def _setup_transforms(self):
        """Setup image transforms based on configuration."""
        transform_list = []

        if self.center_crop:
            transform_list.append(transforms.CenterCrop(self.resolution))
        else:
            transform_list.append(transforms.RandomCrop(self.resolution))

        if self.random_flip:
            transform_list.append(transforms.RandomHorizontalFlip())
        else:
            transform_list.append(transforms.Lambda(lambda x: x))

        self.train_transforms = transforms.Compose(transform_list)

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
        self, before_images: List[PIL.Image.Image], after_images: List[PIL.Image.Image]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess before and after images.

        Args:
            before_images: List of original images
            after_images: List of edited images

        Returns:
            Tuple of (original_images_tensor, edited_images_tensor)
        """
        # Convert images to numpy arrays
        original_images = np.concatenate(
            [convert_to_np(image, self.resolution) for image in before_images]
        )
        edited_images = np.concatenate(
            [convert_to_np(image, self.resolution) for image in after_images]
        )

        # Stack images for synchronized transforms
        images = np.stack([original_images, edited_images])
        images = torch.tensor(images)

        # Normalize to [-1, 1] range
        images = 2 * (images / 255) - 1

        # Apply transforms (both images will undergo the same transform)
        transformed_images = self.train_transforms(images)

        # Separate the images
        original_images, edited_images = transformed_images
        original_images = original_images.reshape(
            -1, 3, self.resolution, self.resolution
        )
        edited_images = edited_images.reshape(-1, 3, self.resolution, self.resolution)

        return original_images, edited_images

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
        after_image = sample["after"]

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

        # Preprocess images
        original_pixel_values, edited_pixel_values = self.preprocess_images(
            [before_image], [after_image]
        )

        # Tokenize the edit instruction
        input_ids = self.tokenize_captions([edit_instruction])

        return {
            "original_pixel_values": original_pixel_values.squeeze(
                0
            ),  # Remove batch dimension
            "edited_pixel_values": edited_pixel_values.squeeze(
                0
            ),  # Remove batch dimension
            "input_ids": input_ids.squeeze(0),  # Remove batch dimension
        }


def collate_fn(examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader.

    Args:
        examples: List of samples from the dataset

    Returns:
        Batched data dictionary
    """
    original_pixel_values = torch.stack(
        [example["original_pixel_values"] for example in examples]
    )
    original_pixel_values = original_pixel_values.to(
        memory_format=torch.contiguous_format
    ).float()

    edited_pixel_values = torch.stack(
        [example["edited_pixel_values"] for example in examples]
    )
    edited_pixel_values = edited_pixel_values.to(
        memory_format=torch.contiguous_format
    ).float()

    input_ids = torch.stack([example["input_ids"] for example in examples])

    return {
        "original_pixel_values": original_pixel_values,
        "edited_pixel_values": edited_pixel_values,
        "input_ids": input_ids,
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

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

    # 4. 遍历示例
    for batch in train_loader:
        print("Original pixel values shape:", batch["original_pixel_values"].shape)
        print("Edited pixel values shape:", batch["edited_pixel_values"].shape)
        print("Input IDs shape:", batch["input_ids"].shape)
        break
