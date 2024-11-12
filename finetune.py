from dataclasses import dataclass
import itertools
import json
from natsort import natsorted
from pathlib import Path
import random
from typing import List

from bioio import BioImage
import einops
import fire
import imageio.v3 as imageio
from loguru import logger
import numpy as np
import torch

import micro_sam.training as sam_training
from micro_sam.training.util import normalize_to_8bit


def make_consecutive_labels(labels: np.ndarray) -> np.ndarray:
    """Make consecutive labels."""
    unique_labels = np.unique(labels)
    new_labels = np.zeros_like(labels)
    for i, label in enumerate(unique_labels):
        new_labels[labels == label] = i
    return new_labels


def crop_bottom_z(img: np.ndarray, axis_str: str, crop_size: int) -> np.ndarray:
    """Crop the bottom of the image.

    Args:
        img: The image to crop.
        axis_str: Specify order of axis like: CZXY, ZCYX, etc.
        crop_size: The size to crop.
    """
    assert len(img.shape) == len(
        axis_str
    ), "The image and axis_str should have the same length."
    original_channels = " ".join(axis_str)
    cropped_channels = original_channels.replace("Z ", "") + " Z"

    cropped_img = einops.rearrange(img, f"{original_channels} -> {cropped_channels}")
    cropped_img = cropped_img[..., (cropped_img.shape[-1] - crop_size) :]
    cropped_img = einops.rearrange(
        cropped_img, f"{cropped_channels} -> {original_channels}", Z=crop_size
    )
    return cropped_img


def split_img(img, n_cuts=2, axis_str="CZYX"):
    """Split the image into n_cuts x n_cuts parts."""
    assert len(img.shape) == len(
        axis_str
    ), "The image and axis_str should have the same length."
    x_idx = axis_str.index("X")
    y_idx = axis_str.index("Y")

    x_dim, y_dim = img.shape[x_idx], img.shape[y_idx]
    x_split, y_split = x_dim // n_cuts, y_dim // n_cuts

    img_splits = []
    for x, y in itertools.product(range(n_cuts), range(n_cuts)):
        x_start, x_end = x * x_split, (x + 1) * x_split
        y_start, y_end = y * y_split, (y + 1) * y_split
        img_splits.append(img[:, :, x_start:x_end, y_start:y_end])
    return img_splits


def subset_img_to_channels(
    img: np.ndarray, reader: BioImage, channels: list[str]
) -> np.ndarray:
    """Subset the image to the channels of interest."""
    channel_indices = [reader.channel_names.index(c) for c in channels]
    return img[channel_indices, ...]


def preprocess_img_data(
    img_dir: str | Path,
    label_dir: str | Path,
    output_dir: str | Path,
    channels_of_interest: list[str],
) -> tuple[list[str], list[str], list[str], list[str]]:
    img_dir = Path(img_dir)
    label_dir = Path(label_dir)
    img_files = list(img_dir.glob("*.aivia.tif"))
    label_files = list(label_dir.glob("*.tif"))
    img_files, label_files = natsorted(img_files), natsorted(label_files)

    train_dir = Path(output_dir) / "train"
    val_dir = Path(output_dir) / "val"

    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)

    val_imgs = []
    train_imgs = []

    for img_file, label_file in zip(img_files, label_files):
        logger.info(f"Processing {img_file} and {label_file}")

        # load label and image data
        label_reader = BioImage(label_file)
        label = label_reader.get_image_data("CZYX")
        img_reader = BioImage(img_file)
        img = img_reader.get_image_data("CZYX")

        # crop label frames
        label = crop_bottom_z(label, "CZYX", img.shape[1])

        # select image channels
        img = subset_img_to_channels(img, img_reader, channels_of_interest)

        # split train & validaton (split array in 4 parts and use the first 3 for training)
        img_splits = split_img(img)
        label_splits = split_img(label)

        # make labels consecutive
        label_splits = [make_consecutive_labels(label) for label in label_splits]

        # select test_idx for validation
        test_idx = random.choice(range(len(img_splits)))

        val_img, val_label = img_splits[test_idx], label_splits[test_idx]
        val_imgs.append((val_img, val_label))

        train_imgs.extend(
            [
                (img, label)
                for i, (img, label) in enumerate(zip(img_splits, label_splits))
                if i != test_idx
            ]
        )

    # save images
    logger.info("Saving images")
    for i, (img, label) in enumerate(train_imgs):
        img = einops.rearrange(img, "C Z Y X -> Y X C Z")
        label = einops.rearrange(label, "C Z Y X -> Y X C Z")

        for j in range(img.shape[-1]):
            imageio.imwrite(train_dir / f"img_{i}_{j}.tif", img[..., j])
            imageio.imwrite(train_dir / f"label_{i}_{j}.tif", label[..., 0, j])

    for i, (img, label) in enumerate(val_imgs):
        img = einops.rearrange(img, "C Z Y X -> Y X C Z")
        label = einops.rearrange(label, "C Z Y X -> Y X C Z")

        for j in range(img.shape[-1]):
            imageio.imwrite(val_dir / f"img_{i}_{j}.tif", img[..., j])
            imageio.imwrite(val_dir / f"label_{i}_{j}.tif", label[..., 0, j])

    train_image_paths = natsorted([str(p) for p in train_dir.glob("img_*.tif")])
    train_label_paths = natsorted([str(p) for p in train_dir.glob("label_*.tif")])
    val_image_paths = natsorted([str(p) for p in val_dir.glob("img_*.tif")])
    val_label_paths = natsorted([str(p) for p in val_dir.glob("label_*.tif")])
    return train_image_paths, train_label_paths, val_image_paths, val_label_paths


@dataclass
class SAMFinetuneConfig:
    batch_size: int
    patch_shape: tuple[int, int]
    train_instance_segmentation: bool
    n_samples: int
    n_objects_per_batch: int
    n_epochs: int
    model_type: str
    checkpoint_name: str
    channels_of_interest: list[str]

    @classmethod
    def from_json(cls, json_path: str | Path):
        with open(json_path, "r") as f:
            config = json.load(f)
        if config.get("patch_shape") is not None:
            config["patch_shape"] = tuple(config["patch_shape"])
        return cls(**config)


def prep_data_loaders(
    train_image_paths: List[str],
    train_label_paths: List[str],
    val_image_paths: List[str],
    val_label_paths: List[str],
    config: SAMFinetuneConfig,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    logger.info("Preparing data loaders")
    train_loader = sam_training.default_sam_loader(
        raw_paths=train_image_paths,
        raw_key=None,
        label_paths=train_label_paths,
        label_key=None,
        is_seg_dataset=False,
        patch_shape=config.patch_shape,
        with_channels=True,
        with_segmentation_decoder=config.train_instance_segmentation,
        batch_size=config.batch_size,
        shuffle=True,
        raw_transform=normalize_to_8bit,
        n_samples=config.n_samples,
    )

    val_loader = sam_training.default_sam_loader(
        raw_paths=val_image_paths,
        raw_key=None,
        label_paths=val_label_paths,
        label_key=None,
        is_seg_dataset=False,
        patch_shape=config.patch_shape,
        with_channels=True,
        with_segmentation_decoder=config.train_instance_segmentation,
        batch_size=config.batch_size,
        raw_transform=normalize_to_8bit,
        shuffle=True,
    )

    return train_loader, val_loader


def main(
    config_path: str | Path,
    img_dir: str | Path,
    label_dir: str | Path,
    output_dir: str | Path,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    config = SAMFinetuneConfig.from_json(config_path)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    train_image_paths, train_label_paths, val_image_paths, val_label_paths = (
        preprocess_img_data(
            img_dir,
            label_dir,
            output_dir,
            config.channels_of_interest,
        )
    )
    train_loader, val_loader = prep_data_loaders(
        train_image_paths, train_label_paths, val_image_paths, val_label_paths, config
    )

    logger.info("Starting training")
    sam_training.train_sam(
        name=config.checkpoint_name,
        save_root=Path(output_dir) / "models",
        model_type=config.model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=config.n_epochs,
        n_objects_per_batch=config.n_objects_per_batch,
        with_segmentation_decoder=config.train_instance_segmentation,
        device=device,
    )


if __name__ == "__main__":
    fire.Fire(main)
