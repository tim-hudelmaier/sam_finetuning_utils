from pathlib import Path
from bioio import BioImage
import einops
import fire
from loguru import logger
import numpy as np

from micro_sam.automatic_segmentation import (
    get_predictor_and_segmenter,
    automatic_instance_segmentation,
)
from micro_sam.training.util import normalize_to_8bit

from finetune import subset_img_to_channels
from adaptive_histogram_equalization import apply_clahe


def run_automatic_segmentation(
    image: np.ndarray,
    ndim: int,
    checkpoint: str | None,
    model_type: str,
    masks_out_path: str,
    device: str | None = None,
    embeddings_out_path: str | None = None,
):
    """
    Run automatic segmentation on a given 2d or 3d image.

    Args:
        image (np.ndarray): The 3D image data to be segmented.
        ndim (int): Number of dimensions of the image data.
        masks_out_path (str): Path to save the segmentation
        checkpoint (str): Path to the model checkpoint to be used for segmentation.
        model_type (str): Type of the model to be used for segmentation.
        device (str | None, optional): Device to run the segmentation on (e.g., 'cuda' or 'cpu').
            Defaults to None, which will auto-select the device.
        embeddings_out_path (str | None, optional): Path to save the embeddings to.
            Defaults to None. If None, embeddings will not be saved.
    """

    predictor, segmentor = get_predictor_and_segmenter(
        model_type=model_type, checkpoint=checkpoint, device=device
    )

    # NOTE: Input path also accepts numpy arrays
    instances = automatic_instance_segmentation(
        predictor=predictor,
        segmenter=segmentor,
        input_path=image,
        output_path=masks_out_path,
        embedding_path=embeddings_out_path,
        ndim=ndim,
    )

    return instances


def main(
    img_path: str,
    model_type: str,
    ndim: int,
    output_path: str,
    channels_path: str | None = None,
    checkpoint: str | None = None,
    clahe: bool = False,
    merge_channels: bool = False,
    merge_method: str = "mean",
    embeddings_out_path: str | None = None,
):
    """Loads an image and runs automatic 3D segmentation or 2D segmentation over the z layers.
    The segmentation results is saved as a TIFF file.

    Args:
        img_path (str): Path to the image to be segmented.
        checkpoint (str): Path to the model checkpoint to be used for segmentation.
        model_type (str): Type of the model to be used for segmentation.
        output_path (str): Path to save the segmentation instance to.
    """
    img_reader = BioImage(img_path)
    img = img_reader.get_image_data("CZYX")

    if channels_path is not None:
        with open(channels_path, "r") as f:
            channels_of_interest = f.read().splitlines()

        img = subset_img_to_channels(img, img_reader, channels_of_interest)

    if clahe:
        img = apply_clahe(img)

    img = normalize_to_8bit(img)

    # no more than 3 channels are currently supported, as SAM is trained on RBG data
    img = einops.rearrange(img, "c z y x -> z y x c")

    # repeating 3x, since segment anything expects 3 channels (rgb)
    # micro sam under the hood does the same when using 2 channels
    if merge_channels:
        logger.info("Merging channels...")
        img = einops.reduce(img, "z y x c -> z y x", merge_method)
        img = einops.repeat(img, "z y x -> z y x c", c=3)
    else:
        if img.shape[-1] == 1:
            img = einops.repeat(img, "z y x c -> z y x c", c=3)
        elif img.shape[-1] == 3:
            pass
        else:
            raise ValueError("Only 1 or 3 channels are currently supported.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if embeddings_out_path is not None:
        embeddings_out_path = Path(embeddings_out_path)
        embeddings_out_path.parent.mkdir(parents=True, exist_ok=True)

    if ndim == 3:
        logger.debug(f"Image shape: {img.shape}")
        _ = run_automatic_segmentation(
            image=img,
            ndim=ndim,
            checkpoint=checkpoint,
            model_type=model_type,
            masks_out_path=output_path,
            embeddings_out_path=embeddings_out_path,
        )
    elif ndim == 2:
        for z in range(img.shape[0]):
            logger.debug(f"Processing slice {z}...")

            if embeddings_out_path is not None:
                layer_embeddings_out_path = embeddings_out_path / f"layer_{z}"
                layer_embeddings_out_path.mkdir(parents=True, exist_ok=True)
            else:
                layer_embeddings_out_path = None

            _ = run_automatic_segmentation(
                image=img[z],
                ndim=ndim,
                checkpoint=checkpoint,
                model_type=model_type,
                masks_out_path=output_path.parent
                / output_path.name.replace(".tif", f"_{z}.tif"),
                embeddings_out_path=layer_embeddings_out_path,
            )


if __name__ == "__main__":
    fire.Fire(main)
