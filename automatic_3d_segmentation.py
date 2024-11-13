from bioio import BioImage
import einops
import fire
from loguru import logger
import numpy as np

from micro_sam.automatic_segmentation import get_predictor_and_segmenter
from micro_sam.multi_dimensional_segmentation import automatic_3d_segmentation
from micro_sam.training.util import normalize_to_8bit

from finetune import subset_img_to_channels
from adaptive_histogram_equalization import apply_clahe


def run_automatic_3d_segmentation(
    image: np.ndarray,
    checkpoint: str,
    model_type: str,
    device: str | None = None,
    embeddings_out_path: str | None = None,
):
    """
    Run automatic 3D segmentation on a given image.

    Args:
        image (np.ndarray): The 3D image data to be segmented.
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

    logger.info("Running automatic 3D segmentation")
    instances = automatic_3d_segmentation(
        volume=image,
        predictor=predictor,
        segmentor=segmentor,
        embedding_path=embeddings_out_path,
    )

    return instances


def save_segmentation_instance_to_tiff(instance: np.ndarray, output_path: str):
    """
    Save the segmentation instance to a TIFF file.
    Args:
        instance (np.ndarray): The segmentation instance to be saved.
        output_path (str): Path to save the segmentation instance to.
    """
    assert instance.ndim == 3, "Instance must be 3D"

    logger.info(f"Saving segmentation instance to {output_path}")
    img_reader = BioImage(image=instance)
    img_reader.save(output_path)


def main(
    img_path: str,
    checkpoint: str,
    model_type: str,
    output_path: str,
    channels_path: str,
    clahe: bool = False,
    embeddings_out_path: str | None = None,
):
    """Loads an image and runs automatic 3D segmentation on it.
    The segmentation results is saved as a TIFF file.

    Args:
        img_path (str): Path to the image to be segmented.
        checkpoint (str): Path to the model checkpoint to be used for segmentation.
        model_type (str): Type of the model to be used for segmentation.
        output_path (str): Path to save the segmentation instance to.
    """
    img_reader = BioImage(img_path)
    img = img_reader.get_image_data("CZYX")

    with open(channels_path, "r") as f:
        channels_of_interest = f.read().splitlines()

    img = subset_img_to_channels(img, img_reader, channels_of_interest)

    if clahe:
        img = apply_clahe(img)

    img = normalize_to_8bit(img)

    # no more than 3 channels are currently supported, as SAM is trained on RBG data
    img = einops.rearrange(img, "c z y x -> z y x c", c=3)
    logger.debug(f"Image shape: {img.shape}")
    instances = run_automatic_3d_segmentation(
        image=img,
        checkpoint=checkpoint,
        model_type=model_type,
        embeddings_out_path=embeddings_out_path,
    )

    save_segmentation_instance_to_tiff(instances, output_path)


if __name__ == "__main__":
    fire.Fire(main)
