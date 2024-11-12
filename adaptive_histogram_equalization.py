from bioio import BioImage
import einops
from loguru import logger
import fire
import numpy as np
from skimage import exposure
from tqdm import tqdm


def apply_clahe(image: np.ndarray, clip_limit: float = 0.9) -> np.ndarray:
    """Apply CLAHE to the image."""

    # input_img: [C, Z, Y, X]
    out_img = np.zeros_like(image)

    logger.info("Applying CLAHE to the image")
    for channel in tqdm(range(image.shape[0])):
        # reorder axis from (Z, Y, X ) to (X, Y, Z)
        img = einops.rearrange(image[channel], "Z Y X -> X Y Z")

        # Rescale image data to range [0, 1]
        img = np.clip(img, np.percentile(img, 5), np.percentile(img, 95))
        img = (img - img.min()) / (img.max() - img.min())

        # Run CLAHE
        img = exposure.equalize_adapthist(img, clip_limit=clip_limit)
        out_img[channel] = einops.rearrange(img, "X Y Z -> Z Y X")

    return out_img


def main(img: str | np.ndarray, out_path: str | None):
    """Load img, if needed and apply CLAHE and optionally save to out_path."""
    if isinstance(img, str):
        if img.endswith(".tiff") or img.endswith(".tif"):
            img_reader = BioImage(img)
            img = img_reader.get_image_data("CZYX")
        elif img.endswith(".npy"):
            raise NotImplementedError(
                "Loading .npy files is not supported yet, please "
                "use tiff or provide image as array to function directly."
            )
        else:
            raise ValueError("Image should be a path to a .tiff file")

    if isinstance(img, np.ndarray):
        out_img = apply_clahe(img)

    if out_path is not None:
        logger.info(f"Saving CLAHE image to {out_path}")
        if out_path.endswith(".tif") or out_path.endswith(".tiff"):
            out_reader = BioImage(img)
            out_reader.save(out_path)
        if out_path.endswith(".npy"):
            np.save(out_path, out_img)
    else:
        return out_img


if __name__ == "__main__":
    fire.Fire(main)
