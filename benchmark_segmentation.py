from bioio import BioImage
import bioio_tifffile
import einops
import fire
from loguru import logger
import numpy as np
import pandas as pd
import skimage
from tqdm import tqdm


def calc_pairwise_distances(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
    "Calculate pairwise distances between two sets of coordinates."
    coords1 = einops.repeat(coords1, "i1 d -> i1 i2 d", i2=coords2.shape[0])
    coords2 = einops.repeat(coords2, "i2 d -> i1 i2 d", i1=coords1.shape[0])
    square_distances = einops.reduce(
        np.square(coords1 - coords2), "i1 i2 d -> i1 i2", "sum"
    )
    return np.sqrt(square_distances)


def open_image(image_path: str, dim_str: str = "ZXY") -> np.ndarray:
    "Open an image and return it as a numpy array."
    reader = BioImage(image_path, reader=bioio_tifffile.Reader)
    image = reader.get_image_data("ZYX")
    image = einops.rearrange(image, f"Z Y X -> {' '.join(dim_str)}")
    return image


def get_labels_and_coords(image: np.ndarray) -> tuple:
    "Get labels and coordinates of the centroids of the regions in the image."
    regionprops = skimage.measure.regionprops_table(
        image, properties=("label", "centroid")
    )
    coords = np.array(
        [
            regionprops["centroid-0"],
            regionprops["centroid-1"],
            regionprops["centroid-2"],
        ]
    )
    labels = regionprops["label"]
    return labels, coords


def calc_iou(args) -> dict:
    "Calculate the intersection over union between two masks."
    seg, gt, seg_label, gt_label = args
    intersection = np.logical_and(seg, gt).sum()
    union = np.logical_or(seg, gt).sum()
    iou = intersection / union
    return {"seg_label": seg_label, "gt_label": gt_label, "iou": iou}


def seg_vs_gt_generator(segs, gts, seg_labels, gt_labels):
    for seg_label in seg_labels:
        for gt_label in gt_labels:
            seg = segs == seg_label
            gt = gts == gt_label
            yield seg, gt, seg_label, gt_label


def main(
    segmentation_path: str,
    ground_truth_path: str,
    output_path: str,
):
    """
    Computes IoU between segmentation and grouund truth masks and stores them in a csv.

    Args:
        segmentation_path (str): Path to the segmentation mask.
        ground_truth_path (str): Path to the ground truth mask.
        output_path (str): Path to the output csv file.
    """
    logger.info("Loading images...")
    segs = open_image(segmentation_path, "ZXY")
    gts = open_image(ground_truth_path, "ZXY")

    logger.info("Loading mask coordinates...")
    seg_labels, seg_coords = get_labels_and_coords(segs)
    gt_labels, gt_coords = get_labels_and_coords(gts)

    logger.info("Calculating pairwise distances...")

    results = []

    logger.info("Calculating IoU...")
    segs_iterator = seg_vs_gt_generator(segs, gts, seg_labels, gt_labels)

    for args_ in tqdm(segs_iterator, total=len(seg_labels) * len(gt_labels)):
        result = calc_iou(args_)
        results.append(result)

    df = pd.DataFrame(results)
    df = df.groupby("gt_label").apply(lambda x: x.loc[x["iou"].idxmax()])

    logger.info("Saving results...")
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    fire.Fire(main)
