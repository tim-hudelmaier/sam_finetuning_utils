from bioio import BioImage
import fire
import napari


def open_napari(img_path: str, segmentation_path: str | None = None):
    """
    Open an image in Napari.
    Args:
        img_path (str): Path to the image to be opened.
        segmentation_path (str | None, optional): Path to the segmentation to be overlaid on the image. Defaults to None.
    """
    viewer = napari.Viewer()

    img_reader = BioImage(img_path)
    img = img_reader.get_image_data("CZYX")
    viewer = napari.view_image(img, channel_axis=0, name=img_reader.channel_names)

    if segmentation_path is not None:
        seg_reader = BioImage(segmentation_path)
        seg = seg_reader.get_image_data("ZYX")
        viewer.add_labels(seg, name="Segmentation")

    napari.run()


if __name__ == "__main__":
    fire.Fire(open_napari)
