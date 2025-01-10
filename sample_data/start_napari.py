from bioio import BioImage
import fire
import napari


def open_napari(
    img_path: str = "sample_data/leica_data_unperturbed_crop.tiff",
):
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
    napari.run()


if __name__ == "__main__":
    fire.Fire(open_napari)
