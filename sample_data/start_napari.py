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

    # FIXME: img channel names are wrong so setting correct ones here
    channel_names = [
        "DAPI (dsDNA bound) Gating",
        "TOMM20_ALEXA_594_Gating",
        "alphaTUBULIN_ATTO_425_Gating",
        "SC35_ATTO_430LS_Gating",
        "SP100_ATTO_633_Gating",
        "WGA_CF770_Gating",
        "SON_ATTO_488_Gating",
        "VIMENTIN_ATTO_490LS_Gating",
        "LAMP1_Oregon_Green_514_Gating",
        "COILIN_ALEXA_532_Gating",
        "GM130_ALEXA_647_Gating",
        "G3BP1_ATTO_550_Gating",
        "TFAM_TYE705_Gating",
        "Ki67_Atto_Rho11_Gating",
        "NPM1_ALEXA_750_Gating",
    ]

    img_reader = BioImage(img_path)
    img = img_reader.get_image_data("CZYX")
    viewer = napari.view_image(img, channel_axis=0, name=channel_names)
    napari.run()


if __name__ == "__main__":
    fire.Fire(open_napari)
