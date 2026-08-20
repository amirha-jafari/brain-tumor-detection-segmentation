import segmentation_models_pytorch as smp


def build_unet(encoder_name="resnet34", in_channels=3, out_channels=1):
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=out_channels,
        activation=None,
    )
    return model