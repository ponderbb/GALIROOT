import albumentations as A

from src import utils


def generate_transform(config_path: str) -> A.Compose:
    config = utils.load_config(config_path)
    transform = []

    if config["horizontal_flip"]:
        transform.append(A.HorizontalFlip())

    if config["vertical_flip"]:
        transform.append(A.VerticalFlip())

    if config["shift_scale_rotate"]["apply"]:
        transform.append(
            A.ShiftScaleRotate(
                scale_limit=config["shift_scale_rotate"]["scale_limit"],
                rotate_limit=config["shift_scale_rotate"]["rotate_limit"],
                p=config["shift_scale_rotate"]["p"],
            )
        )

    if config["color_jitter"]:
        transform.append(A.ColorJitter())

    if config["channel_dropout"]:
        transform.append(A.ChannelDropout())

    if config["gaussian_blur"]:
        transform.append(A.GaussianBlur())

    return A.Compose(transform)
