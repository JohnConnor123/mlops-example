# flake8: noqa: F401
import os

import albumentations as A
import albumentations.augmentations.functional as F
import numpy as np
from albumentations.pytorch import ToTensorV2
from datasets import SegmentationDataset
from skimage.io import imread
from skimage.transform import resize
from torch.utils.data import DataLoader

URL = "https://www.dropbox.com/s/8lqrloi0mxj2acu/PH2Dataset.rar"


def download_dataset(url):
    os.system(f"wget -P mlflow\\Files -c {URL}")
    if not os.path.exists(".\\mlflow\\Files\\PH2Dataset"):
        os.system(
            'c:\\"Program Files"\\WinRAR\\unrar.exe x '
            ".\\mlflow\\Files\\PH2Dataset.rar .\\mlflow\\Files\\"
        )

    images = []
    lesions = []
    root_dir = "mlflow\\Files\\PH2Dataset"

    for root, _dirs, files in os.walk(os.path.join(root_dir, "PH2 Dataset images")):
        if root.endswith("_Dermoscopic_Image"):
            images.append(imread(os.path.join(root, files[0])))
        if root.endswith("_lesion"):
            lesions.append(imread(os.path.join(root, files[0])))

    X = [resize(x, (256, 256), mode="constant", anti_aliasing=True) for x in images]
    Y = [
        resize(y, (256, 256), mode="constant", anti_aliasing=False) > 0.5
        for y in lesions
    ]

    X = np.array(X, np.float32)
    Y = np.array(Y, np.float32)

    print(f"Loaded {len(X)} images")
    print(f"Loaded {len(lesions)} lesions")
    return X, Y


def dataset_augmentations(X, Y):
    batch_size = 8

    # Ожидает на вход (H W C). Получая C перед HW, выдает ошибку.
    # imshow/plot тоже ожидают (H W C), а вот тензоры в pytorch уже ожидают (C H W)
    train_transforms = A.Compose(
        [
            # A.RandomCrop(p=0.25, width=220, height=220),
            A.HorizontalFlip(p=0.65),
            A.RandomBrightnessContrast(
                p=0.65, brightness_limit=(-0.05, 0.05), contrast_limit=(-0.05, 0.05)
            ),
            A.RandomRotate90(p=0.65),
            # A.MaskDropout(p=0.25),
            A.MultiplicativeNoise(p=0.25, multiplier=(0.95, 1.05)),
            # A.Resize(height=256, width=256),
            A.OneOf(
                [
                    A.NoOp(),
                    A.MultiplicativeNoise(multiplier=(0.97, 1.03)),
                    A.GaussNoise(var_limit=0.05),
                    # A.ISONoise(intensity=(0.05, 0.1))
                ]
            ),
            A.OneOf(
                [
                    A.NoOp(p=0.8),
                    A.HueSaturationValue(
                        hue_shift_limit=0.1, sat_shift_limit=0.1, val_shift_limit=0.1
                    ),
                    A.RGBShift(r_shift_limit=0.1, g_shift_limit=0.1, b_shift_limit=0.1),
                ]
            ),
        ]
    )

    val_transforms = A.Compose([])

    train_dataset = SegmentationDataset(
        images=X[tr], masks=np.expand_dims(Y[tr], axis=3), transforms=train_transforms
    )

    val_dataset = SegmentationDataset(
        images=X[val], masks=np.expand_dims(Y[val], axis=3), transforms=val_transforms
    )

    test_dataset = SegmentationDataset(
        images=X[ts], masks=np.expand_dims(Y[ts], axis=3), transforms=val_transforms
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":
    X, Y = download_dataset(URL)

    ix = np.random.choice(len(X), len(X), False)
    tr, val, ts = np.split(ix, [100, 150])
    print("train:", len(tr), "\nval:", len(val), "\ntest:", len(ts))

    train_dataloader, val_dataloader, test_dataloader = dataset_augmentations(X, Y)
    print(
        "train:", train_dataloader, "\nval:", val_dataloader, "\ntest:", test_dataloader
    )
