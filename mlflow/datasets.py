import torch
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(self, images, masks, transforms=None):
        self.images = images
        self.masks = masks
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        # print("BEFORE image.shape", image.dtype, image.shape)
        # print("BEFORE mask.shape", mask.dtype, mask.shape)

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask.squeeze())
            image = torch.tensor(transformed["image"]).permute(2, 0, 1)
            mask = torch.tensor(transformed["mask"]).unsqueeze(2).permute(2, 0, 1)
        # print("AFTER image.shape", image.dtype, image.shape)
        # print("AFTER mask.shape", mask.dtype, mask.shape)

        return image, mask
