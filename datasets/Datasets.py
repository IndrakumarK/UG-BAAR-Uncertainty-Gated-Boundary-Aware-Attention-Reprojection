# datasets/dataset.py
from torch.utils.data import Dataset
import torch


class SegmentationDataset(Dataset):
    def __init__(self, images, masks):
        assert len(images) == len(masks), "Images and masks length mismatch"
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx].float()
        y = self.masks[idx].long()
        return x, y
