import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


class ShapesDataset(torch.utils.data.Dataset):
    def __init__(self, root_path="../data/shapes_dataset", transform=None):
        super().__init__()

        self.root_path = Path(root_path)
        self.transform = transform
        meta_data_path = self.root_path / "metadata.json"

        meta_data = json.load(open(meta_data_path))
        file_paths = np.array([Path(item["path"]) for item in meta_data])
        constraints = np.array([int(item["constraint"]) for item in meta_data])

        rng = np.random.default_rng()
        random_indices = np.arange(len(file_paths))
        rng.shuffle(random_indices)

        self.file_paths = file_paths[random_indices]
        self.constraints = constraints[random_indices]


    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        mask = torch.from_numpy(np.array(Image.open(self.root_path / self.file_paths[idx]))) / 255.0

        if self.transform:
            mask = self.transform(mask)

        return mask, 0, self.constraints[idx], idx