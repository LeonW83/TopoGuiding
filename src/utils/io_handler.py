import os
import json
import glob
import torch
import numpy as np
from PIL import Image

from src.utils.visualization import show_image


def save_masks_to_disk(batch, labels, path="../results/shapes/", append_to_existing=False):
    offset=0
    if os.path.exists(path):
        if not append_to_existing:
            files = glob.glob(os.path.join(path, "masks/*"), recursive=True)
            for f in files:
                os.remove(f)
        else:
            files = glob.glob(os.path.join(path, "masks/*"))

            numbers = [int(os.path.splitext(os.path.basename(f))[0]) for f in files]
            offset = max(numbers) + 1
            if os.path.exists(os.path.join(path, "labels.pt")):
                old_labels = torch.load(os.path.join(path, "labels.pt"))
                labels = torch.cat((old_labels, labels.cpu()), 0)

    os.makedirs(os.path.join(path, "masks"), exist_ok=True)

    batch = batch / 2 + 0.5
    batch = batch.clamp(0, 1)
    batch = batch.permute(0, 2, 3, 1) * 255

    for idx, img in enumerate(batch):
        np_img = img.squeeze(-1).cpu().numpy().astype(np.uint8)
        Image.fromarray(np_img).save(os.path.join(os.path.join(path, "masks"), str(idx + offset) + ".png"))

    torch.save(labels.cpu(), os.path.join(path, "labels.pt"))


def load_masks_from_disk(path="../results/shapes/"):
    if not os.path.exists(os.path.join(path, "masks/")):
        raise FileNotFoundError("Mask files not found.")
    if not os.path.exists(os.path.join(path, "labels.pt")):
        raise FileNotFoundError("Labels file not found.")

    labels = torch.load(os.path.join(path, "labels.pt"))

    files = glob.glob(os.path.join(path, "masks/*"))

    files = sorted(files, key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))

    if len(files) != labels.shape[0]:
        raise ValueError("Length of labels file does not match the number of masks.")

    images = [torch.from_numpy(np.array(Image.open(file))).float() for file in files]
    batch = torch.stack(images)

    if len(batch.size()) == 3:
        batch = batch.unsqueeze(-1)

    batch = batch.permute(0, 3, 1, 2)
    batch = ((batch / 255.0) - 0.5) * 2.0

    return batch, labels


def load_mask_batch_from_disk(path="../results/shapes/", batch_size: int = 32, start_index: int = 0):
    if not os.path.exists(os.path.join(path, "masks/")):
        raise FileNotFoundError("Mask files not found.")
    if not os.path.exists(os.path.join(path, "labels.pt")):
        raise FileNotFoundError("Labels file not found.")

    labels = torch.load(os.path.join(path, "labels.pt"))

    files = glob.glob(os.path.join(path, "masks/*"))
    files = sorted(files, key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))

    if start_index >= len(files):
        return [], torch.tensor([], dtype=torch.long), len(files)

    end_index = min(start_index + batch_size, len(files))

    files_batch = files[start_index:end_index]
    images_batch = [torch.from_numpy(np.array(Image.open(file))).float() for file in files_batch]
    labels_batch = labels[start_index:end_index]

    images_batch = torch.stack(images_batch)

    if len(images_batch.size()) == 3:
        images_batch = images_batch.unsqueeze(-1)

    images_batch = images_batch.permute(0, 3, 1, 2)
    images_batch = ((images_batch / 255.0) - 0.5) * 2.0

    return images_batch, labels_batch, len(files)


def save_evaluation_metrics(metrics, path="../results/shapes/"):
    with open(os.path.join(path, "test_results.json"), "w") as f:
        json.dump(metrics, f, indent=4)




