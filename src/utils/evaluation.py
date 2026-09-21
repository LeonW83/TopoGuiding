""" Script used for quantitative evaluation"""

import torch
import numpy as np
import cripser as cr
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.utils.io_handler import load_mask_batch_from_disk


def interpolate(nparr, omin = 0., omax = 1.):
    imin  = np.min(nparr)
    imax = np.max(nparr)

    denom = imax - imin
    if denom == 0:
        denom = 1
    return (nparr-imin)*(omax-omin)/denom + omin

def get_persistence(likelihood, topo_dim):
    lh = 1 - likelihood
    pd = cr.computePH(lh, maxdim=1, location="birth")  # dim birth death x1  y1  z1  x2  y2  z2
    pd_arr_lh = pd[pd[:, 0] == topo_dim]  # 0 or 1-dim topological features
    pd_lh = pd_arr_lh[:, 1:3]  # birth time and death time

    for i in pd_lh:
        if i[1] > 1.0:
            i[1] = 1.0

    pd_pers = abs(pd_lh[:, 1] - pd_lh[:, 0])

    return torch.from_numpy(pd_pers), torch.from_numpy(pd_lh[:, 1]) # persistence, death times

# Uses the first channel only (supports only 1-channel masks). Otherwise, persistence is hard to estimate.
def evaluate(masks, gt_labels, topo_dim: int=0, num_classes: int =None, mask_mode: str="hard"):

    if not mask_mode in ["soft", "hard"]:
        raise NotImplementedError("Only soft or hard mask modes are implemented.")

    # find predicted labels (number of components/holes)
    betti_numbers = []
    for idx in range(masks.size()[0]):
        inter = masks[idx, 0, :, :]

        if mask_mode == "hard":
            inter[inter <= 0.] = -1.0
            inter[inter > 0.] = 1.0

        #show_image(inter.unsqueeze(0))
        inter = torch.squeeze(inter).cpu().detach().numpy()
        likelihood = interpolate(inter)

        if np.min(likelihood) == 1.0 or np.max(likelihood) == 0.0:
            print("[WARN] A mask is only black/white.")

        pers, deaths = get_persistence(likelihood, topo_dim)

        betti_num=-1
        if mask_mode == "soft":
            sorted_pers, sorted_indices = torch.sort(pers, descending=True)

            diffs = torch.abs(torch.diff(sorted_pers))
            betti_num = torch.argmax(diffs) + 1
        elif mask_mode == "hard":
            betti_num = pers.shape[0]

        betti_numbers.append(betti_num)

    betti_numbers = torch.tensor(betti_numbers, device="cpu")


    acc = accuracy_score(gt_labels.cpu().numpy(), betti_numbers.numpy())

    precision, recall, macro_f1, support = precision_recall_fscore_support(
        gt_labels.cpu().numpy(),
        betti_numbers.cpu().numpy(),
        labels=np.unique(gt_labels.cpu().numpy()),
        average="macro",
        zero_division=0
    )

    metrics = {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "macro_f1": macro_f1,
    }

    return metrics

# Uses the first channel only (supports only 1-channel masks). Otherwise, persistence is hard to estimate.
def evaluate_from_disk(path="../results/shapes/", topo_dim: int=0, batch_size: int = 32, mask_mode: str="hard"):

    if not mask_mode in ["soft", "hard"]:
        raise NotImplementedError("Only soft or hard mask modes are implemented.")

    _, _, num_masks = load_mask_batch_from_disk(path, batch_size, 0)

    betti_numbers, labels = [], []
    start_index = 0
    while start_index < num_masks:
        mask_batch, label_batch, _ = load_mask_batch_from_disk(path, batch_size, start_index)

        labels.append(label_batch)

        # find predicted labels (number of components/holes)
        betti_numbers_batch = []
        for idx in range(mask_batch.size()[0]):
            inter = mask_batch[idx, 0, :, :]

            if mask_mode == "hard":
                inter[inter <= 0.] = -1.0
                inter[inter > 0.] = 1.0

            inter = torch.squeeze(inter).cpu().detach().numpy()
            likelihood = interpolate(inter)

            if np.min(likelihood) == 1.0 or np.max(likelihood) == 0.0:
                print("[WARN] A mask is only black/white.")

            pers, deaths = get_persistence(likelihood, topo_dim)

            betti_num = -1
            if mask_mode == "soft":
                sorted_pers, sorted_indices = torch.sort(pers, descending=True)

                diffs = torch.abs(torch.diff(sorted_pers))
                betti_num = torch.argmax(diffs) + 1
            elif mask_mode == "hard":
                betti_num = pers.shape[0]

            betti_numbers_batch.append(betti_num)

        betti_numbers.append(torch.tensor(betti_numbers_batch, device="cpu"))

        start_index += batch_size


    betti_numbers = torch.cat(betti_numbers)
    labels = torch.cat(labels)

    acc = accuracy_score(labels.cpu().numpy(), betti_numbers.numpy())

    precision, recall, macro_f1, support = precision_recall_fscore_support(
        labels.cpu().numpy(),
        betti_numbers.cpu().numpy(),
        labels=np.unique(labels.cpu().numpy()),
        average="macro",
        zero_division=0
    )

    metrics = {
        "n_masks": num_masks,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "macro_f1": macro_f1,
    }

    return metrics


