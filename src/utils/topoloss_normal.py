# Original loss functions of TopoDiffusionNet, with additional multi-process support.
# Code originally from https://github.com/Saumya-Gupta-26/TopoDiffusionNet/blob/master/improved_diffusion/topoloss_normal.py.

from __future__ import print_function, division

import torch
import numpy as np
from gudhi.wasserstein import wasserstein_distance
import cripser as cr
import os
from concurrent.futures import ProcessPoolExecutor

# debug flag. If set to True, prints additional information useful for debugging.
printstuff = False


def compute_dgm_force(stu_lh_dgm, tea_lh_dgm):
    """
    Compute the persistent diagram of the image

    Args:
        stu_lh_dgm: likelihood persistent diagram of student model.
        tea_lh_dgm: likelihood persistent diagram of teacher model.

    Returns:
        idx_holes_to_remove: The index of student persistent points that require to remove for the following training process [aka stu dots matched to diagonal]
        off_diagonal_match: The index pairs of persistent points that requires to fix in the following training process [aka stu dots matched to tea]

    """
    if stu_lh_dgm.shape[0] == 0:
        idx_holes_to_remove, off_diagonal_match = np.zeros((0,2)), np.zeros((0,2))
        return idx_holes_to_remove, off_diagonal_match

    if (tea_lh_dgm.shape[0] == 0):
        tea_pers = None
        tea_n_holes = 0
    else:
        tea_pers = abs(tea_lh_dgm[:, 1] - tea_lh_dgm[:, 0])
        tea_n_holes = tea_pers.size

    if (tea_pers is None or tea_n_holes == 0):
        idx_holes_to_remove = list(set(range(stu_lh_dgm.shape[0])))
        off_diagonal_match = list()
    else:
        idx_holes_to_remove, off_diagonal_match = get_matchings(stu_lh_dgm, tea_lh_dgm)

    return idx_holes_to_remove, off_diagonal_match


def getCriticalPoints_cr(likelihood, topo_dim, threshold):

    lh = 1 - likelihood
    pd = cr.computePH(lh, maxdim=1, location="birth") # dim birth death x1  y1  z1  x2  y2  z2
    pd_arr_lh = pd[pd[:, 0] == topo_dim] # 0 or 1-dim topological features
    pd_lh = pd_arr_lh[:, 1:3] # birth time and death time
    # birth critical points
    bcp_lh = pd_arr_lh[:, 3:5]
    # death critical points
    dcp_lh = pd_arr_lh[:, 6:8]
    pairs_lh_pa = pd_arr_lh.shape[0] != 0 and pd_arr_lh is not None

    # if the death time is inf, set it to 1.0
    for i in pd_lh:
        if i[1] > 1.0:
            i[1] = 1.0

    pd_pers = abs(pd_lh[:, 1] - pd_lh[:, 0])
    valid_idx = np.where(pd_pers > threshold)[0]
    noisy_idx = np.where(pd_pers <= threshold)[0]

    #return pd_lh_filtered, bcp_lh_filtered, dcp_lh_filtered, pairs_lh_pa
    return pd_lh, bcp_lh, dcp_lh, pairs_lh_pa, valid_idx, noisy_idx

def get_matchings(lh_stu, lh_tea):

    cost, matchings = wasserstein_distance(lh_stu, lh_tea, matching=True)

    dgm1_to_diagonal = matchings[matchings[:,1] == -1, 0] # dots in stu that matched to diagonal
    off_diagonal_match = np.delete(matchings, np.where(matchings == -1)[0], axis=0) # remove any diagonal dots in stu (tea had extra dots which mapped to stu diagonal)

    return dgm1_to_diagonal, off_diagonal_match

def interpolate(nparr, omin = 0., omax = 1.):
    imin  = np.min(nparr)
    imax = np.max(nparr)

    denom = imax - imin
    if denom == 0:
        denom = 1
    return (nparr-imin)*(omax-omin)/denom + omin

def _compute_weight_and_ref_maps(inter_np, xstart_np, topo_birth, topo_death, topo_noisy, topo_dim, pd_threshold = 0.):

    # inter_tensor, xstart_tensor are in [-1, 1] range
    # do normalization to [0,1] so that the PD calculations are correct. But when actual topoloss is computed, use the original [-1,1] itself

    likelihood = interpolate(inter_np)
    gt = interpolate(xstart_np)

    topo_cp_weight_map = np.zeros(likelihood.shape)
    topo_cp_ref_map = np.zeros(likelihood.shape)

    if(np.min(likelihood) == 1 or np.max(likelihood) == 0): return topo_cp_weight_map, topo_cp_ref_map, False
    if(np.min(gt) == 1 or np.max(gt) == 0): return topo_cp_weight_map, topo_cp_ref_map, False

    # Get the critical points of predictions and ground truth
    pd_lh, bcp_lh, dcp_lh, pairs_lh_pa, valid_idx_lh, noisy_idx_lh = getCriticalPoints_cr(likelihood, topo_dim, threshold=pd_threshold)
    pd_gt, bcp_gt, dcp_gt, pairs_lh_gt, valid_idx_gt, noisy_idx_gt = getCriticalPoints_cr(gt, topo_dim, threshold=0.)

    # select pd with high threshold to match
    pd_lh_for_matching = pd_lh[valid_idx_lh]
    pd_gt_for_matching = pd_gt[valid_idx_gt]

    # If the pairs not exist, continue for the next loop
    if not(pairs_lh_pa):
        return topo_cp_weight_map, topo_cp_ref_map, False
    if not(pairs_lh_gt):
        return topo_cp_weight_map, topo_cp_ref_map, False

    idx_holes_to_remove_for_matching, off_diagonal_for_matching = compute_dgm_force(pd_lh_for_matching, pd_gt_for_matching)

    idx_holes_to_remove = []
    off_diagonal_match = []

    if (len(idx_holes_to_remove_for_matching) > 0):
        for i in idx_holes_to_remove_for_matching:
            index_pd_lh_removed = np.where(np.all(pd_lh == pd_lh_for_matching[i], axis=1))[0][0]
            idx_holes_to_remove.append(index_pd_lh_removed)

    for k in noisy_idx_lh:
        idx_holes_to_remove.append(k)

    if len(off_diagonal_for_matching) > 0:
        for idx, (i, j) in enumerate(off_diagonal_for_matching):
            index_pd_lh = np.where(np.all(pd_lh == pd_lh_for_matching[i], axis=1))[0][0]
            index_pd_gt = np.where(np.all(pd_gt == pd_gt_for_matching[j], axis=1))[0][0]
            off_diagonal_match.append((index_pd_lh, index_pd_gt))

    if (len(off_diagonal_match) > 0 or len(idx_holes_to_remove) > 0):
        for (idx, (hole_indx, j)) in enumerate(off_diagonal_match):
            if topo_birth and (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[0] and int(
                    bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) < likelihood.shape[1]):
                topo_cp_weight_map[int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1])] = 1 # push birth to the corresponding teacher birth i.e. min birth prob or likelihood
                topo_cp_ref_map[int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1])] = xstart_np[int(bcp_gt[j][0]), int(bcp_gt[j][1])] #pd_gt[j][0]

            if topo_death and (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[
                0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) <
                    likelihood.shape[1]):
                topo_cp_weight_map[int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1])] = 1  # push death to the corresponding teacher death i.e. max death prob or likelihood
                topo_cp_ref_map[int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1])] = xstart_np[int(dcp_gt[j][0]), int(dcp_gt[j][1])] #pd_gt[j][1]

        if topo_noisy:
            for hole_indx in idx_holes_to_remove:
                if topo_birth and (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[
                    0] and int(bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) <
                        likelihood.shape[1]):
                    topo_cp_weight_map[int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1])] = 1  # push to diagonal

                    if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[
                        0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) <
                            likelihood.shape[1]):
                        topo_cp_ref_map[int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1])] = \
                            inter_np[int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1])] # lh_patch[int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1])]
                    else:
                        topo_cp_ref_map[int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1])] = 1

                if topo_death and (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[
                    0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) <
                        likelihood.shape[1]):
                    topo_cp_weight_map[int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1])] = 1  # push to diagonal

                    if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[
                        0] and int(bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) <
                            likelihood.shape[1]):
                        topo_cp_ref_map[int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1])] = \
                            inter_np[int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1])] # lh_patch[int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1])]
                    else:
                        topo_cp_ref_map[int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1])] = 0

    return topo_cp_weight_map, topo_cp_ref_map, True


def _worker(args):
    inter_np, xstart_np, topo_birth, topo_death, topo_noisy, topo_dim = args
    return _compute_weight_and_ref_maps(inter_np, xstart_np, topo_birth,
                                         topo_death, topo_noisy, topo_dim)


class TopoLossMSE2D(torch.nn.Module):
    """Multi-Core TopoLossMSE2D (ONE topo_dim, no combination of
       0dim + 1dim). Parallelized across (batch, channel) using
       ProcessPoolExecutor. Signature identical to the original."""

    def __init__(self, topo_birth, topo_death, topo_noisy, topo_dim, n_workers=None):
        super().__init__()
        self.topo_birth = topo_birth
        self.topo_death = topo_death
        self.topo_noisy = topo_noisy
        self.topo_dim = topo_dim


        if n_workers is None:
            n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK",
                                           os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1)))

        self.n_workers = max(1, n_workers)
        self._pool = None  # lazy init

        print(f"[TDN] Paralleling topological loss across {n_workers} workers")

    def _get_pool(self):
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=self.n_workers)
        return self._pool

    def shutdown(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def forward(self, pred, target):
        N, C, H, W = pred.shape
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        tasks = []
        for idx in range(N):
            for ch in range(C):
                tasks.append((pred_np[idx, ch], target_np[idx, ch],
                              self.topo_birth, self.topo_death,
                              self.topo_noisy, self.topo_dim))

        pool = self._get_pool()
        results = list(pool.map(_worker, tasks, chunksize=1))

        loss_val = torch.zeros(N, device=pred.device, dtype=pred.dtype)
        k = 0
        for idx in range(N):
            acc = torch.zeros((), device=pred.device, dtype=pred.dtype)
            for ch in range(C):
                weight_map, ref_map, valid = results[k]
                k += 1
                if valid:
                    w = torch.from_numpy(weight_map).to(pred.device, pred.dtype)
                    r = torch.from_numpy(ref_map).to(pred.device, pred.dtype)
                    acc = acc + ((pred[idx, ch] * w - r) ** 2).sum()
            loss_val[idx] = acc / C

        return loss_val
