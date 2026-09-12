# Baseline ST-KNN (Algorithm 1)
# Uses a 3D grid to make searching faster

import numpy as np
from collections import defaultdict
from utils import STData, spatiotemporal_dist

class VoxelGrid:
    def __init__(self, obs: STData, bins_xy=10, bins_t=24):
        self.bins_xy = bins_xy
        self.bins_t = bins_t
        self.grid = defaultdict(list)
        
        # loop through all known data points
        for i in range(len(obs.v)):
            x, y = obs.xy[i]
            t = obs.t[i]
            
            # calculate which grid cell this point belongs to
            ix = max(0, min(int(x * bins_xy), bins_xy - 1))
            iy = max(0, min(int(y * bins_xy), bins_xy - 1))
            it = max(0, min(int(t * bins_t), bins_t - 1))
            
            self.grid[(ix, iy, it)].append(i)
            
    def get_voxel_coords(self, x, y, t):
        ix = max(0, min(int(x * self.bins_xy), self.bins_xy - 1))
        iy = max(0, min(int(y * self.bins_xy), self.bins_xy - 1))
        it = max(0, min(int(t * self.bins_t), self.bins_t - 1))
        return ix, iy, it
        
    def get_points_in_ring(self, cx, cy, ct, radius):
        # find all points in a square ring around the center cell
        points = []
        for ix in range(cx - radius, cx + radius + 1):
            if ix < 0 or ix >= self.bins_xy: continue
            for iy in range(cy - radius, cy + radius + 1):
                if iy < 0 or iy >= self.bins_xy: continue
                for it in range(ct - radius, ct + radius + 1):
                    if it < 0 or it >= self.bins_t: continue
                    
                    # only grab cells that are exactly on the edge of the current radius
                    if radius == 0 or max(abs(ix - cx), abs(iy - cy), abs(it - ct)) == radius:
                        points.extend(self.grid.get((ix, iy, it), []))
        return points

def knn_st_voxel_predict(voxel_grid: VoxelGrid, obs: STData, x_xy, x_t, alpha=0.5, k=4, eps=1e-3):
    # predict missing value using baseline method
    ix, iy, it = voxel_grid.get_voxel_coords(x_xy[0], x_xy[1], x_t)
    
    candidates = []
    radius = 0
    # keep looking outwards until we find enough candidates
    max_radius = max(voxel_grid.bins_xy, voxel_grid.bins_t)
    while len(candidates) < k and radius <= max_radius:
        candidates.extend(voxel_grid.get_points_in_ring(ix, iy, it, radius))
        radius += 1
        
    if not candidates:
        return 0.0 # return 0 if nothing is found nearby
        
    cand_idx = np.array(candidates)
    
    # compute the exact distance for these candidates
    d = spatiotemporal_dist(obs.xy[cand_idx], obs.t[cand_idx], x_xy, x_t, alpha)
    
    # sort them to find the closest ones
    k_final = min(k, len(d))
    idx_sort = np.argsort(d)[:k_final]
    best_idx = cand_idx[idx_sort]
    best_d = d[idx_sort]
    
    # use inverse distance weighting (IDW) formula
    w = 1.0 / (best_d + eps)
    return float(np.sum(w * obs.v[best_idx]) / np.sum(w))
