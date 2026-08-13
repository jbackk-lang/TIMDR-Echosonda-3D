import numpy as np
from scipy.spatial import KDTree

class TIMDREchosonda3D:
    def __init__(self, k_neighbors=12):
        self.k = k_neighbors

    def flow_3d(self, points):
        pts = np.array(points)
        x, y, z = pts[:,0], pts[:,1], pts[:,2]
        dzdx = np.gradient(z, x)
        dzdy = np.gradient(z, y)
        flow = np.vstack([dzdx, dzdy]).T
        return flow

    def twist_3d(self, points, threshold=0.35):
        flow = self.flow_3d(points)
        dflow = np.gradient(flow, axis=0)
        twist_strength = np.linalg.norm(dflow, axis=1)
        twist_points = np.where(twist_strength > threshold)[0]
        return twist_points, twist_strength

    def trm_surface(self, points):
        pts = np.array(points)
        tree = KDTree(pts[:, :2])
        smooth = pts.copy()
        for i, p in enumerate(pts):
            _, idx = tree.query(p[:2], k=self.k)
            neighborhood = pts[idx]
            smooth[i, 2] = np.median(neighborhood[:,2])
        return smooth

    def curvature(self, points):
        pts = np.array(points)
        z = pts[:,2]
        d2 = np.gradient(np.gradient(z))
        return d2

    def segment_shapes(self, points, twist_points):
        pts = np.array(points)
        z = pts[:,2]
        segments = {"uskoki": [], "jamy": [], "garby": [], "polki": []}
        for idx in twist_points:
            if idx == 0 or idx == len(z)-1:
                continue
            left = z[idx-1]
            mid = z[idx]
            right = z[idx+1]
            if mid < left and mid < right:
                segments["jamy"].append(idx)
            elif mid > left and mid > right:
                segments["garby"].append(idx)
            elif abs(left - right) < 0.1:
                segments["polki"].append(idx)
            else:
                segments["uskoki"].append(idx)
        return segments
