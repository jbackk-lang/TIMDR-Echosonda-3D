import numpy as np
from timdr_echosonda_3d import TIMDREchosonda3D
from timdr_midwater_targets import TIMDRMidWaterDetector

print("=== TIMDR-Echosonda-3D (dane ze zgloszenia) ===")
ech = TIMDREchosonda3D(k_neighbors=4)
points = [
    [0, 0, 10], [1, 0, 12], [2, 0, 15], [3, 0, 40], [4, 0, 42], [5, 0, 43],
]
flow = ech.flow_3d(points)
twist_pts, twist_strength = ech.twist_3d(points)
smooth = ech.trm_surface(points)
curv = ech.curvature(points)
segments = ech.segment_shapes(points, twist_pts)
print("FLOW:", flow)
print("TWIST:", twist_pts)
print("TRM:", smooth)
print("CURVATURE:", curv)
print("SEGMENTS:", segments)

print("\n=== TIMDR Mid-Water Target Detector (ryby / wieloryby / statki) ===")
rng = np.random.default_rng(0)
bottom = np.column_stack([
    rng.uniform(0, 100, 150), rng.uniform(0, 100, 150), 50.0 + rng.normal(0, 0.1, 150)
])
det = TIMDRMidWaterDetector(bottom)

# przykladowe surowe echa: [x, y, depth, amplitude] - mieszanka dna i celow
returns = [
    [50, 50, 49.9, 0.7],   # dno (odrzucone)
    [20, 20, 15.0, 0.25],  # pojedyncza ryba
    [80, 80, 10.0, 0.95],  # mocny, plytki cel
]
targets = det.detect_targets(returns, min_clearance=1.0)
print("Wykryte cele w toni (nie na dnie):", targets)
labels, unique = det.cluster_targets(targets, eps=2.0)
classified = det.classify_targets(targets, labels)
print("Klasyfikacja:", classified)
