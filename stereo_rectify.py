#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stereo rectification (ORB + Essential) for RGB or grayscale pairs,
driven by a YAML config file.
Requires: PyYAML (pip install pyyaml), OpenCV, NumPy
"""

import yaml
import cv2
import numpy as np
import os

# --- 1. Load config.yaml ---
with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

# --- 2. Unpack parameters ---
# Image paths
img1_path = cfg['images']['left']
img2_path = cfg['images']['right']

# Camera intrinsics & extrinsics
cam = cfg['camera']
f_mm       = cam['focal_length_mm']    # focal length in mm
pixel_size = cam['pixel_size_m']       # pixel pitch in meters
baseline   = cam['baseline_m']         # stereo baseline in meters
dist_coeffs = np.array(cam.get('dist_coeffs', [0,0,0,0,0]), dtype=np.float64)

# Rectification settings
rect_cfg   = cfg['rectify']
alpha      = rect_cfg.get('alpha', -1)     # [-1..1], -1 = max crop
output_dir = rect_cfg.get('output_dir', '.')  # where to save maps & images

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# --- 3. Load stereo pair in color + create grayscale copies ---
img1_color = cv2.imread(img1_path, cv2.IMREAD_COLOR)
img2_color = cv2.imread(img2_path, cv2.IMREAD_COLOR)
assert img1_color is not None and img2_color is not None, "Could not load input images!"
img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

# --- 4. Image size & compute focal in pixels ---
h, w = img1.shape[:2]
print(f"Using pixel size: {pixel_size*1e6:.2f} µm")
f_px = (f_mm * 1e-3) / pixel_size
cx, cy = w / 2.0, h / 2.0

# Build intrinsic matrix K
K = np.array([
    [f_px,   0.0, cx],
    [0.0,   f_px, cy],
    [0.0,    0.0, 1.0]
], dtype=np.float64)

# --- 5. ORB feature detection & matching ---
orb = cv2.ORB_create(2000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = sorted(bf.match(des1, des2), key=lambda m: m.distance)[:1000]
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

# --- 6. Estimate Essential matrix & recover pose ---
E, _       = cv2.findEssentialMat(pts1, pts2, K,
                                  method=cv2.RANSAC,
                                  prob=0.999, threshold=1.0)
_, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

# --- 7. Stereo rectification ---
T = t * baseline  # scale unit-norm t by true baseline
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    cameraMatrix1=K, distCoeffs1=dist_coeffs,
    cameraMatrix2=K, distCoeffs2=dist_coeffs,
    imageSize=(w, h),
    R=R, T=T,
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha=alpha
)

# --- 8. Print the new rectified intrinsics ---
K_rect1 = P1[:, :3]
K_rect2 = P2[:, :3]
print("New rectified left intrinsics:\n", K_rect1)
print("New rectified right intrinsics:\n", K_rect2)

# --- 9. Compute rectification maps ---
map1x, map1y = cv2.initUndistortRectifyMap(
    K, dist_coeffs, R1, P1, (w, h), cv2.CV_32FC1
)
map2x, map2y = cv2.initUndistortRectifyMap(
    K, dist_coeffs, R2, P2, (w, h), cv2.CV_32FC1
)

# Save the maps
np.save(os.path.join(output_dir, 'map1x.npy'), map1x)
np.save(os.path.join(output_dir, 'map1y.npy'), map1y)
np.save(os.path.join(output_dir, 'map2x.npy'), map2x)
np.save(os.path.join(output_dir, 'map2y.npy'), map2y)
print("Saved rectification maps to", output_dir)

# --- 10. Apply remap to warp both grayscale & RGB images ---
rect1       = cv2.remap(img1,       map1x, map1y, cv2.INTER_LINEAR)
rect2       = cv2.remap(img2,       map2x, map2y, cv2.INTER_LINEAR)
rect1_color = cv2.remap(img1_color, map1x, map1y, cv2.INTER_LINEAR)
rect2_color = cv2.remap(img2_color, map2x, map2y, cv2.INTER_LINEAR)

# --- 11. Save rectified outputs ---
base1 = os.path.splitext(os.path.basename(img1_path))[0]
base2 = os.path.splitext(os.path.basename(img2_path))[0]

# Create subfolders for color and grayscale outputs
color_dir = os.path.join(output_dir, 'color')
gray_dir = os.path.join(output_dir, 'grayscale')
os.makedirs(color_dir, exist_ok=True)
os.makedirs(gray_dir, exist_ok=True)

# Save grayscale rectified images
cv2.imwrite(os.path.join(gray_dir, f'{base1}_rect.png'), rect1)
cv2.imwrite(os.path.join(gray_dir, f'{base2}_rect.png'), rect2)

# Save color rectified images
cv2.imwrite(os.path.join(color_dir, f'{base1}_rect_color.png'), rect1_color)
cv2.imwrite(os.path.join(color_dir, f'{base2}_rect_color.png'), rect2_color)

print("Saved rectified images to", output_dir)
