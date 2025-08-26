#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply precomputed stereo rectification maps to new images.

Inputs:
  - Directory containing map1x.npy, map1y.npy, map2x.npy, map2y.npy
  - New left/right images (must have same resolution as maps)

Usage examples:
  # Single pair
  python apply_rectification.py \
      --maps ./output_dir_from_prev_script \
      --left path/to/new_left.png \
      --right path/to/new_right.png \
      --out ./rectified_out

  # Batch (folders of equal-length left/right images)
  python apply_rectification.py \
      --maps ./output_dir_from_prev_script \
      --left_dir ./new_lefts \
      --right_dir ./new_rights \
      --out ./rectified_out
"""

import os
import glob
import cv2
import numpy as np
import argparse

def load_maps(maps_dir):
    map1x = np.load(os.path.join(maps_dir, 'map1x.npy'))
    map1y = np.load(os.path.join(maps_dir, 'map1y.npy'))
    map2x = np.load(os.path.join(maps_dir, 'map2x.npy'))
    map2y = np.load(os.path.join(maps_dir, 'map2y.npy'))
    h, w = map1x.shape[:2]
    return (map1x, map1y, map2x, map2y), (w, h)

def remap_image(img_path, mapx, mapy, expected_size):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    h, w = img.shape[:2]
    if (w, h) != expected_size:
        raise ValueError(f"Image {img_path} size {w}x{h} != expected {expected_size}")
    rect = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
    return rect

def write_out(img, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)

def sorted_image_list(folder):
    exts = ('*.png','*.jpg','*.jpeg','*.bmp','*.tif','*.tiff')
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    return sorted(files)

def parse_args():
    p = argparse.ArgumentParser(description="Apply precomputed rectification maps to new images.")
    p.add_argument('--maps', required=True, help="Directory with map1x.npy/map1y.npy/map2x.npy/map2y.npy")
    p.add_argument('--left', help="Left image path (single pair)")
    p.add_argument('--right', help="Right image path (single pair)")
    p.add_argument('--left_dir', help="Folder of left images (batch mode)")
    p.add_argument('--right_dir', help="Folder of right images (batch mode)")
    p.add_argument('--out', required=True, help="Output directory")
    p.add_argument('--prefix', default='', help="Optional filename prefix")
    p.add_argument('--suffix', default='_rect', help="Suffix for output names")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    (map1x, map1y, map2x, map2y), expected_size = load_maps(args.maps)

    os.makedirs(args.out, exist_ok=True)

    # --- Single pair ---
    if args.left and args.right:
        baseL = os.path.splitext(os.path.basename(args.left))[0]
        baseR = os.path.splitext(os.path.basename(args.right))[0]

        rectL = remap_image(args.left,  map1x, map1y, expected_size)
        rectR = remap_image(args.right, map2x, map2y, expected_size)

        outL = os.path.join(args.out, f"{args.prefix}{baseL}{args.suffix}.png")
        outR = os.path.join(args.out, f"{args.prefix}{baseR}{args.suffix}.png")
        write_out(rectL, outL)
        write_out(rectR, outR)
        print(f"Saved:\n  {outL}\n  {outR}")
        exit(0)

    # --- Batch mode ---
    if args.left_dir and args.right_dir:
        left_list  = sorted_image_list(args.left_dir)
        right_list = sorted_image_list(args.right_dir)
        if len(left_list) != len(right_list):
            raise RuntimeError(f"Left/right folder counts differ: {len(left_list)} vs {len(right_list)}")

        for L, R in zip(left_list, right_list):
            baseL = os.path.splitext(os.path.basename(L))[0]
            baseR = os.path.splitext(os.path.basename(R))[0]
            rectL = remap_image(L, map1x, map1y, expected_size)
            rectR = remap_image(R, map2x, map2y, expected_size)
            outL = os.path.join(args.out, f"{args.prefix}{baseL}{args.suffix}.png")
            outR = os.path.join(args.out, f"{args.prefix}{baseR}{args.suffix}.png")
            write_out(rectL, outL)
            write_out(rectR, outR)
            print(f"Saved:\n  {outL}\n  {outR}")
        exit(0)

    raise SystemExit("Provide either --left & --right OR --left_dir & --right_dir.")
