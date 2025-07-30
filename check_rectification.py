#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
from pathlib import Path

def draw_grid(img,
              spacing: int = 50,
              color: tuple = (0, 0, 255),
              thickness: int = 1,
              font_scale: float = 0.4):
    """
    Draws a regular grid over the image with coordinate labels.
    - spacing: pixel interval between lines (both horizontal & vertical)
    - color: BGR tuple for lines & text
    - thickness: line thickness
    - font_scale: size of the coordinate text
    """
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # vertical grid lines + x-axis labels
    for x in range(0, w, spacing):
        cv2.line(img, (x, 0), (x, h-1), color, thickness)
        # put the x-coordinate label at the top, slightly offset
        cv2.putText(img, str(x),
                    (x+2, int(10 + font_scale*10)),
                    font, font_scale, color, 1, cv2.LINE_AA)

    # horizontal grid lines + y-axis labels
    for y in range(0, h, spacing):
        cv2.line(img, (0, y), (w-1, y), color, thickness)
        # put the y-coordinate label on the left, slightly offset
        cv2.putText(img, str(y),
                    (2, y-2),
                    font, font_scale, color, 1, cv2.LINE_AA)

    return img

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--left',        required=True,  help="path to left image")
    p.add_argument('--right',       required=True,  help="path to right image")
    p.add_argument('--out_dir',     default='rect_check', help="where to save outputs")
    p.add_argument('--max_matches', type=int, default=200, help="max # of ORB matches to use")
    p.add_argument('--visual_test', action='store_true', help="draw grid for visual inspection")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    # 1) Load & detect/match ORB features
    img1 = cv2.imread(args.left,  cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(args.right, cv2.IMREAD_GRAYSCALE)
    orb = cv2.ORB_create(1000)
    k1, d1 = orb.detectAndCompute(img1, None)
    k2, d2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(d1, d2), key=lambda m: m.distance)[:args.max_matches]

    pts1 = np.float32([k1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([k2[m.trainIdx].pt for m in matches])

    # 2) Estimate Fundamental matrix (with RANSAC) → keep only inliers
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    mask = mask.ravel().astype(bool)
    in1, in2 = pts1[mask], pts2[mask]

    # 3) Numeric “same-row” test
    dys     = np.abs(in1[:,1] - in2[:,1])
    med_dy  = np.median(dys)
    mean_dy = np.mean(dys)

    print("\n=== Left-vs-Right Vertical Offset ===\n")
    print(f"Matches (inliers): {len(in1)}")
    print(f"Median |Δy| = {med_dy:.3f} px")
    print(f"Mean   |Δy| = {mean_dy:.3f} px\n")

    # 4) Visual test: optionally draw horizontal lines at each inlier y (from left) on both images
    if getattr(args, "visual_test", False):  # default to False if not present
        color1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        color2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        draw_grid(color1, spacing=50)
        draw_grid(color2, spacing=50)

        cv2.imwrite(str(out_dir/'grid_left.png'),  color1)
        cv2.imwrite(str(out_dir/'grid_right.png'), color2)

    print(f"Saved comparison images to '{out_dir}/':")
    print("  - horiz_left.png")
    print("  - horiz_right.png\n")
    print("Open them side-by-side to visually verify how well features align vertically.")

if __name__ == '__main__':
    main()
