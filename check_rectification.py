#!/usr/bin/env python3
"""
check_rectification.py

Check how well two stereo images are rectified by comparing ORB matches.

Usage:
    python check_rectification.py --left path/to/left.png --right path/to/right.png [--visual_test]

Arguments:
    --left         Path to left image
    --right        Path to right image
    --max_matches  Max number of ORB matches to use (default=200)
    --visual_test  If set, draw grid overlays for visual inspection
"""

import argparse
import cv2
import numpy as np

VISUAL_TEST = True  # set to True to visualize feature matches

def draw_grid(img,
              spacing: int = 50,
              color: tuple = (0, 0, 255),
              thickness: int = 1,
              font_scale: float = 0.4):
    """Draws a regular grid over the image with coordinate labels."""
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # vertical grid lines + x-axis labels
    for x in range(0, w, spacing):
        cv2.line(img, (x, 0), (x, h-1), color, thickness)
        cv2.putText(img, str(x),
                    (x+2, int(10 + font_scale*10)),
                    font, font_scale, color, 1, cv2.LINE_AA)

    # horizontal grid lines + y-axis labels
    for y in range(0, h, spacing):
        cv2.line(img, (0, y), (w-1, y), color, thickness)
        cv2.putText(img, str(y),
                    (2, y-2),
                    font, font_scale, color, 1, cv2.LINE_AA)

    return img

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--left',        required=True,  help="Path to left image")
    p.add_argument('--right',       required=True,  help="Path to right image")
    p.add_argument('--max_matches', type=int, default=200, help="Max # of ORB matches to use")
    p.add_argument('--visual_test', action='store_true', help="Draw grid overlays for inspection")
    args = p.parse_args()

    # 1) Load & detect/match ORB features
    img1 = cv2.imread(args.left,  cv2.IMREAD_COLOR)
    img2 = cv2.imread(args.right, cv2.IMREAD_COLOR)
    orb = cv2.ORB_create(1000)
    k1, d1 = orb.detectAndCompute(img1, None)
    k2, d2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(d1, d2), key=lambda m: m.distance)[:args.max_matches]

    pts1 = np.float32([k1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([k2[m.trainIdx].pt for m in matches])

    # Optional: show match visualization
    if args.visual_test:        
        print(f"Found {len(matches)} matches")
        img_matches = cv2.drawMatches(img1, k1, img2, k2, matches[:250], None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow("Matches", img_matches)
        cv2.waitKey(0)
        grid1 = draw_grid(img1.copy(), spacing=50)
        grid2 = draw_grid(img2.copy(), spacing=50)
        cv2.imshow("Left + Grid", grid1)
        cv2.imshow("Right + Grid", grid2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 2) Estimate Fundamental matrix → keep only inliers
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    mask = mask.ravel().astype(bool)
    in1, in2 = pts1[mask], pts2[mask]

    # 3) Numeric vertical-offset test
    dys     = np.abs(in1[:,1] - in2[:,1])
    med_dy  = np.median(dys)
    mean_dy = np.mean(dys)

    print("\n=== Left-vs-Right Vertical Offset (inliers) ===\n")
    print(f"Matches: {len(in1)}")
    print(f"Median |Δy| = {med_dy:.3f} px")
    print(f"Mean   |Δy| = {mean_dy:.3f} px\n")

    # Also show pre-RANSAC
    dys_all = np.abs(pts1[:, 1] - pts2[:, 1])
    print("=== All Matches Vertical Offset (pre-RANSAC) ===")
    print(f"Matches: {len(pts1)}")
    print(f"Median |Δy| = {np.median(dys_all):.3f} px")
    print(f"Mean   |Δy| = {np.mean(dys_all):.3f} px\n")

if __name__ == '__main__':
    main()
