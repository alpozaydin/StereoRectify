# StereoRectifier

A Python tool for stereo image rectification that automatically estimates extrinsic parameters using ORB feature matching and essential matrix estimation. Perfect for stereo vision applications where camera calibration data is limited or unavailable.

## 🎯 What it does

Stereo rectification transforms a pair of stereo images so that corresponding points lie on the same horizontal line (epipolar lines become horizontal). This simplifies stereo matching algorithms and is essential for:

- **Stereo depth estimation**
- **3D reconstruction**
- **Computer vision applications**
- **Autonomous driving systems**

## ✨ Features

- **Automatic extrinsic estimation** using ORB features and essential matrix
- **No manual calibration required** - works with unknown camera poses
- **Supports both RGB and grayscale images**
- **Configurable rectification parameters**
- **Quality verification tools**
- **Saves rectification maps for reuse**

## 📋 Requirements

```bash
pip install opencv-python numpy pyyaml
```

## 🚀 Quick Start

### 1. Prepare your images
Place your stereo image pair in your project directory.

### 2. Configure the tool
Edit `config.yaml` with your camera parameters and image paths:

```yaml
images:
  left:  path/to/your/left_image.png
  right: path/to/your/right_image.png

camera:
  focal_length_mm: 50.0      # Your camera's focal length in mm
  pixel_size_m: 7.4e-6       # Pixel size in meters (e.g., 7.4 µm)
  baseline_m: 0.270          # Distance between cameras in meters
  dist_coeffs: [0.0, 0.0, 0.0, 0.0, 0.0]  # Distortion coefficients

rectify:
  alpha: -1                  # Rectification crop factor (-1 = max crop)
  output_dir: rectified_out  # Output directory
```

### 3. Run rectification
```bash
python stereo_rectify.py
```

### 4. Verify results
```bash
python check_rectification.py --left rectified_out/grayscale/left_rect.png --right rectified_out/grayscale/right_rect.png --visual_test
```

## 📁 Output Structure

After running the tool, you'll find:

```
rectified_out/
├── color/                    # Rectified RGB images
│   ├── left_rect_color.png
│   └── right_rect_color.png
├── grayscale/               # Rectified grayscale images
│   ├── left_rect.png
│   └── right_rect.png
├── map1x.npy               # Rectification maps for left camera
├── map1y.npy
├── map2x.npy               # Rectification maps for right camera
└── map2y.npy
```

## 🔧 Configuration Parameters

### Camera Parameters
- **`focal_length_mm`**: Camera focal length in millimeters
- **`pixel_size_m`**: Pixel pitch in meters (e.g., 7.4e-6 for 7.4 µm pixels)
- **`baseline_m`**: Distance between stereo cameras in meters
- **`dist_coeffs`**: Lens distortion coefficients [k1, k2, p1, p2, k3]

### Rectification Settings
- **`alpha`**: Crop factor for rectified images
  - `-1`: Maximum crop (no black borders)
  - `0`: No crop (may include black borders)
  - `1`: Minimum crop (preserves all image data)

## 🔍 Quality Verification

The `check_rectification.py` script helps verify rectification quality:

```bash
python check_rectification.py \
  --left rectified_out/grayscale/left_rect.png \
  --right rectified_out/grayscale/right_rect.png \
  --visual_test
```

This will:
- Calculate vertical offsets between corresponding points
- Generate visual comparison images with grids
- Report median and mean vertical alignment errors

## 🛠️ Advanced Usage

### Using saved rectification maps
The tool saves rectification maps as `.npy` files. You can reuse these for new images from the same camera setup:

```python
import cv2
import numpy as np

# Load saved maps
map1x = np.load('rectified_out/map1x.npy')
map1y = np.load('rectified_out/map1y.npy')
map2x = np.load('rectified_out/map2x.npy')
map2y = np.load('rectified_out/map2y.npy')

# Apply to new images
img1 = cv2.imread('new_left.png')
img2 = cv2.imread('new_right.png')

rect1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
rect2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📚 References

- OpenCV stereo rectification documentation
- ORB feature detection and matching
- Essential matrix estimation for stereo vision
