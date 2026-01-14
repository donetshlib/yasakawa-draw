# Vision-Guided Contour Drawing in RoboDK (OpenCV + Yaskawa HC10DTP)

This project detects an object contour from a photo (taken above a sheet of paper), converts the contour points from pixels to millimeters using a blue calibration line, and then drives a **Yaskawa HC10DTP** robot in **RoboDK** to trace the contour.

## What it does

Given an input image:

1. **Finds the white paper** and ignores everything outside of it (sets background to pure white).
2. **Detects a blue calibration line** on the paper (known length: **10 cm**) and computes the scale (**mm per pixel**).
3. **Detects the main object contour** inside the paper area (Canny edges + masking).
4. **Approximates / smooths** the contour (OpenCV polygon approximation).
5. **Converts pixel coordinates to robot coordinates** (mm) using the scale and a simple coordinate mapping.
6. **Moves the robot** to the start point and traces the contour using `MoveL` commands.

## Requirements

* Python 3.8+ (recommended)
* RoboDK installed
* RoboDK Python API available (`robodk`)
* A RoboDK station with:

  * Robot item named: **`Yaskawa HC10DTP`**
  * Reference target/item named: **`DrawHome`**

Python packages:

* `opencv-python`
* `numpy`
* `robodk`

## Installation

```bash
pip install opencv-python numpy robodk
```

## Setup

### 1) Input image

In the script, set:

```python
IMAGE_PATH = r"C:\RoboDK_Draw\clipper_photo2.jpg"
```

**Recommendation:** store images locally (not in Git) and point `IMAGE_PATH` to your local file.

The image should contain:

* A **white sheet of paper** visible clearly
* A **blue line** on the paper used for calibration (known length: **10 cm**)
* The **object** you want to outline inside the paper

### 2) Calibration line length

By default:

```python
KNOWN_LINE_LENGTH_CM = 10.0
```

If your calibration line is different, change this value.

### 3) Robot settings (important)

These parameters control the robot motion:

```python
Z_SAFE = 0.0
Z_DRAW = 0.0
SPEED_DRAW = 100.0
PHOTO_OFFSET_X = 0.0
PHOTO_OFFSET_Y = 0.0
```

* `Z_SAFE`: safe height above the surface
* `Z_DRAW`: drawing height (touching or near paper)
* `PHOTO_OFFSET_X/Y`: offsets to align photo coordinates to the robot frame

You must tune these values for your setup (tool, table height, reference frame).

### 4) RoboDK item names

The script expects:

```python
robot = RDK.Item("Yaskawa HC10DTP")
home  = RDK.Item("DrawHome")
```

If your station uses different names, update them in the script.

## Run

```bash
python your_script_name.py
```

The script will open debug windows (masks, edges, detected contour).
Press any key to continue after verifying detection.

## Output / Debug windows

The script shows:

* Detected paper mask (cropped)
* Inner paper area
* Blue mask + detected blue line rectangle
* Object region mask
* Edges and dilated edges
* Final detected object contour + bounding box + center point

This helps you adjust HSV thresholds and edge detection parameters if needed.

## How pixel → robot conversion works

After the object center is found, each contour point is converted:

* Compute `dx`, `dy` relative to the object center in pixels
* Convert to mm using `scale_mm_per_px`
* Map to robot axes (example mapping used in the script):

```python
x_robot = -dy_mm + PHOTO_OFFSET_X
y_robot =  dx_mm + PHOTO_OFFSET_Y
```

If your coordinate system is rotated or mirrored, adjust this mapping.

## Common tuning tips

* If the **paper is not detected**, adjust the white HSV thresholds: `lower_white`, `upper_white`
* If the **blue line is not detected**, adjust: `lower_blue`, `upper_blue`
* If the **object contour is noisy**, tune:

  * Canny thresholds: `cv2.Canny(gray, 50, 150)`
  * Morphology kernels/iterations
  * `min_area`

## Notes / Limitations

* Lighting changes can affect HSV thresholds.
* The approach assumes the paper is the largest white region in the image.
* For better contour quality, consider increasing smoothing or using contour resampling.
