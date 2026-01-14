import cv2
import numpy as np
from robodk import robolink, robomath as rdm

# =============================
# Image parameters
# =============================

IMAGE_PATH = r'C:\RoboDK_Draw\clipper_photo2.jpg'

# Known blue line length (calibration)
KNOWN_LINE_LENGTH_CM = 10.0
KNOWN_LINE_LENGTH_MM = KNOWN_LINE_LENGTH_CM * 10.0

# =============================
# Robot parameters
# =============================

Z_SAFE = 0.0       # Safe height above the table (mm)
Z_DRAW = 0.0       # Drawing height (close to the surface)
SPEED_DRAW = 100.0 # Movement speed (mm/s)

PHOTO_OFFSET_X = 0.0  # mm
PHOTO_OFFSET_Y = 0.0  # mm

scale_mm_per_px = None

# =============================
# Connect to RoboDK
# =============================

RDK = robolink.Robolink()
robot = RDK.Item('Yaskawa HC10DTP')
home = RDK.Item('DrawHome')

if not robot.Valid():
    raise Exception('Robot "Yaskawa HC10DTP" not found')
if not home.Valid():
    raise Exception('Target "DrawHome" not found')

robot.setSpeed(SPEED_DRAW)
home_pose = home.Pose()

# =============================
# Load image
# =============================

img_orig = cv2.imread(IMAGE_PATH)
if img_orig is None:
    raise Exception(f'Failed to load image: {IMAGE_PATH}')

print(f'Original image size: {img_orig.shape[1]}x{img_orig.shape[0]} pixels')

# =============================
# STEP 1. Detect white paper and "whiten" everything outside it
# =============================

hsv_full = cv2.cvtColor(img_orig, cv2.COLOR_BGR2HSV)

# White mask (paper sheet)
lower_white = np.array([0, 0, 180], dtype=np.uint8)
upper_white = np.array([180, 60, 255], dtype=np.uint8)
mask_white_full = cv2.inRange(hsv_full, lower_white, upper_white)

kernel7 = np.ones((7, 7), np.uint8)
mask_white_full = cv2.morphologyEx(mask_white_full, cv2.MORPH_CLOSE, kernel7, iterations=2)

contours_white, _ = cv2.findContours(
    mask_white_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

if len(contours_white) == 0:
    raise Exception('White paper not found! This logic requires a white paper background.')

# Take the largest white contour as the paper
paper_cnt = max(contours_white, key=cv2.contourArea)

# Paper mask with the same size as the original image
mask_paper_full = np.zeros(img_orig.shape[:2], np.uint8)
cv2.drawContours(mask_paper_full, [paper_cnt], -1, 255, thickness=cv2.FILLED)

# Crop both image and mask by the paper bounding box
x_p, y_p, w_p, h_p = cv2.boundingRect(paper_cnt)
img = img_orig[y_p:y_p + h_p, x_p:x_p + w_p].copy()
mask_paper = mask_paper_full[y_p:y_p + h_p, x_p:x_p + w_p].copy()

# Replace everything outside the paper with pure white (ignore table/background)
img[mask_paper == 0] = [255, 255, 255]

h_img, w_img = img.shape[:2]
print(f'Size after cropping to paper: {w_img}x{h_img} pixels')

# =============================
# STEP 2. Detect the blue line and compute scale
# =============================

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# HSV range for the blue line (tuned for phone_photo.jpg)
lower_blue = np.array([102, 80, 150], dtype=np.uint8)
upper_blue = np.array([118, 255, 255], dtype=np.uint8)

mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel7, iterations=1)

contours_blue, _ = cv2.findContours(
    mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

if len(contours_blue) == 0:
    raise Exception('Blue line not found! Try different HSV thresholds or draw the line brighter.')

# Select the line: long, thin contour in the lower half of the paper
line_contour = None
max_width = 0

for c in contours_blue:
    x_b, y_b, w_b, h_b = cv2.boundingRect(c)
    if h_b == 0:
        continue
    aspect = w_b / float(h_b)

    # Line: width > height, very elongated, and located in the lower half
    if w_b > h_b and aspect > 5 and y_b > h_img // 2:
        if w_b > max_width:
            max_width = w_b
            line_contour = c

# If not found by geometry, fallback: contour with maximum width
if line_contour is None:
    line_contour = max(contours_blue, key=lambda c: cv2.boundingRect(c)[2])

blue_cnt = line_contour

# Minimum area rectangle for the line
rect = cv2.minAreaRect(blue_cnt)
(center_x, center_y), (line_w, line_h), angle = rect
line_length_px = max(line_w, line_h)

if line_length_px < 10:
    raise Exception('Detected "line" is too short. Most likely detection is wrong.')

# Scale based only on the 10 cm blue line
scale_mm_per_px = KNOWN_LINE_LENGTH_MM / line_length_px
scale_cm_per_px = KNOWN_LINE_LENGTH_CM / line_length_px

print(f'Blue line length: {line_length_px:.1f} pixels')
print(f'Scale: {scale_cm_per_px:.4f} cm/pixel ({scale_mm_per_px:.4f} mm/pixel)')

# Debug: show which contour is treated as the line
img_blue_debug = img.copy()
box = cv2.boxPoints(rect)
box = np.int32(box)
cv2.drawContours(img_blue_debug, [box], 0, (255, 0, 0), 2)

# =============================
# STEP 3. Detect the object ONLY inside the paper
# =============================

# 1) Canny
blur = cv2.GaussianBlur(img, (5, 5), 0)
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)

# 2) Inner paper area (remove paper border)
kernel3 = np.ones((3, 3), np.uint8)
paper_inner = cv2.erode(mask_paper, kernel3, iterations=3)

# 3) Object region: inside paper and NOT blue
mask_not_blue = cv2.bitwise_not(mask_blue)
mask_object_region = cv2.bitwise_and(paper_inner, mask_not_blue)
mask_object_region = cv2.morphologyEx(mask_object_region, cv2.MORPH_OPEN, kernel3, iterations=1)

# 4) Keep edges only in the object region
edges_obj = cv2.bitwise_and(edges, edges, mask=mask_object_region)

# 5) Dilate edges
edges_dilated = cv2.dilate(edges_obj, kernel7, iterations=1)

# 6) Find object contours
contours_obj, _ = cv2.findContours(
    edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

min_area = 500.0
candidate_contours = [c for c in contours_obj if cv2.contourArea(c) > min_area]

if len(candidate_contours) == 0:
    raise Exception('No object contours found inside the paper. Check background/masks.')

main_cnt = max(candidate_contours, key=cv2.contourArea)
area_px = cv2.contourArea(main_cnt)

print(
    f'Object contour area: {area_px:.1f} px^2 '
    f'(~ {area_px * (scale_mm_per_px ** 2) / 100.0:.1f} cm^2)'
)

x, y, w, h = cv2.boundingRect(main_cnt)
cx = x + w / 2.0
cy = y + h / 2.0

print(
    f'Object size in photo: {w}x{h} pixels '
    f'(~ {w * scale_mm_per_px:.1f} x {h * scale_mm_per_px:.1f} mm)'
)
print(f'Object center (pixels): ({cx:.1f}, {cy:.1f})')

# =============================
# Visualization (debug)
# =============================

img_debug = img.copy()
cv2.drawContours(img_debug, [main_cnt], -1, (0, 255, 0), 2)
cv2.circle(img_debug, (int(cx), int(cy)), 5, (0, 0, 255), -1)
cv2.rectangle(img_debug, (x, y), (x + w, y + h), (0, 255, 255), 2)

cv2.imshow('Mask paper (cropped)', mask_paper)
cv2.imshow('Paper inner', paper_inner)
cv2.imshow('Blue mask', mask_blue)
cv2.imshow('Blue line debug', img_blue_debug)
cv2.imshow('Object region', mask_object_region)
cv2.imshow('Edges object', edges_obj)
cv2.imshow('Edges dilated', edges_dilated)
cv2.imshow('Detected Object', img_debug)
print('Press any key to continue robot motion...')
cv2.waitKey(0)
cv2.destroyAllWindows()

# =============================
# Contour approximation / smoothing
# =============================

epsilon = 0.0 * cv2.arcLength(main_cnt, True)
approx_cnt = cv2.approxPolyDP(main_cnt, epsilon, True)
print(f'Number of points after approximation: {len(approx_cnt)}')

# =============================
# Convert pixels to robot coordinates
# =============================

def pixel_to_robot_coords(px, py, center_x, center_y):
    dx_px = px - center_x
    dy_px = py - center_y

    dx_mm = dx_px * scale_mm_per_px
    dy_mm = dy_px * scale_mm_per_px

    x_robot = -dy_mm + PHOTO_OFFSET_X
    y_robot = dx_mm + PHOTO_OFFSET_Y

    return x_robot, y_robot

# =============================
# Robot motion and contour tracing
# =============================

x_center, y_center = pixel_to_robot_coords(cx, cy, cx, cy)
pose_center_safe = home_pose * rdm.transl(x_center, y_center, Z_SAFE)

print('Moving to object center...')
robot.MoveJ(pose_center_safe)

print('Starting contour tracing...')
trajectory_points = []

for i in range(len(approx_cnt)):
    px, py = approx_cnt[i, 0]
    x_robot, y_robot = pixel_to_robot_coords(px, py, cx, cy)
    trajectory_points.append((x_robot, y_robot))

trajectory_points.append(trajectory_points[0])

x_start, y_start = trajectory_points[0]
pose_start_safe = home_pose * rdm.transl(x_start, y_start, Z_SAFE)
pose_start_draw = home_pose * rdm.transl(x_start, y_start, Z_DRAW)

robot.MoveJ(pose_start_safe)
robot.MoveL(pose_start_draw)

for i, (x, y) in enumerate(trajectory_points[1:], 1):
    pose_draw = home_pose * rdm.transl(x, y, Z_DRAW)
    robot.MoveL(pose_draw)
    if i % 10 == 0:
        print(f'Processed points: {i}/{len(trajectory_points)}')

pose_end_safe = home_pose * rdm.transl(x_start, y_start, Z_SAFE)
robot.MoveL(pose_end_safe)
robot.MoveJ(home_pose)

print('Tracing finished!')
print(f'Total traced points: {len(trajectory_points)}')
