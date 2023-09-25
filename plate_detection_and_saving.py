import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
from imutils import contours


os.chdir('/Users/sara/Desktop/part/computervision_task3')
image = cv2.imread("merged_data/images/merged_image88.jpg")

coordinates = []
with open("merged_data/txtFiles/coordinates88.txt", "r") as file:
    for line in file:
        x, y = map(float, line.strip().split(", "))
        coordinates.append((x, y))

image = np.array(image)
coordinates = np.array(coordinates, dtype=np.float32)

rect_width = 500
rect_height = 200

destination_points = np.array([[0, 0], [rect_width, 0], [rect_width, rect_height], [0, rect_height]], dtype=np.float32)
# Calculate the perspective transformation matrix
perspective_matrix = cv2.getPerspectiveTransform(coordinates, destination_points)
# Apply the perspective transformation to rectify the plate
rectified_plate = cv2.warpPerspective(image, perspective_matrix, (rect_width, rect_height))

image_filename = os.path.join(f"detected_plate.jpg")
cv2.imwrite(image_filename, rectified_plate)

# Apply Laplacian filter for sharpening
kernel = np.array([[-1, -1, -1],
                   [-1, 9, -1],
                   [-1, -1, -1]])
sharpened_image = cv2.filter2D(rectified_plate, -1, kernel)

# Apply Gaussian blur for denoising
denoised_image = cv2.GaussianBlur(sharpened_image, (5, 5), 0)

image_filename = os.path.join(f"denoised_detected_plate.jpg")
cv2.imwrite(image_filename, denoised_image)

ref = cv2.cvtColor(denoised_image[30:180, 50:490], cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(ref, 80, 255, cv2.THRESH_BINARY_INV)[1]

refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
refCnts = imutils.grab_contours(refCnts)
refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]

letters = []
for (i, c) in enumerate(refCnts):
    # compute the bounding box for the digit, extract it, and resize
    (x, y, w, h) = cv2.boundingRect(c)
    if w >= 17 and h >= 80:
        roi = ref[y:y + h, x:x + w]
        roi = cv2.resize(roi, (80, 150))
        letters.append(roi)

num_cols = 3
num_rows = 3
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5))

# Iterate through images and display them
for i, img in enumerate(letters):
    row = i // num_cols
    col = i % num_cols

    axes[row, col].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[row, col].axis('off')

fig.delaxes(axes[2, 2])

plt.tight_layout()

# Save the combined image
plt.savefig('combined_test_images.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.show()
