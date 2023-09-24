import cv2
import numpy as np
import os
import random

dataset_dir = '/Users/sara/Desktop/part/computervision_task3/'
os.chdir(dataset_dir)
backgrounds = os.listdir(dataset_dir+'/backgrounds')
plates = os.listdir(dataset_dir+'/plates')
number_of_images = 1
size_of_result = (224, 224)

for item in range(number_of_images):

    random_background_dir = random.choice(backgrounds)
    random_plate_dir = random.choice(plates)
    background_images = cv2.imread('backgrounds/'+random_background_dir)
    plate_images = cv2.imread('plates/'+random_plate_dir)

    # factors
    scaling_factor = random.uniform(0.2, 1)
    angle = random.randint(-60, 60)

    # Resize the smaller image to fit within the dimensions of the larger image
    if plate_images.shape[0] >= background_images.shape[0] or plate_images.shape[1] >= background_images.shape[1]:
        plate_images = cv2.resize(plate_images, (background_images.shape[1], background_images.shape[0]))

    # resize
    resized_plate_images = cv2.resize(plate_images, None, fx=scaling_factor, fy=scaling_factor)

    height, width, _ = resized_plate_images.shape
    center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

    # Calculate the new dimensions
    cosine = np.abs(rotation_matrix[0, 0])
    sine = np.abs(rotation_matrix[0, 1])
    new_width = int((height * sine) + (width * cosine))
    new_height = int((height * cosine) + (width * sine))

    # Adjust the rotation matrix (centering)
    rotation_matrix[0, 2] += (new_width / 2) - (width / 2)
    rotation_matrix[1, 2] += (new_height / 2) - (height / 2)

    # Apply the rotation to the image while keeping all of it within the canvas
    rotated_plate_images = cv2.warpAffine(resized_plate_images, rotation_matrix, (new_width, new_height))

    plate_coordinates = [(0, 0), (width, 0), (width, height), (0, height)]
    lis = list(plate_coordinates)
    for i, coor in enumerate(plate_coordinates):
        v = [coor[0], coor[1], 1]
        calc = np.dot(rotation_matrix, v)
        lis[i] = (calc[0], calc[1])

    # Resize the smaller image to fit within the dimensions of the larger image
    if rotated_plate_images.shape[0] >= background_images.shape[0] or rotated_plate_images.shape[1] >= background_images.shape[1]:
        rotated_plate_images = cv2.resize(rotated_plate_images, None, fx=0.1, fy=0.1)
        for i, coor in enumerate(lis):
            lis[i] = (lis[i][0]/10, lis[i][1]/10)

    gray_foreground = cv2.cvtColor(rotated_plate_images, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_foreground, 1, 255, cv2.THRESH_BINARY)
    # Invert the mask (to use as a filter)
    mask_inv = cv2.bitwise_not(mask)
    height2, width2, _ = rotated_plate_images.shape

    y = random.randint(0, background_images.shape[0] - rotated_plate_images.shape[0])
    x = random.randint(0, background_images.shape[1] - rotated_plate_images.shape[1])

    roi = background_images[y:height2+y, x:x+width2]
    # Apply the mask to the foreground and background
    foreground = cv2.bitwise_and(rotated_plate_images, rotated_plate_images, mask=mask)
    background = cv2.bitwise_and(roi, roi, mask=mask_inv)
    # Combine the foreground and background to get the result
    result = background + foreground
    # Place the result back into the original background
    background_images[y:y+height2, x:x+width2] = result

    corner_coordinates = [(int(i) + x, int(j) + y) for i, j in lis]

    resized_img = cv2.resize(background_images, size_of_result)
    new_coordinates = []
    for i, j in corner_coordinates:
        i = (i * size_of_result[1]) // background_images.shape[1]
        j = (j * size_of_result[0]) // background_images.shape[0]
        if i >= size_of_result[1]:
            i = size_of_result[1]
        if j >= size_of_result[0]:
            j = size_of_result[0]
        if i < 0:
            i = 0
        if j < 0:
            j = 0
        new_coordinates.append((i, j))

    filename = os.path.join(dataset_dir, "merged_data/txtFiles/")
    with open(filename + f'coordinates{item + 1}.txt', 'w') as file:
        for i, j in new_coordinates:
            file.write(f"{int(i)}, {int(j)}\n")

    image_filename = os.path.join(dataset_dir, f"merged_data/images/merged_image{item + 1}.jpg")
    cv2.imwrite(image_filename, resized_img)
