# plate-detection
The purpose of this project is to detect 4 corner coordinates of a license plate in an image, and then segment the parts of the license plate.

## Data Generation
In this phase of the project, there were 100 license plate images and 30 background images. I had to generate a large number of images so that a license plate could be placed randomly on any background image. In order to bring the generated images closer to the real data, the license plate images are placed in the background image after applying some geometric transformations.

<img src="https://github.com/sararajabzadeh/plate-detection/blob/main/images/merged_image1%20copy.jpg">


## Object Localization
To find the corners of the license plate in the generated data, I had to use mobilenetv2 with imagenet weights.


## Plate Detection and Saving
Finally, I had to extract the localized image, warp it, and segment each part of the license plate.

<img src="https://github.com/sararajabzadeh/plate-detection/blob/main/images/denoised_detected_plate%20copy.jpg" width="300">

<img src="https://github.com/sararajabzadeh/plate-detection/blob/main/images/combined_test_images.png" width="500">
