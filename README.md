# plate-detection
The purpose of this project is to detect 4 corner coordinates of a license plate in an image, and then segment the parts of the license plate.

## Data Generation
In this phase of the project, There were 100 licence plate images and 30 background images. I had to generate a large number of images so that a license plate can be placed randomly in a background image. In order to bring the generated images closer to the real data, the license plate images are placed in the background image after applying geometric transformations.

## Object Localization
To find the corners of the license plate in the generared data, I had to use mobilenetv2 with imagenet weights.

## Plate Detection and Saving
Finally, I had to extract the localized image, warp it, and segments each part of the license plate.
