# TO FIND IMAGE SIZE

import cv2

# Load an image
image_path = './train1/66500927_1705485488446.jpg'
image = cv2.imread(image_path)

# Get dimensions
height, width, channels = image.shape
print(f"Width: {width}, Height: {height}")