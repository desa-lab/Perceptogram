import numpy as np
import cv2
from tqdm import tqdm

test_images = np.load(f'data/thingseeg2_metadata/test_images.npy').astype(np.uint8)
train_images = np.load(f'data/thingseeg2_metadata/train_images.npy').astype(np.uint8)

print(train_images.shape, test_images.shape)

# Initialize a list to store the processed images
processed_images = []

# Loop through each image
for img in tqdm(train_images, total=len(train_images)):
    # Convert the image to grayscale
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Convert the grayscale image to a 3-channel image
    grayscale_img = cv2.cvtColor(grayscale_img, cv2.COLOR_GRAY2RGB)
    
    # Append the processed image to the list
    processed_images.append(grayscale_img)

# Convert the list back to a NumPy array
processed_images = np.array(processed_images)

# Save the processed images
np.save('data/thingseeg2_metadata/train_images_grayscale.npy', processed_images)

# Initialize a list to store the processed images
processed_images = []

# Loop through each image
for img in tqdm(test_images, total=len(test_images)):
    # Convert the image to grayscale
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Convert the grayscale image to a 3-channel image
    grayscale_img = cv2.cvtColor(grayscale_img, cv2.COLOR_GRAY2RGB)
    
    # Append the processed image to the list
    processed_images.append(grayscale_img)

# Convert the list back to a NumPy array
processed_images = np.array(processed_images)

# Save the processed images
np.save('data/thingseeg2_metadata/test_images_grayscale.npy', processed_images)