import os
import random
import shutil

# Set your image directory
IMG_DIR = '../DATASET/DATASET/labeledFaceNearBalance/Vincent'

# Define the percentage of images to purge
purge_percentage = 22

# Get the list of all image files in the directory
all_images = [f for f in os.listdir(IMG_DIR) if os.path.isfile(os.path.join(IMG_DIR, f))]

# Calculate the number of images to keep and purge
total_images = len(all_images)
images_to_keep = int((100 - purge_percentage) / 100 * total_images)
images_to_purge = total_images - images_to_keep

# Randomly select images to purge
images_to_purge_list = random.sample(all_images, images_to_purge)

# Create a directory to move purged images to (optional)
purged_dir = os.path.join(IMG_DIR, 'purged_images')
os.makedirs(purged_dir, exist_ok=True)

# Move the purged images to the purged directory (optional)
for image in images_to_purge_list:
    image_path = os.path.join(IMG_DIR, image)
    shutil.move(image_path, os.path.join(purged_dir, image))

print(f"Purged {images_to_purge} out of {total_images} images randomly.")
