import os
import shutil


def main(source_folder, destination_folder):
    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Walk through the source folder and its subfolders
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            # Check if the file is an image (you can add more image extensions)
            if file.endswith(('.jpg', '.png', '.jpeg', '.gif')):
                # Construct the source and destination paths for the file
                source_path = os.path.join(root, file)
                destination_path = os.path.join(destination_folder, file)

                # Move the image to the destination folder
                shutil.move(source_path, destination_path)

    print(f"All images have been moved to {destination_folder}.")


if __name__ == '__main__':
    SOURCE = '../../../DATASET/LFW'
    DESTINATION = '../../../DATASET/LFW_unlabelled/data'
    main(SOURCE, DESTINATION)
