import os
import pandas as pd
import shutil


def main():
    # Video path location
    CATEGORY = 'V3'
    # Report data extraction
    REPORT = f'../DATASET/face_{CATEGORY}.csv'

    # Load your CSV data into a pandas DataFrame
    df = pd.read_csv(REPORT)  # Replace with the actual path to your CSV file

    # Define the source directory where your unlabeled images are located
    source_directory = '../DATASET/extractedFace/Video3/Archive Video3'
    destination_directory = '../DATASET/extractedFace/Video3/cluster'

    print(df.head())

    # Iterate through the DataFrame rows
    for index, row in df.iterrows():
        cluster = str(row['cluster'])  # Convert cluster value to string
        image_index = index
        image_filename = f'face_{CATEGORY}_{image_index}.jpg'  # Adjust the filename format if needed

        # Create a directory for the cluster if it doesn't exist
        cluster_directory = os.path.join(destination_directory, cluster)
        os.makedirs(cluster_directory, exist_ok=True)

        # Define the new filename for the image
        new_image_filename = f'{cluster}_{image_index}.jpg'

        # Construct the source and destination paths
        source_path = os.path.join(source_directory, image_filename)
        destination_path = os.path.join(cluster_directory, new_image_filename)

        if not os.path.exists(source_path):
            # Create a directory for the cluster if it doesn't exist
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)

        # Copy the image to the cluster folder and rename it
        shutil.copy(source_path, destination_path)  # Use shutil.move() to move instead of copy if desired


if __name__ == '__main__':
    main()
