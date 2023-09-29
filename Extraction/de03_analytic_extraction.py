import os
import time
import pandas as pd
import random
import numpy as np

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

# Video path location
CATEGORY = 'V1'
# Boundary face capture
DIRECTORY = '../DATASET/extractedFace/Extracted/Video 3'
# Report data extraction
REPORT = f'../DATASET/face_{CATEGORY}.csv'


def group_size():
    df = pd.read_csv(REPORT)

    # Assuming you have a DataFrame named 'df' with a column named 'y'
    # Replace 'df' and 'y' with your actual DataFrame and column names

    # Group the DataFrame by the 'y' column and calculate the sum
    sums = df['w'].value_counts()

    # Create a bar chart
    plt.figure(figsize=(22, 10))
    plt.bar(sums.index, sums.values)

    # Add labels and a title
    plt.xlabel('Unique Values in Column "w"')
    plt.ylabel('Sum of Occurrences')
    plt.title('size figure distribution')

    plt.xticks(range(0, 160, 5))

    # Show the plot
    plt.show()




def lookup_cluster(df, idx):
    fdf = df[df['cluster'] == idx]
    fdf = fdf.sort_values(by='name_face', ascending=True)
    fdf.head(10)
    return fdf


def prelabel(df):
    cond = [
        df['cluster'] == 0,
        df['cluster'] == 1,
        df['cluster'] == 2,
        df['cluster'] == 3,
        df['cluster'] == 4,
        df['cluster'] == 5,
        df['cluster'] == 6,
        df['cluster'] == 7,
        df['cluster'] == 8,
        df['cluster'] == 9,
        df['cluster'] == 10,
        df['cluster'] == 11,
    ]
    choice = [
        'Edward',
        'G1',
        'Edward',
        'Arief',
        'G1',
        'Vincent_',
        'Whisnu_',
        'Edward',
        'Unknown',
        'David',
        'Unknown',
        'Vincent'
    ]

    df['label'] = np.select(cond, choice, default='N/A')
    return df


def group_blob():
    df = pd.read_csv(REPORT)
    df = df.sort_values(by=['x', 'y'])

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Set the axis limits
    ax.set_xlim(0, 1920)
    ax.set_ylim(0, 1280)

    # Invert the y-axis so that (0,0) is at the top left
    ax.invert_yaxis()

    # Plot the data
    ax.scatter(df.x, df.y, marker="o")

    # Show the plot
    plt.show()

    # Assuming df is your DataFrame with columns 'x' and 'y'
    # Sample data
    data = df[['x', 'y']].values

    # Specify the number of clusters (k)
    k = 12

    # Fit the KMeans model
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(data)

    # Get cluster centers and labels
    cluster_centers = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_

    # Add cluster labels to the DataFrame
    df['cluster'] = cluster_labels

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Set the axis limits
    ax.set_xlim(0, 1920)
    ax.set_ylim(0, 1280)

    # Invert the y-axis so that (0,0) is at the top left
    ax.invert_yaxis()

    # Plot the data points and cluster centers
    plt.scatter(df['x'], df['y'], c=cluster_labels, cmap='viridis', s=50)
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', s=200, label='Cluster Centers')
    plt.legend()
    plt.show()

    # Display the DataFrame with cluster labels
    print(df)

    idx = 0
    df = lookup_cluster(df, idx)

    df['label'] = np.nan

    # Applied Labelling
    # df =  prelabel(df)


if __name__ == '__main__':
    group_size()
    # group_blob()
