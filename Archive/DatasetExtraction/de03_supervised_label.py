import os
import pandas as pd
import numpy as np
import time as time
import matplotlib.pyplot as plt

INPUT_REPORT = '../DATASET/face.csv'


def main():
    df = pd.read_csv(INPUT_REPORT)
    df = df.sort_values(by=['x', 'y'])

    df['label'] = np.nan

    for i in df.index:
        if df['x'][i] < 1450 and df['y'][i] < 500:
            df['label'][i] = 'Stanley'
        else:
            df['label'][i] = 'Edward'

    print(df.head(100))

    print(df['label'].unique())

    df.to_csv('face_label.csv', header=True, index=False)


if __name__ == '__main__':
    main()
