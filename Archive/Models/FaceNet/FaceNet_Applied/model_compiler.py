from architecture import *

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import normalize
from keras.models import Sequential


def main():
    IMG_DIR = '../../../../DATASET/DATASET/labeledFaceNearBalance'
    SIZE = 160

    dataset = []
    label = []

    for face_names in os.listdir(IMG_DIR):
        person_dir = os.path.join(IMG_DIR, face_names)

        # Check if it's a directory
        if os.path.isdir(person_dir):
            # Iterate through the images in the person's folder
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)

                # Read the image using OpenCV
                image = cv2.imread(image_path)

                # Resize the image to the specified size
                image = cv2.resize(image, (SIZE, SIZE))

                # Append the resized image to the dataset
                dataset.append(image)

                # Append the label (person's name) to the labels list
                label.append(face_names)

    # Create a LabelEncoder instance
    label_encoder = LabelEncoder()
    label = label_encoder.fit_transform(label)

    # Convert the dataset and labels to NumPy arrays for further processing
    dataset = np.array(dataset)
    label = np.array(label)

    print(dataset, label)

    print("Dataset size is:", dataset.shape)
    print("Label size is: ", label.shape)

    # # # --- Model Test Train Splits --- # # #
    X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=42)
    X_train = normalize(X_train, axis=1)
    X_test = normalize(X_test, axis=1)

    model = InceptionResNetV1()

    model.trainable = False

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())

    history = model.fit(X_train, y_train,
                        batch_size=8,
                        verbose=1,
                        epochs=50,
                        validation_data=(X_test, y_test),
                        shuffle=False
                        )

    model.save('../../../../../Models/Saved/facenet_lr_50eps.h5')

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.plot(epochs, acc, 'y', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    n = 23  # Select the index of image to be loaded for testing
    img = X_test[n]
    plt.imshow(img)
    input_img = np.expand_dims(img, axis=0)
    print("The prediction for this image is: ", model.predict(input_img))
    print("The actual label for this image is: ", y_test[n])


if __name__ == '__main__':
    main()

    # TODO IMPORTANT | ValueError: Shapes (None, 1) and (None, 128) are incompatible
    # go look for using encodings data i guess
