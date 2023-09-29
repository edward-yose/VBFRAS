import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from keras.utils import to_categorical

from keras.applications.vgg16 import VGG16


def main():
    IMG_DIR = '../../../DATASET/DATASET/labeledFaceNearBalance'
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

    # Normalize pixel values to between 0 and 1
    x_train, x_test = X_train / 255.0, X_test / 255.0

    # to categorical one-hot encoding
    y_trains = to_categorical(y_train)
    y_tests = to_categorical(y_test)

    model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

    for layer in model.layers:
        layer.trainable = False

    print(model.summary())

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(X_train, y_trains,
                        batch_size=8,
                        verbose=1,
                        epochs=50,
                        validation_data=(X_test, y_tests),
                        shuffle=False
                        )

    model.save('lul_lr_50eps.h5')

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

    from keras.models import load_model
    # load model
    model = load_model('lul_lr_50eps.h5')

    _, acc = model.evaluate(X_test, y_test)
    print("Accuracy = ", (acc * 100.0), "%")

    mythreshold = 0.8
    from sklearn.metrics import confusion_matrix

    y_pred = (model.predict(X_test) >= mythreshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # ROC
    from sklearn.metrics import roc_curve
    y_preds = model.predict(X_test).ravel()

    fpr, tpr, thresholds = roc_curve(y_test, y_preds)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'y--')
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.show()

    import pandas as pd
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'thresholds': pd.Series(thresholds, index=i)})
    ideal_roc_thresh = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]  # Locate the point where the value is close to 0
    print("Ideal threshold is: ", ideal_roc_thresh['thresholds'])

    from sklearn.metrics import auc
    auc_value = auc(fpr, tpr)
    print("Area under curve, AUC = ", auc_value)


if __name__ == '__main__':
    main()
