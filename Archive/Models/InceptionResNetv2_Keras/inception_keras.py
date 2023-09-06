'''
**Source** SEMI-RAW
https://medium.com/@sabarishsabarish244/face-recognition-using-transfer-learning-9da98ff0ca4f
'''

from keras.applications import InceptionResNetV2

# model needs 224x224 pixels as input
img_rows, img_cols = 224, 224

# Re-loads the model without the top or FC layers
model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, 3))

# Here we freeze the last 4 layers
# Layers are set to trainable as True by default
for layer in model.layers:
    layer.trainable = False

    # Let's print our layers
for (i, layer) in enumerate(model.layers):
    print(str(i) + " " + layer.__class__.__name__, layer.trainable)
