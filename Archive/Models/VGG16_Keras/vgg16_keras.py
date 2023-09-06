'''
**Source**
https://medium.com/analytics-vidhya/face-recognition-using-transfer-learning-and-vgg16-cf4de57b9154
https://medium.com/@p.rajeshbabu6666/face-recognition-using-transfer-learning-64de34de68a6
'''

from keras.applications import vgg16

img_rows, img_cols = 224, 224

vgg = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, 3))

# Transfer learning by freeze last 4 layers
for layer in vgg.layers:
    layer.trainable = False

# Printing layers so far
for (i, layer) in enumerate(vgg.layers):
    print(str(i) + " " + layer.__class__.__name__, layer.trainable)
