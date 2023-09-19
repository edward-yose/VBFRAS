from keras.applications import VGG16, VGG19, InceptionResNetV2
from keras import Model


def build_vgg16(shape):
    vgg = VGG16(weights="imagenet", include_top=False, input_shape=shape)
    return Model(inputs=vgg.inputs, outputs=vgg.layers[10].output)


def build_vgg19(shape):
    vgg = VGG19(weights="imagenet", include_top=False, input_shape=shape)
    return Model(inputs=vgg.inputs, outputs=vgg.layers[10].output)


def build_inception_resnet_v2(shape):
    vgg = InceptionResNetV2(weights="imagenet", include_top=False, input_shape=shape)
    return Model(inputs=vgg.inputs, outputs=vgg.layers[10].output)


if __name__ == '__main__':
    shape = (128, 128, 3)
    # model = build_vgg16(shape)
    # model = build_vgg19(shape)
    model = build_inception_resnet_v2(shape)

    layers_info = []

    for layer in model.layers:
        layer_info = {
            "name": layer.name,
            "class_name": layer.__class__.__name__,
            "trainable": layer.trainable,
            "output_shape": layer.output_shape,
        }
        layers_info.append(layer_info)

    # Print the list of layers' information
    for layer_info in layers_info:
        print(layer_info)
