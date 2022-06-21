# Source code file including dictionaries with augmentation layers, TensorFlow models, and rating transformers functions

from tensorflow.keras.layers import RandomBrightness, RandomContrast, RandomCrop, RandomFlip, RandomHeight, RandomRotation, RandomTranslation, RandomWidth, RandomZoom
from tensorflow.keras.applications import inception_v3, mobilenet, vgg16, resnet

from rating_transformers import *
from losses import earth_mover_loss

# Augmentation layers for augmentation experiments
LAYERS_DICT = {
    'brightness': RandomBrightness,
    'contrast': RandomContrast,
    'crop': RandomCrop,
    'flip': RandomFlip,
    'height': RandomHeight,
    'rotation': RandomRotation,
    'translation': RandomTranslation,
    'width': RandomWidth,
    'zoom': RandomZoom
}

# Base models for augmentation experiments. Each dict entry includes the function to generate the model,
# together with its input shape and its preprocessing function
MODELS_DICT = {
    'mobilenet': (mobilenet.MobileNet, (224,224,3), mobilenet.preprocess_input),
    'inception': (inception_v3.InceptionV3, (299, 299, 3), inception_v3.preprocess_input),
    'vgg16': (vgg16.VGG16, (224, 224, 3), vgg16.preprocess_input),
    'resnet': (resnet.ResNet50, (224, 224, 3), resnet.preprocess_input)
}

# Rating transform functions. Each dict entry includes the transfomer function to change the format of the
# model, the loss function to be used with each format, the activation function of the final layer and the number of output neurons
TRANSFORMERS_DICT = {
    'distribution': (distribution_transform, earth_mover_loss, 'softmax', 10),
    'mean': (mean_transform, 'mean_squared_error', 'linear', 1),
    'binary': (binary_transform, 'binary_crossentropy', 'softmax', 1),
    'weights': (weights_transform, 'categorical_crossentropy', 'softmax', 2)
}