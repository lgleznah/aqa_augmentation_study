# Source code file including dictionaries with augmentation layers, TensorFlow models, and rating transformers functions

from tensorflow.keras.layers import RandomBrightness, RandomContrast, RandomCrop, RandomFlip, RandomHeight, RandomRotation, RandomTranslation, RandomWidth, RandomZoom
from tensorflow.keras.applications import inception_v3, mobilenet, vgg16, resnet

from rating_transformers import *
from losses import earth_mover_loss

from ast import literal_eval

# Accepted data types in experiment files, and how to parse them
TYPE_CONVERTERS = {
    'int': int,
    'float': float,
    'tuple': literal_eval,
    'string': str
}

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
# model, the CSV columns to use for each transformation, the loss function to be used with each format, 
# the activation function of the final layer, the number of output neurons, and the monitoring metric
TRANSFORMERS_DICT = {
    'distribution': (distribution_transform, slice(1,11), earth_mover_loss, 'softmax', 10, earth_mover_loss),
    'mean': (mean_transform, slice(1, 11), 'mean_squared_error', 'linear', 1, 'mse'),
    'binary': (binary_transform, slice(1, 11), 'binary_crossentropy', 'sigmoid', 1, 'accuracy'),
    'ovr-binary': (ovr_binary, 1, 'binary_crossentropy', 'sigmoid', 1, 'accuracy'),
    'weights': (weights_transform, slice(1, 11), 'categorical_crossentropy', 'softmax', 2, 'categorical_accuracy'),
    'tenclass': (tenclass_transform, 1, 'sparse_categorical_crossentropy', 'softmax', 10, 'sparse_categorical_accuracy')
}

# Accepted datasets in the experiment files. Each dict entry specifies the environment variables containing the
# paths to their CSV description file and the base path to their images, respectively
DATASETS_DICT = {
    'ava': ('AVA_info_folder', 'AVA_images_folder'),
    'ava-small': ('AVA_small_info_folder', 'AVA_images_folder'),
    'photozilla': ('Photozilla_info_folder', 'Photozilla_images_folder'),
    'photozilla-unbalanced': ('Photozilla_unbalanced_info_folder', 'Photozilla_unbalanced_images_folder'),
    'photozilla-ovr-aerial': ('Photozilla_ovr_aerial_info_folder', 'Photozilla_images_folder'),
    'photozilla-ovr-architecture': ('Photozilla_ovr_architecture_info_folder', 'Photozilla_images_folder'),
    'photozilla-ovr-event': ('Photozilla_ovr_event_info_folder', 'Photozilla_images_folder'),
    'photozilla-ovr-fashion': ('Photozilla_ovr_fashion_info_folder', 'Photozilla_images_folder'),
    'photozilla-ovr-food': ('Photozilla_ovr_food_info_folder', 'Photozilla_images_folder'),
    'photozilla-ovr-nature': ('Photozilla_ovr_nature_info_folder', 'Photozilla_images_folder'),
    'photozilla-ovr-sports': ('Photozilla_ovr_sports_info_folder', 'Photozilla_images_folder'),
    'photozilla-ovr-street': ('Photozilla_ovr_street_info_folder', 'Photozilla_images_folder'),
    'photozilla-ovr-wedding': ('Photozilla_ovr_wedding_info_folder', 'Photozilla_images_folder'),
    'photozilla-ovr-wildlife': ('Photozilla_ovr_wildlife_info_folder', 'Photozilla_images_folder'),
    'ps-ovr-aerial': ('ps_ovr_aerial_info_folder', 'Photozilla_images_folder'),
}