# Utilities for generating a TensorFlow Dataset object for various image datasets
# This code has been adapted from code from ferrubio, in the private repository: https://github.com/ferrubio/AQA-framework

import valid_parameters_dicts as vpd

import pandas as pd
import numpy as np
import os
import PIL.Image
import random
import tensorflow as tf

from sklearn.model_selection import train_test_split
from autoaugment_policies import ImageNetPolicy

def image_parser_generator(input_shape, augment_rate):
    def parse_image(filename, label):
        image = tf.io.read_file(filename)
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, input_shape)
        return image, label
    
    return parse_image

def image_parser_generator_autoaugment(input_shape, augment_rate):
    def autoaugment(filename):
        pil_img = PIL.Image.open(filename.numpy()).convert('RGB')
        if random.random() < augment_rate:
            pil_img = ImageNetPolicy(fillcolor=(256, 256, 256))(pil_img)
        pil_img = pil_img.resize(input_shape)
        img = np.array(pil_img)
        return img
    
    def parse_image(filename, label):
        # Apply ImageNet auto-augment policy
        image = tf.py_function(autoaugment, [filename], tf.uint8)

        return image, label
    
    return parse_image

def get_class_weights(info_folder, label_column):
    info_df = pd.read_csv(f'{info_folder}/info.csv', index_col=0)

    counts = info_df.iloc[:,label_column].value_counts().to_dict()
    total_examples = len(info_df)

    scaled_counts = {}
    for label, count in counts.items():
        scaled_counts[label] = (1 / count) * (total_examples / 2)

def generate_dataset_with_splits(dataset_specs, label_columns, output_format, input_shape, batch_size, test_split, val_split, random_seed=1000, labels_only=False, autoaugment=False, autoaugment_rate=0):
    
    # Fetch paths from the enviroment required for loading AVA
    images_folder = os.environ[dataset_specs[1]]
    info_folder = os.environ[dataset_specs[0]]

    # Load ava_info DataFrame, and build image paths from IDs
    info_csv = pd.read_csv(f'{info_folder}/info.csv', index_col=0)
    file_list = np.array([images_folder + f'/{i}' for i in np.array(info_csv.loc[:,'id'])])

    # Fetch initial labels (original ratings) from the DataFrame, and transform them
    labels = vpd.TRANSFORMERS_DICT[output_format][0](np.array(info_csv.iloc[:,label_columns]))

    # Perform training, validation and testing splits, Stratify only for tenclass problems
    # Fetch partitions manually if these are specified in the info CSV file
    if ('partition' in info_csv.columns):
        partitions = info_csv['partition'].to_numpy()
        train_image_paths, train_labels = file_list[partitions == 0], labels[partitions == 0]
        val_image_paths, val_labels = file_list[partitions == 1], labels[partitions == 1]
        test_image_paths, test_labels = file_list[partitions == 2], labels[partitions == 2]

    else:
        stratify_train_test = None if output_format not in ['tenclass', 'ovr-binary', 'binary'] else labels
        train_image_paths, test_image_paths, train_labels, test_labels = train_test_split(file_list, labels, test_size = test_split, random_state = random_seed, stratify=stratify_train_test)

        stratify_train_val = None if output_format not in ['tenclass', 'ovr-binary', 'binary'] else train_labels
        train_image_paths, val_image_paths, train_labels, val_labels = train_test_split(train_image_paths, train_labels, test_size = val_split, random_state = random_seed, stratify=stratify_train_val)

    if (labels_only):
        return train_labels, val_labels, test_labels

    # Generate datasets
    parser_func_train = image_parser_generator if not autoaugment else image_parser_generator_autoaugment
    parser_func_valtest = image_parser_generator
    image_parser_train = parser_func_train(input_shape[:2], autoaugment_rate)
    image_parser_valtest = parser_func_valtest(input_shape[:2], autoaugment_rate)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels)).shuffle(10000, seed=random_seed).map(image_parser_train).batch(batch_size).prefetch(-1)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_image_paths, val_labels)).map(image_parser_valtest).batch(batch_size).prefetch(-1)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_image_paths, test_labels)).map(image_parser_valtest).batch(batch_size).prefetch(-1)

    return train_dataset, val_dataset, test_dataset