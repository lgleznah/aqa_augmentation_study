# Utilities for generating a TensorFlow Dataset object for various image datasets
# This code has been adapted from code from ferrubio, in the private repository: https://github.com/ferrubio/AQA-framework

import valid_parameters_dicts as vpd

import pandas as pd
import numpy as np
import os

import tensorflow as tf

from sklearn.model_selection import train_test_split

def image_parser_generator(input_shape):
    def parse_image(filename, label):
        image = tf.io.read_file(filename)
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, input_shape)
        return image, label
    
    return parse_image

def get_class_weights(info_folder, label_column):
    info_df = pd.read_csv(f'{info_folder}/info.csv', index_col=0)

    counts = info_df.iloc[:,label_column].value_counts().to_dict()
    total_examples = len(info_df)

    scaled_counts = {}
    for label, count in counts.items():
        scaled_counts[label] = (1 / count) * (total_examples / 2)

def generate_dataset_with_splits(dataset_specs, label_columns, output_format, input_shape, batch_size, test_split, val_split, random_seed=1000, labels_only=False):
    
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
    image_parser = image_parser_generator(input_shape[:2])
    train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels)).shuffle(1024, seed=random_seed).map(image_parser).batch(batch_size).prefetch(-1)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_image_paths, val_labels)).map(image_parser).batch(batch_size).prefetch(-1)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_image_paths, test_labels)).map(image_parser).batch(batch_size).prefetch(-1)

    return train_dataset, val_dataset, test_dataset