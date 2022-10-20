# Utilities for generating a TensorFlow Dataset object for the AVA aesthetics dataset
# This code has been adapted from code from ferrubio, in the private repository: https://github.com/ferrubio/AQA-framework

import valid_parameters_dicts as vpd

import pandas as pd
import numpy as np
import os
import gzip

import tensorflow as tf

from sklearn.model_selection import train_test_split

def image_parser_generator(input_shape, preprocess_function):
    def parse_image(filename, label):
        image = tf.io.read_file(filename)
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, input_shape)
        return preprocess_function(image), label
    
    return parse_image

def generate_dataset_with_splits(output_format, preprocessing_function, input_shape, batch_size, test_split=0.08, val_split=0.2, random_seed=1000, labels_only=False):
    
    # Fetch paths from the enviroment required for loading AVA
    ava_images_folder = os.environ['AVA_images_folder']
    ava_info_folder = os.environ['AVA_info_folder']

    # Load ava_info DataFrame
    file_pickle = gzip.open(f'{ava_info_folder}/AVA_info.pklz','rb',2)
    ava_info = pd.read_pickle(file_pickle, compression=None)
    ava_info.loc[:,'id'] = ava_info['id'].apply(str)
    ava_info.sort_values(['id'],inplace=True)
    ava_info.reset_index(inplace=True,drop=True)

    # TODO: change dataframe to remove these images
    bad_idxs = ['729377', '179118', '230701', '277832', '371434', '440774']
    ava_info = ava_info[~ava_info['id'].isin(bad_idxs)]

    file_list = np.array([ava_images_folder + '/{:}.jpg'.format(i) for i in np.array(ava_info.loc[:,'id'])])

    # Fetch initial labels (original ratings) from the DataFrame, and transform them
    labels = vpd.TRANSFORMERS_DICT[output_format][0](np.array(ava_info.iloc[:,2:12]))

    # Perform training, validation and testing splits
    train_image_paths, test_image_paths, train_labels, test_labels = train_test_split(file_list, labels, test_size = test_split, random_state = random_seed)
    train_image_paths, val_image_paths, train_labels, val_labels = train_test_split(train_image_paths, train_labels, test_size = val_split, random_state = random_seed)

    if (labels_only):
        return train_labels, val_labels, test_labels

    # Generate datasets
    image_parser = image_parser_generator(input_shape[:2], preprocessing_function)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels)).shuffle(1024).map(image_parser).batch(batch_size).prefetch(-1)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_image_paths, val_labels)).map(image_parser).batch(batch_size).prefetch(-1)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_image_paths, test_labels)).map(image_parser).batch(batch_size).prefetch(-1)

    return train_dataset, val_dataset, test_dataset