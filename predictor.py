from experiment_parser import parse_experiment_file
from augmented_model_generator import get_augmented_model_and_preprocess
from dataset_generator import generate_dataset_with_splits

import valid_parameters_dicts as vpd

import os
import sys
import json

import tensorflow as tf
import numpy as np

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

if __name__ == '__main__':

    experiment_index = int(sys.argv[1])
    experiment_file = os.path.join(os.environ['AQA_AUGMENT_EXPERIMENTS_PATH'], f'{sys.argv[2]}.yaml')

    # Parse specified experiment file
    experiment_dict = parse_experiment_file(experiment_file)

    exp = experiment_dict['exps'][experiment_index]
    seed = experiment_dict['seed']

    # Generate model preprocessing function
    model_with_augmentation, preprocess_func = get_augmented_model_and_preprocess(exp, seed)

    # Generate training, validation and test datasets
    output_format = exp['output_format']
    batch_size = exp['batch_size']
    input_shape = vpd.MODELS_DICT[exp['base_model']][1]
    dataset_specs = vpd.DATASETS_DICT[experiment_dict['dataset']]
    label_columns = vpd.TRANSFORMERS_DICT[output_format][1]
    _, _, test_dataset = generate_dataset_with_splits(dataset_specs, label_columns, output_format, preprocess_func, input_shape, batch_size, random_seed = seed)

    checkpoints_dir = f'./augmentation-chkpt/{os.path.splitext(os.path.basename(experiment_file))[0]}'
    model_with_augmentation.load_weights(os.path.join(checkpoints_dir, f"{exp['name']}_bestmodel.h5"))

    print(model_with_augmentation.summary())

    # Run model inference
    predictions = model_with_augmentation.predict(test_dataset)

    # Save predictions for further analysis
    predictions_dir = f'./augmentation-preds/{os.path.splitext(os.path.basename(experiment_file))[0]}'
    if not os.path.exists(predictions_dir):
        os.mkdir(predictions_dir)

    np.save(os.path.join(predictions_dir, f"{exp['name']}_predictions.npy"), predictions)