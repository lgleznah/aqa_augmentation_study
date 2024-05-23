from experiment_parser import parse_experiment_file
from augmented_model_generator import get_augmented_model
from dataset_generator import generate_dataset_with_splits, get_class_weights

import valid_parameters_dicts as vpd

import os
import sys
import json
import random

import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

def set_seed(seed: int = 42, contrast_experiment: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
    # When running on the CuDNN backend, two further options must be set
    # As of TF 2.10, there is no deterministic implementation for RandomContrast, so avoid setting
    # these variables in this case.

    if not contrast_experiment:
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

def main():
    '''
    Run training on the specified experiment in the specified experiments file.
    Accepted command-line arguments are:

        - First argument: the index of the experiment to run
        - Second argument: the experiments file where the specified experiment is located
        - Third argument: whether to rerun the experiment in case it was already completed
    '''
    experiment_index = int(sys.argv[1])
    experiment_file = os.path.join(os.environ['AQA_AUGMENT_EXPERIMENTS_PATH'], f'{sys.argv[2]}.yaml')
    ignore_completed = sys.argv[3].lower() == 'true'

    # Parse specified experiment file
    experiment_dict = parse_experiment_file(experiment_file)

    exp = experiment_dict['exps'][experiment_index]
    seed = experiment_dict['seed']
    has_contrast_exp = 'contrast' in [layer['layer'] for layer in exp['layers']]
    set_seed(seed, has_contrast_exp)

    # Ignore experiment if it already exists
    histories_dir = f'./augmentation-hist/{os.path.splitext(os.path.basename(experiment_file))[0]}'
    if os.path.exists(os.path.join(histories_dir,f"{exp['name']}_history.json")) and ignore_completed:
        print('Experiment already exists! Exiting...')
        return

    # Generate both the model with the specified augmentation techniques and its preprocessing function
    model_with_augmentation = get_augmented_model(exp, seed)

    # Generate training, validation and test datasets
    output_format = exp['output_format']
    batch_size = exp['batch_size']
    test_split = experiment_dict['test_split']
    val_split = experiment_dict['val_split']
    input_shape = vpd.MODELS_DICT[exp['base_model']][1]
    dataset_specs = vpd.DATASETS_DICT[experiment_dict['dataset']]
    label_columns = vpd.TRANSFORMERS_DICT[output_format][1]
    is_autoaugment = exp['layers'] and ('autoaugment' == exp['layers'][0]['layer'])
    autoaugment_rate = 0 if not is_autoaugment else exp['layers'][0]['rate']
    train_dataset, val_dataset, _ = generate_dataset_with_splits(dataset_specs, label_columns, output_format, input_shape, batch_size, test_split, val_split, random_seed=seed, autoaugment=is_autoaugment, autoaugment_rate=autoaugment_rate)

    # Show model summary and compile model
    print(model_with_augmentation.summary())

    _, _, loss_function, _, _, _ = vpd.TRANSFORMERS_DICT[exp['output_format']]
    lr = experiment_dict['lr']
    metric = [vpd.TRANSFORMERS_DICT[output_format][5]]
    model_with_augmentation.compile(loss=loss_function, optimizer=Adam(learning_rate=lr, decay=1e-8), metrics=metric)

    # Create model checkpoint
    checkpoints_dir = f'./augmentation-chkpt/{os.path.splitext(os.path.basename(experiment_file))[0]}'
    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)
    best_checkpoint = ModelCheckpoint(filepath=os.path.join(checkpoints_dir,f"{exp['name']}_bestmodel.h5"), monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weigts_only=True)

    # Create history folder
    histories_dir = f'./augmentation-hist/{os.path.splitext(os.path.basename(experiment_file))[0]}'
    if not os.path.exists(histories_dir):
        os.mkdir(histories_dir)

    # Set callbacks
    callbacks = [best_checkpoint]

    if experiment_dict['use_plateau']:
        plateau = ReduceLROnPlateau(patience=5, factor=0.5)
        callbacks.append(plateau)

    # Load class weights
    weights = None
    if experiment_dict['weight_classes']:
        info_csv_path = os.environ[dataset_specs[0]]
        weights = get_class_weights(info_csv_path, label_columns)

    # Fit model!
    epochs = experiment_dict['epochs']
    history = model_with_augmentation.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=callbacks, class_weight=weights)

    # Save history file
    with open(os.path.join(histories_dir, f"{exp['name']}_history.json"), 'w') as f:
        json.dump(history.history, f)

if __name__ == '__main__':
    main()