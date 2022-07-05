from experiment_parser import parse_experiment_file
from augmented_model_generator import get_augmented_model_and_preprocess
from dataset_generator import generate_dataset_with_splits

import layers_models_transforms_dicts as lmd

import os
import sys
import json

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

if __name__ == '__main__':

    experiment_index = int(sys.argv[1])
    experiment_file = sys.argv[2]

    # Parse specified experiment file
    experiment_dict = parse_experiment_file(experiment_file)

    exp = experiment_dict['exps'][experiment_index]

    # Generate both the model with the specified augmentation techniques and its preprocessing function
    model_with_augmentation, preprocess_func = get_augmented_model_and_preprocess(exp)

    # Generate training, validation and test datasets
    output_format = exp['output_format']
    batch_size = exp['batch_size']
    input_shape = lmd.MODELS_DICT[exp['base_model']][1]
    train_dataset, val_dataset, _ = generate_dataset_with_splits(output_format, preprocess_func, input_shape, batch_size)

    # Show model summary and compile model
    print(model_with_augmentation.summary())

    _, loss_function, _, _ = lmd.TRANSFORMERS_DICT[exp['output_format']]
    model_with_augmentation.compile(loss=loss_function, optimizer=Adam(learning_rate=1e-05, decay=1e-8))

    # Create model checkpoint
    checkpoints_dir = './augmentation-chkpt'
    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)
    best_checkpoint = ModelCheckpoint(filepath=os.path.join(checkpoints_dir,f"{exp['name']}_bestmodel.h5"), monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    # Create history folder
    histories_dir = './augmentation-hist'
    if not os.path.exists(histories_dir):
        os.mkdir(histories_dir)

    # Fit model!
    history = model_with_augmentation.fit(train_dataset, validation_data=val_dataset, epochs=20, callbacks=[best_checkpoint])

    # Save history file
    with open(os.path.join(histories_dir, f"{exp['name']}_history.json"), 'w') as f:
        json.dump(history.history, f)