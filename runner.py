from pickletools import optimize
from experiment_parser import parse_experiment_file
from augmented_model_generator import get_augmented_model_and_preprocess
from dataset_generator import generate_dataset_with_splits

import layers_models_transforms_dicts as lmd

import os

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

if __name__ == '__main__':
    # Parse specified experiment file
    experiment_dict = parse_experiment_file('test_file.yaml')

    # TODO: parametrize experiment to execute
    exp = experiment_dict['exps'][0]

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
    model_with_augmentation.compile(loss=loss_function, optimizer=Adam(learning_rate=1e-05))

    # Create model checkpoint
    checkpoints_dir = '../augmentation-chkpt'
    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)
    best_checkpoint = ModelCheckpoint(filepath=os.path.join(checkpoints_dir,f"{exp['name']}_bestmodel.h5"), monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    # Fit model!
    history = model_with_augmentation.fit(train_dataset, validation_data=val_dataset, epochs=20, callbacks=[best_checkpoint])