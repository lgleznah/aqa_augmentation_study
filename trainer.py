from experiment_parser import parse_experiment_file
from augmented_model_generator import get_augmented_model_and_preprocess
from dataset_generator import generate_dataset_with_splits

import valid_parameters_dicts as vpd

import os
import sys
import json

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

'''
Run training on the specified experiment in the specified experiments file.
Accepted command-line arguments are:

    - First argument: the index of the experiment to run
    - Second argument: the experiments file where the specified experiment is located
    - Third argument: whether to rerun the experiment in case it was already completed
'''
def main():

    experiment_index = int(sys.argv[1])
    experiment_file = os.path.join(os.environ['AQA_AUGMENT_EXPERIMENTS_PATH'], f'{sys.argv[2]}.yaml')
    rerun_completed = sys.argv[3].lower == 'true'

    # Parse specified experiment file
    experiment_dict = parse_experiment_file(experiment_file)

    exp = experiment_dict['exps'][experiment_index]
    seed = experiment_dict['seed']

    # Ignore experiment if it already exists
    histories_dir = f'./augmentation-hist/{os.path.splitext(os.path.basename(experiment_file))[0]}'
    if os.path.exists(os.path.join(histories_dir,f"{exp['name']}_history.json")) and rerun_completed:
        print('Experiment already exists! Exiting...')
        return

    # Generate both the model with the specified augmentation techniques and its preprocessing function
    model_with_augmentation, preprocess_func = get_augmented_model_and_preprocess(exp, seed)

    # Generate training, validation and test datasets
    output_format = exp['output_format']
    batch_size = exp['batch_size']
    input_shape = vpd.MODELS_DICT[exp['base_model']][1]
    dataset_specs = vpd.DATASETS_DICT[exp['dataset']]
    label_columns = vpd.TRANSFORMERS_DICT[output_format][1]
    train_dataset, val_dataset, _ = generate_dataset_with_splits(dataset_specs, label_columns, output_format, preprocess_func, input_shape, batch_size, random_seed=seed)

    # Show model summary and compile model
    print(model_with_augmentation.summary())

    _, _, loss_function, _, _ = vpd.TRANSFORMERS_DICT[exp['output_format']]
    model_with_augmentation.compile(loss=loss_function, optimizer=Adam(learning_rate=1e-05, decay=1e-8))

    # Create model checkpoint
    checkpoints_dir = f'./augmentation-chkpt/{os.path.splitext(os.path.basename(experiment_file))[0]}'
    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)
    best_checkpoint = ModelCheckpoint(filepath=os.path.join(checkpoints_dir,f"{exp['name']}_bestmodel.h5"), monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weigts_only=True)

    # Create history folder
    histories_dir = f'./augmentation-hist/{os.path.splitext(os.path.basename(experiment_file))[0]}'
    if not os.path.exists(histories_dir):
        os.mkdir(histories_dir)

    # Fit model!
    history = model_with_augmentation.fit(train_dataset, validation_data=val_dataset, epochs=20, callbacks=[best_checkpoint])

    # Save history file
    with open(os.path.join(histories_dir, f"{exp['name']}_history.json"), 'w') as f:
        json.dump(history.history, f)

if __name__ == '__main__':
    main()