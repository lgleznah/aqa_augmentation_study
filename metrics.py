import os
import json
import sys

from losses import earth_mover_loss
from experiment_parser import parse_experiment_file
from dataset_generator import generate_dataset_with_splits

import valid_parameters_dicts as vpd
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import balanced_accuracy_score, accuracy_score, mean_squared_error, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import entropy

def bal_accuracy_thirds(ground, pred):
    """
    Compute balanced accuracy of (pred) w.r.t. (ground) on each tercile of the ground truth (ground)
    
    Assumes both predictions and ground-truth to be a single probability value, in the range (0,1)

    Prints the results in a file called model_name, located in the directory specified by the environment variable "AQA_results"
    """
    metrics_dict = {}
    ground_terciles = np.percentile(ground, [i*100/3 for i in range(1,4)])
    
    index_bad = np.argwhere(ground < ground_terciles[0])
    index_normal = np.argwhere((ground >= ground_terciles[0]) & (ground < ground_terciles[1]))
    index_good = np.argwhere(ground >= ground_terciles[1])
    
    if (index_bad.size != 0):
        bal_acc_bad = balanced_accuracy_score(ground[index_bad] > 0.5, pred[index_bad] > 0.5)
    else:
        bal_acc_bad = 0
    bal_acc_normal = balanced_accuracy_score(ground[index_normal] > 0.5, pred[index_normal] > 0.5)
    bal_acc_good = balanced_accuracy_score(ground[index_good] > 0.5, pred[index_good]> 0.5)
    
    metrics_dict['first_tercile_balanced_accuracy'] = bal_acc_bad
    metrics_dict['second_tercile_balanced_accuracy'] = bal_acc_normal
    metrics_dict['third_tercile_balanced_accuracy'] = bal_acc_good

    return metrics_dict

def get_distribution_metrics_and_plot(ground, pred):
    """
    Compute balanced accuracy, mean EMD distance, and accuracy of a set of predictions (pred)
    WRT to another ground-truth (ground), and print results, naming the other ground truth as (ground-name).
    
    After printing such metrics, plot predictions and ground-truth together.
    
    Assumes both predictions and ground-truths to be normalized vote distributions, from 1 to 10
    """
    metrics_dict = {}

    pred_means = np.sum(pred * np.arange(0.1,1.1,0.1), axis=1) / np.sum(pred, axis=1)
    ground_means = np.sum(ground * np.arange(0.1,1.1,0.1), axis=1) / np.sum(ground, axis=1)
                
    bal_accuracy = balanced_accuracy_score(ground_means > 0.5, pred_means > 0.5)
    mean_emd = np.mean([earth_mover_loss(tf.constant(pred[i], dtype=tf.float64), tf.constant(ground[i], dtype=tf.float64)) for i in range(len(pred))])
    accuracy = accuracy_score(ground_means > 0.5, pred_means > 0.5)
    mse = mean_squared_error(ground, pred)
    avg_entropy_pred = np.mean([entropy(p, base=2) for p in np.vstack((1-pred_means,pred_means)).T])
    avg_entropy_grnd = np.mean([entropy(g, base=2) for g in np.vstack((1-ground_means,ground_means)).T])
    bal_accuracy_maxrating = balanced_accuracy_score(np.argmax(ground, axis=1), np.argmax(pred, axis=1))
    accuracy_maxrating = accuracy_score(np.argmax(ground, axis=1), np.argmax(pred, axis=1))
    bal_accuracy_meanbin = balanced_accuracy_score(np.floor(ground_means * 9 + 1), np.floor(pred_means * 9 + 1))
    accuracy_meanbin = accuracy_score(np.floor(ground_means), np.floor(pred_means))

    metrics_dict['binary_balanced_accuracy'] = bal_accuracy
    metrics_dict['binary_accuracy'] = accuracy
    metrics_dict['bal_accuracy_maxrating'] = bal_accuracy_maxrating
    metrics_dict['accuracy_maxrating'] = accuracy_maxrating
    metrics_dict['bal_accuracy_meanbin'] = bal_accuracy_meanbin
    metrics_dict['accuracy_meanbin'] = accuracy_meanbin
    metrics_dict['mean_emd'] = mean_emd
    metrics_dict['mean_squared_error'] = mse
    metrics_dict['average_prediction_entropy'] = avg_entropy_pred
    metrics_dict['average_groundtruth_entropy'] = avg_entropy_grnd
    
    balaccs_per_tercile = bal_accuracy_thirds(ground_means, pred_means)

    return {**metrics_dict, **balaccs_per_tercile}
    
def get_binary_metrics_and_plot(ground, pred):
    """
    Compute balanced accuracy, mean EMD distance, and accuracy of a set of predictions (pred)
    WRT to another ground-truth (ground), and print results, naming the other ground truth as (ground-name).
    
    After printing such metrics, plot predictions and ground-truth together.
    
    Assumes both predictions and ground-truths to be a 2-component probability distribution, in the range (0,1). 
    """
    metrics_dict = {}

    bal_accuracy = balanced_accuracy_score(ground[:,1] > 0.5, pred[:,1] > 0.5)
    accuracy = accuracy_score(ground[:,1] > 0.5, pred[:,1] > 0.5)
    mse = mean_squared_error(ground, pred)
    confmat = confusion_matrix(ground[:,1] > 0.5, pred[:,1] > 0.5).tolist()
    avg_entropy_pred = np.mean([entropy(p, base=2) for p in pred])
    avg_entropy_grnd = np.mean([entropy(g, base=2) for g in ground])

    # Component swapping fix
    if (accuracy < 0.5):
        pred = 1 - pred
        bal_accuracy = balanced_accuracy_score(ground[:,1] > 0.5, pred[:,1] > 0.5)
        accuracy = accuracy_score(ground[:,1] > 0.5, pred[:,1] > 0.5)
        mse = mean_squared_error(ground, pred)

    metrics_dict['binary_balanced_accuracy'] = bal_accuracy
    metrics_dict['binary_accuracy'] = accuracy
    metrics_dict['mean_squared_error'] = mse
    metrics_dict['average_prediction_entropy'] = avg_entropy_pred
    metrics_dict['average_groundtruth_entropy'] = avg_entropy_grnd
    metrics_dict['confusion_matrix'] = confmat
        
    balaccs_per_tercile = bal_accuracy_thirds(ground[:,1], pred[:,1])

    return {**metrics_dict, **balaccs_per_tercile}

def get_tenclass_metrics_and_plot(ground, pred):
    """
    Compute balanced accuracy, mean EMD distance, and accuracy of a set of predictions (pred)
    WRT to another ground-truth (ground), and print results, naming the other ground truth as (ground-name).
    
    After printing such metrics, plot predictions and ground-truth together.
    
    Assumes both predictions and ground-truths to be a 2-component probability distribution, in the range (0,1). 
    """
    metrics_dict = {}

    sparse_predictions = np.argmax(pred, axis=1)
    onehot_groundtruth = OneHotEncoder().fit_transform(list(zip(ground))).toarray()
    bal_accuracy = balanced_accuracy_score(ground, sparse_predictions)
    accuracy = accuracy_score(ground, sparse_predictions)
    mse = mean_squared_error(onehot_groundtruth, pred)
    confmat = confusion_matrix(ground, sparse_predictions).tolist()

    metrics_dict['binary_balanced_accuracy'] = bal_accuracy
    metrics_dict['binary_accuracy'] = accuracy
    metrics_dict['mean_squared_error'] = mse
    metrics_dict['confusion_matrix'] = confmat
        
    return metrics_dict
            
######################################################################################################################################
def main():

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    class_names = ['aerial', 'architecture', 'event', 'fashion', 'food', 'nature', 'sports', 'street', 'wedding', 'wildlife']

    experiment_index = int(sys.argv[1])
    experiment_file = os.path.join(os.environ['AQA_AUGMENT_EXPERIMENTS_PATH'], f'{sys.argv[2]}.yaml')

    # Parse specified experiment file
    experiment_dict = parse_experiment_file(experiment_file)

    exp = experiment_dict['exps'][experiment_index]
    seed = experiment_dict['seed']

    predictions_dir = f'./augmentation-preds/{os.path.splitext(os.path.basename(experiment_file))[0]}'

    output_format = exp['output_format']
    batch_size = exp['batch_size']
    input_shape = vpd.MODELS_DICT[exp['base_model']][1]
    dataset_specs = vpd.DATASETS_DICT[experiment_dict['dataset']]
    label_columns = vpd.TRANSFORMERS_DICT[output_format][1]
    _, _, test_scores = generate_dataset_with_splits(dataset_specs, label_columns, output_format, input_shape, batch_size, labels_only=True, random_seed=seed)
        
    predictions = np.load(os.path.join(predictions_dir, f"{exp['name']}_predictions.npy"))
    groundtruth = test_scores
    
    plot_dir = os.path.join(predictions_dir, f"{exp['name']}_graphs")
    results_dir = f'./augmentation-results/{os.path.splitext(os.path.basename(experiment_file))[0]}'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    # Each form of ground-truth (distribution, binary weights/classes) requires
    # a different way of obtaining metrics.
                  
    # Distribution-like ground-truths
    if (output_format == 'distribution'):
        metrics = get_distribution_metrics_and_plot(groundtruth, predictions)

    # Binary-like ground-truths
    if (output_format == 'weights'):
        metrics = get_binary_metrics_and_plot(groundtruth, predictions)

    # Ten-class ground-truths
    if (output_format == 'tenclass'):
        metrics = get_tenclass_metrics_and_plot(groundtruth, predictions)
        confmat_dir = os.path.join('figures', os.path.splitext(os.path.basename(experiment_file))[0], 'confmats')
        if not os.path.exists(confmat_dir):
            os.makedirs(confmat_dir, exist_ok=True)
        
        confmat = metrics['confusion_matrix']
        df_cm = pd.DataFrame(confmat, index=[i for i in class_names], columns=[i for i in class_names])
        plt.figure(figsize=(10,7))
        sns.heatmap(df_cm, annot=True)
        plt.savefig(os.path.join(confmat_dir, f"{exp['name']}_confmat.svg"))

    with open(os.path.join(results_dir, f"{exp['name']}_results.json"), 'w') as f:
        json.dump(metrics, f)

if __name__ == '__main__':
    main()