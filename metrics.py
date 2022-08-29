import os
import warnings

from losses import earth_mover_loss

from experiment_parser import parse_experiment_file
from augmented_model_generator import get_augmented_model_and_preprocess
from dataset_generator import generate_dataset_with_splits

import layers_models_transforms_dicts as lmd

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import sys

from sklearn.metrics import balanced_accuracy_score, accuracy_score, mean_squared_error, mean_absolute_error, confusion_matrix
from scipy.stats import rankdata
from scipy.stats import entropy

def bal_accuracy_thirds(ground, pred, model_name, results_dir):
    """
    Compute balanced accuracy of (pred) w.r.t. (ground) on each tercile of the ground truth (ground)
    
    Assumes both predictions and ground-truth to be a single probability value, in the range (0,1)

    Prints the results in a file called model_name, located in the directory specified by the environment variable "AQA_results"
    """
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
    
    with open(os.path.join(results_dir, f"{model_name}_results.txt"), 'a') as f:
        print(f'Balanced accuracy results by ground-truth quality terciles: ', file=f)
        print(f'Balanced accuracy, 1st tercile: {bal_acc_bad}', file=f)
        print(f'Balanced accuracy, 2nd tercile: {bal_acc_normal}', file=f)
        print(f'Balanced accuracy, 3rd tercile: {bal_acc_good}', file=f)
        print(f"\n{'#'*80}\n", file=f)

def get_distribution_metrics_and_plot(ground, pred, ground_name, plot_dir, plot_name, model_name, results_dir):
    """
    Compute balanced accuracy, mean EMD distance, and accuracy of a set of predictions (pred)
    WRT to another ground-truth (ground), and print results, naming the other ground truth as (ground-name).
    
    After printing such metrics, plot predictions and ground-truth together.
    
    Assumes both predictions and ground-truths to be normalized vote distributions, from 1 to 10
    """
    pred_means = np.sum(pred * np.arange(0.1,1.1,0.1), axis=1) / np.sum(pred, axis=1)
    ground_means = np.sum(ground * np.arange(0.1,1.1,0.1), axis=1) / np.sum(ground, axis=1)
                
    bal_accuracy = balanced_accuracy_score(ground_means > 0.5, pred_means > 0.5)
    mean_emd = np.mean([earth_mover_loss(tf.constant(pred[i], dtype=tf.float64), tf.constant(ground[i], dtype=tf.float64)) for i in range(len(pred))])
    accuracy = accuracy_score(ground_means > 0.5, pred_means > 0.5)
    mse = mean_squared_error(ground, pred)
    avg_entropy_pred = np.mean([entropy(p, base=2) for p in np.vstack((1-pred_means,pred_means)).T])
    avg_entropy_grnd = np.mean([entropy(g, base=2) for g in np.vstack((1-ground_means,ground_means)).T])
    
    with open(os.path.join(results_dir, f"{model_name}_results.txt"), 'a') as f:
        print(f"RESULTS W.R.T. {ground_name}: ", file=f)
        print("Balanced accuracy: " + str(bal_accuracy), file=f)
        print("Accuracy: " + str(accuracy), file=f)
        print("Mean EMD distance: " + str(mean_emd), file=f)
        print("Mean squared error: " + str(mse), file=f)
        print("Average entropy (predictions): " + str(avg_entropy_pred), file=f)
        print(f"Average entropy ({ground_name}): " + str(avg_entropy_grnd), file=f)
        print(f"{bal_accuracy:.4f} & {accuracy:.4f} & {mean_emd:.4f} & {mse:.4f} & {avg_entropy_grnd:.4f} \\\\ \\hline", file=f)
        print("--------------------------------------------------------", file=f)
    
    bal_accuracy_thirds(ground_means, pred_means, model_name, results_dir)
    
def get_binary_metrics_and_plot(ground, pred, ground_name, plot_dir, plot_name, model_name, results_dir):
    """
    Compute balanced accuracy, mean EMD distance, and accuracy of a set of predictions (pred)
    WRT to another ground-truth (ground), and print results, naming the other ground truth as (ground-name).
    
    After printing such metrics, plot predictions and ground-truth together.
    
    Assumes both predictions and ground-truths to be a 2-component probability distribution, in the range (0,1). 
    """
    bal_accuracy = balanced_accuracy_score(ground[:,1] > 0.5, pred[:,1] > 0.5)
    accuracy = accuracy_score(ground[:,1] > 0.5, pred[:,1] > 0.5)
    mse = mean_squared_error(ground, pred)
    avg_entropy_pred = np.mean([entropy(p, base=2) for p in pred])
    avg_entropy_grnd = np.mean([entropy(g, base=2) for g in ground])

    # Component swapping fix
    if (accuracy < 0.5):
        pred = 1 - pred
        bal_accuracy = balanced_accuracy_score(ground[:,1] > 0.5, pred[:,1] > 0.5)
        accuracy = accuracy_score(ground[:,1] > 0.5, pred[:,1] > 0.5)
        mse = mean_squared_error(ground, pred)
        
    with open(os.path.join(results_dir, f"{model_name}_results.txt"), 'a') as f:
        print(f"RESULTS W.R.T. {ground_name}: ", file=f)
        print("Balanced accuracy: " + str(bal_accuracy), file=f)
        print("Accuracy: " + str(accuracy), file=f)
        print("Mean squared error: " + str(mse), file=f)
        print("Average entropy (predictions): " + str(avg_entropy_pred), file=f)
        print(f"Average entropy ({ground_name}): " + str(avg_entropy_grnd), file=f)
        print(f"{bal_accuracy:.4f} & {accuracy:.4f} & {mse:.4f} & {avg_entropy_grnd:.4f} \\\\ \\hline", file=f)
        print("--------------------------------------------------------", file=f)
        
    bal_accuracy_thirds(ground[:,1], pred[:,1], model_name, results_dir)
    
######################################################################################################################################
def main():
    experiment_index = int(sys.argv[1])
    experiment_file = sys.argv[2]

    # Parse specified experiment file
    experiment_dict = parse_experiment_file(experiment_file)

    exp = experiment_dict['exps'][experiment_index]

    predictions_dir = f'./augmentation-preds/{os.path.splitext(os.path.basename(experiment_file))[0]}'

    # Generate test dataset
    _, preprocess_func = get_augmented_model_and_preprocess(exp)

    output_format = exp['output_format']
    batch_size = exp['batch_size']
    input_shape = lmd.MODELS_DICT[exp['base_model']][1]
    _, _, test_scores = generate_dataset_with_splits(output_format, preprocess_func, input_shape, batch_size, labels_only=True)
        
    predictions = np.load(os.path.join(predictions_dir, f"{exp['name']}_predictions.npy"))
    groundtruth = test_scores
    
    plot_dir = os.path.join(predictions_dir, f"{exp['name']}_graphs")
    results_dir = f'./augmentation-results/{os.path.splitext(os.path.basename(experiment_file))[0]}'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    # Each form of ground-truth (distribution, binary weights/classes) requires
    # a different way of obtaining metrics.
              
    warnings.filterwarnings("ignore", category=UserWarning)

    # Reset results file
    with open(os.path.join(results_dir, f"{exp['name']}_results.txt"), 'w') as f:
        print(f"\n{'#'*80}\n", file=f)
    
    # Distribution-like ground-truths
    if (output_format == 'distribution'):
        get_distribution_metrics_and_plot(groundtruth, predictions, "groundtruth", plot_dir, "against_groundtruth", exp['name'], results_dir)

    # Binary-like ground-truths
    if (output_format == 'weights'):
        get_binary_metrics_and_plot(groundtruth, predictions, "groundtruth", plot_dir, "against_groundtruth", exp['name'], results_dir)

if __name__ == '__main__':
    main()