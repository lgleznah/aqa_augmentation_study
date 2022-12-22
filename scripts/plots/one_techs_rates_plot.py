import matplotlib.pyplot as plt
import os
import json
import sys

def main():
    '''
    Plot the results of experiments in which the augmentation probability has been modified.

    Command-line arguments:
        - 1st argument: the name of the experiment group to plot
        - 2nd argument: the name of the metric to plot
        - 3rd argument: the name of the corresponding, unaugmented baseline
        - 4th argument and so on: the specific rates to plot

    Returns:
        - Nothing. The plot itself is saved on figures/{experiment_group}/plot_{metric_name}
    '''

    #experiment_names = ['brightness', 'contrast', 'flip', 'rotation', 'translation', 'zoom']
    experiment_names = ['brightness', 'flip', 'rotation', 'translation', 'zoom']

    # Parse command-line arguments
    experiment_group = sys.argv[1]
    metric_name = sys.argv[2]
    baseline_name = sys.argv[3]
    rates = [ int(x) for x in sys.argv[4:] ]

    # Create directory in which to save plot (if it does not exist)
    plot_dir = os.path.join('figures', experiment_group)
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    # Fetch baseline result, and plot it as a horizontal line
    with open(os.path.join('augmentation-results', baseline_name, 'baseline_results.json'), 'r') as f:
        baseline_results = json.load(f)
        baseline_metric =baseline_results[metric_name]

    # Plot results for all augmentation techniques and given rates
    for exp in experiment_names:
        metric_values = [baseline_metric]
        for rate in rates:
            with open(os.path.join('augmentation-results', experiment_group, f'{exp}_{rate}_results.json'), 'r') as f:
                exp_results = json.load(f)
                metric_values.append(exp_results[metric_name])
        
        plt.plot([0] + rates, metric_values, label=exp.capitalize())

    plt.axhline(y = baseline_metric, color = 'g', linestyle = '--', label = 'Baseline')

    # Set vertical limits, if plotting accuracy or balanced accuracy. Leave no upper limit, otherwise
    if metric_name in ['balanced_accuracy', 'accuracy']:
        plt.ylim(0,1)
    else:
        plt.ylim(bottom = 0)

    # If plotting balanced accuracy, also plot a THICC red line on 0.5, to highlight ZeroR performance
    if metric_name == 'balanced_accuracy':
        plt.axhline(y = 0.5, color = 'r', linestyle = 'dotted', label = 'ZeroR performance', linewidth = 1.5)

    plt.xticks([0,10,20,30,40,50,60,70,80,90,100])
    plt.xlim(0, 100)

    plt.ylim(0,1)
    plt.legend()
    plt.savefig(os.path.join(plot_dir, f'plot_{metric_name}.png'))

if __name__ == "__main__":
    main()