import matplotlib.pyplot as plt
import os
import json
import sys

def main():
    '''
    Plot the results of experiments in which the augmentation intensity has been modified.
    Since flipping experiments do not have an intensity per-sé, these are plotted
    as a barplot, for all possible kinds of flipping

    Command-line arguments:
        - 1st argument: the name of the experiment group to plot
        - 2nd argument: the name of the metric to plot
        - 3rd argument: the name of the corresponding, unaugmented baseline
        - 4th argument and so on: the specific rates to plot

    Returns:
        - Nothing. The plot itself is saved on figures/{experiment_group}/plot_{metric_name}
    '''

    experiment_names = ['brightness', 'contrast', 'flip', 'rotation', 'translation', 'zoom']
    flip_values = ['horizontal', 'vertical', 'both']

    # Parse command-line arguments
    experiment_group = sys.argv[1]
    metric_name = sys.argv[2]
    baseline_name = sys.argv[3]
    rates = [ int(x) for x in sys.argv[4:] ]

    # Create directory in which to save plot (if it does not exist)
    plot_dir = os.path.join('figures', experiment_group)
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    # Plot results for all augmentation techniques and given intensities
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8), sharey='row')
    for exp in experiment_names:
        metric_values = []
        if (exp != 'flip'):
            for rate in rates:
                with open(os.path.join('augmentation-results', experiment_group, f'{exp}_{rate}_results.json'), 'r') as f:
                    exp_results = json.load(f)
                    metric_values.append(exp_results[metric_name])
            
            ax1.set_xlabel("Augmentation intensity")
            ax1.set_ylabel(metric_name.replace("_", " ").capitalize())
            ax1.plot(rates, metric_values, label=exp)

        else:
            for flip in flip_values:
                with open(os.path.join('augmentation-results', experiment_group, f'{exp}_{flip}_results.json'), 'r') as f:
                    exp_results = json.load(f)
                    metric_values.append(exp_results[metric_name])

            ax2.set_xlabel("Flip type")
            ax2.yaxis.set_tick_params(labelbottom=True)
            ax2.bar(flip_values, metric_values)

    # Fetch baseline result, and plot it as a horizontal line, on both figures
    with open(os.path.join('augmentation-results', baseline_name, 'baseline_results.json'), 'r') as f:
        baseline_results = json.load(f)
        baseline_metric = baseline_results[metric_name]

    ax1.axhline(y = baseline_metric, color = 'g', linestyle = '--', label = 'Baseline')
    ax2.axhline(y = baseline_metric, color = 'g', linestyle = '--', label = 'Baseline')

    # Set vertical limits, if plotting accuracy or balanced accuracy. Leave no upper limit, otherwise
    if metric_name in ['balanced_accuracy', 'accuracy']:
        plt.ylim(0,1)
    else:
        plt.ylim(bottom = 0)

    # If plotting balanced accuracy, also plot a THICC red line on 0.5, to highlight ZeroR performance
    if metric_name == 'balanced_accuracy':
        ax1.axhline(y = 0.5, color = 'r', linestyle = 'dotted', label = 'ZeroR performance', linewidth = 1.5)

    ax1.set_xticks([10,20,30,40,50,60,70,80,90,100])
    ax1.set_xlim(10, 100)

    ax1.legend()
    ax2.legend()
    plt.savefig(os.path.join(plot_dir, f'plot_{metric_name}.png'))

if __name__ == "__main__":
    main()