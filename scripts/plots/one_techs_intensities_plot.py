import matplotlib.pyplot as plt
import os
import json
import sys

experiment_names_colors = {
    'brightness': '#0061ff', 
    'contrast': '#60efff', 
    'flip': '#f7b267', 
    'rotation': '#f79d65', 
    'translation': '#f4845f', 
    'zoom': '#f27059'
}

metrics_per_type = {
    'distribution': ['binary_balanced_accuracy', 'binary_accuracy', 'bal_accuracy_maxrating', 'accuracy_maxrating', 'bal_accuracy_meanbin', 'accuracy_meanbin', 'mean_emd', 'mean_squared_error', 'average_prediction_entropy', 'average_groundtruth_entropy'],
    'tenclass': ['binary_balanced_accuracy', 'binary_accuracy', 'mean_squared_error'],
    'binary': ['binary_balanced_accuracy', 'binary_accuracy', 'mean_squared_error']
}

titles_per_baseline = {
    'baseline_ava_small': 'AVA small',
    'baseline_photozilla_ovr_aerial': 'Photozilla-aerial',
    'baseline_photozilla_ovr_architecture': 'Photozilla-architecture',
    'baseline_photozilla_ovr_event': 'Photozilla-event',
    'baseline_photozilla_ovr_fashion': 'Photozilla-fashion',
    'baseline_photozilla_ovr_food': 'Photozilla-food',
    'baseline_photozilla_ovr_nature': 'Photozilla-nature',
    'baseline_photozilla_ovr_sports': 'Photozilla-sports',
    'baseline_photozilla_ovr_street': 'Photozilla-street',
    'baseline_photozilla_ovr_wedding': 'Photozilla-wedding',
    'baseline_photozilla_ovr_wildlife': 'Photozilla-wildlife',
}

flip_values = ['h', 'v', 'hv']

def fig_gen(experiment_group, out_format, baseline_name, rates, save=False, metric=None, ax1=None, ax2=None, legend=True):
    '''
    Plot the results of experiments in which the augmentation probability has been modified.

    Arguments:
        - experiment_group: the name of the experiment group to plot
        - out_format: the output format of the experiment group
        - baseline_name: the name of the corresponding, unaugmented baseline
        - rates: a list with the specific rates to plot
        - save: whether to save the figure (True) or to return it (False)
        - metric: what metric to print. If None, generate plots of all metrics
        - ax1 and ax2: Matplotlib Axes in which to print. Can be "None" if "save" is True

    Returns:
        - Nothing or the figure, depending on the value of "save"
    '''
    metrics = metrics_per_type[out_format]

    # Create directory in which to save plot (if it does not exist)
    plot_dir = os.path.join('figures', experiment_group)
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    metrics_to_plot = metrics if metric is None else [metric]

    # TODO Add different limits per metric
    for metric_name in metrics_to_plot:
        axes_to_plot_1 = plt.subplots(1,1)[1] if ax1 is None else ax1
        axes_to_plot_2 = plt.subplots(1,1)[1] if ax2 is None else ax2
        # Fetch baseline result, and plot it as a horizontal line
        with open(os.path.join('augmentation-results', baseline_name, 'baseline_results.json'), 'r') as f:
            baseline_results = json.load(f)
            baseline_metric = baseline_results[metric_name]

        # Plot results for all augmentation techniques and given rates
        for exp in experiment_names_colors:
            metric_values = [baseline_metric]
            if (exp != 'flip'):
                for rate in rates:
                    with open(os.path.join('augmentation-results', experiment_group, f'{exp}_{rate}_results.json'), 'r') as f:
                        exp_results = json.load(f)
                        metric_values.append(exp_results[metric_name])
                
                axes_to_plot_1.set_ylabel(metric_name.replace("_", " ").capitalize())
                axes_to_plot_1.plot([0] + rates, metric_values, label=exp, c=experiment_names_colors[exp])

            else:
                for flip in flip_values:
                    with open(os.path.join('augmentation-results', experiment_group, f'{exp}_{flip}_results.json'), 'r') as f:
                        exp_results = json.load(f)
                        metric_values.append(exp_results[metric_name])

                axes_to_plot_2.yaxis.set_tick_params(labelbottom=True)
                axes_to_plot_2.bar(flip_values, metric_values[1:], color=experiment_names_colors[exp])

        axes_to_plot_1.axhline(y = baseline_metric, color = 'g', linestyle = (0,(1,5)), label = 'Baseline')
        axes_to_plot_2.axhline(y = baseline_metric, color = 'g', linestyle = (0,(1,5)), label = 'Baseline')

        # Set vertical limits, if plotting accuracy or balanced accuracy. Leave no upper limit, otherwise
        if metric_name in ['balanced_accuracy', 'accuracy']:
            axes_to_plot_1.set_ylim(0,1)
            axes_to_plot_2.set_ylim(0,1)
        else:
            axes_to_plot_1.set_ylim(bottom = 0)
            axes_to_plot_2.set_ylim(bottom = 0)

        # If plotting balanced accuracy, also plot a THICC red line on 0.5, to highlight ZeroR performance
        if metric_name == 'balanced_accuracy':
            axes_to_plot_1.axhline(y = 0.5, color = 'r', linestyle = 'dotted', label = 'ZeroR performance', linewidth = 1.5)
            axes_to_plot_2.axhline(y = 0.5, color = 'r', linestyle = 'dotted', label = 'ZeroR performance', linewidth = 1.5)

        axes_to_plot_1.set_xticks([0,25,50,75,100])
        axes_to_plot_2.set_xticks([0,1,2])
        axes_to_plot_2.set_xticklabels(['horizontal', 'vertical', 'both'])
        axes_to_plot_1.set_xlim(0, 100)
        axes_to_plot_2.set_xlim(-0.5, 2.5)

        axes_to_plot_1.set_ylim(0,1)
        axes_to_plot_2.set_ylim(0,1)

        if legend:
            axes_to_plot_1.legend()
            axes_to_plot_2.legend()

        axes_to_plot_1.set_xlabel('Augmentation intensity')
        axes_to_plot_2.set_xlabel('Flipping technique')
        axes_to_plot_1.set_ylabel(metric_name.replace('_', ' ').capitalize())
        axes_to_plot_1.set_title(titles_per_baseline[baseline_name], position=(1, 1))

        if save:
            plt.savefig(os.path.join(plot_dir, f'plot_{metric_name}.eps'))
        

if __name__ == "__main__":

    # Parse command-line arguments
    experiment_group = sys.argv[1]
    out_format = sys.argv[2]
    baseline_name = sys.argv[3]
    rates = [ int(x) for x in sys.argv[4:] ]

    fig_gen(experiment_group, out_format, baseline_name, rates, save=True)