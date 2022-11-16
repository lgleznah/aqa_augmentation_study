import matplotlib.pyplot as plt
import os
import sys

def main():
    experiment_names = ['brightness', 'contrast', 'flip', 'rotation', 'translation', 'zoom']
    metric_lines = {'balanced_accuracy': 4, 'accuracy': 5, 'Mean_EMD': 6, 'MSE': 7}
    flip_values = ['horizontal', 'vertical', 'both']

    experiment_group = sys.argv[1]
    metric_name = sys.argv[2]
    metric_line = metric_lines[metric_name]
    rates = [ int(x) for x in sys.argv[3:] ]

    plot_dir = os.path.join('figures', experiment_group)
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8), sharey='row')
    for exp in experiment_names:
        metric_values = []
        if (exp != 'flip'):
            for rate in rates:
                with open(os.path.join('augmentation-results', experiment_group, f'{exp}_{rate}_results.txt'), 'r') as f:
                    exp_results = f.readlines()
                    metric_values.append(float(exp_results[metric_line].split(':')[1][1:]))
            
            ax1.set_xlabel("Augmentation intensity")
            ax1.set_ylabel(metric_name.replace("_", " ").capitalize())
            ax1.plot(rates, metric_values, label=exp)
            ax1.legend()

        else:
            for flip in flip_values:
                with open(os.path.join('augmentation-results', experiment_group, f'{exp}_{flip}_results.txt'), 'r') as f:
                    exp_results = f.readlines()
                    metric_values.append(float(exp_results[metric_line].split(':')[1][1:]))

            ax2.set_xlabel("Flip type")
            ax2.yaxis.set_tick_params(labelbottom=True)
            ax2.bar(flip_values, metric_values)

    
    plt.savefig(os.path.join(plot_dir, f'plot_{metric_name}.png'))

if __name__ == "__main__":
    main()