import matplotlib.pyplot as plt
import os
import sys

def main():
    experiment_names = ['brightness', 'contrast', 'flip', 'rotation', 'translation', 'zoom']
    metric_lines = {'balanced_accuracy': 4, 'accuracy': 5, 'Mean_EMD': 6, 'MSE': 7}

    experiment_group = sys.argv[1]
    metric_name = sys.argv[2]
    metric_line = metric_lines[metric_name]
    rates = [ int(x) for x in sys.argv[3:] ]

    for exp in experiment_names:
        metric_values = []
        for rate in rates:
            with open(os.path.join('augmentation-results', experiment_group, f'{exp}_{rate}_results.txt'), 'r') as f:
                exp_results = f.readlines()
                metric_values.append(float(exp_results[metric_line].split(':')[1][1:]))
        
        plt.plot(rates, metric_values, label=exp)
    
    plt.legend()
    plt.savefig(os.path.join('figures', experiment_group, f'plot_{metric_name}.png'))

if __name__ == "__main__":
    main()