from cProfile import label
import matplotlib.pyplot as plt
import os

def main():
    experiment_names = ['brightness', 'contrast', 'flip', 'rotation', 'translation', 'zoom']
    rates = [10, 25, 50, 100]

    for exp in experiment_names:
        accuracies = []
        bal_accuracies = []
        for rate in rates:
            with open(os.path.join('augmentation-results', 'one_techs_rates', f'{exp}_{rate}_results.txt'), 'r') as f:
                exp_results = f.readlines()
                accuracies.append(float(exp_results[5].split(':')[1][1:]))
                bal_accuracies.append(float(exp_results[4].split(':')[1][1:]))
        
        plt.plot(rates, accuracies, label=exp)
    
    plt.legend()
    plt.savefig('plot_accuracy.png')


if __name__ == "__main__":
    main()