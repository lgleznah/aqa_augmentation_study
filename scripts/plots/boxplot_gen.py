import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from experiment_lists import baselines, low_intensity, high_intensity, techniques, rates

scale = 2
palette = ["#0011ff","#ff7b00","#ff8800","#ff9500","#ffa200","#ffaa00","#ffb700","#ffc300","#ffd000","#ffdd00","#ffea00"]

def main():
    results_list = []
    baseline_results = {}

    sns.set(rc={"figure.figsize":(8.27*scale, 11.69*scale)})
    sns.set_palette(palette)

    for baseline in baselines:
        with open(os.path.join('augmentation-results', baseline, 'baseline_results.json'), 'r') as f:
            results = json.load(f)
            baseline_results[baseline] = results['binary_balanced_accuracy']

    for experiments, plot_name in [(low_intensity, 'low_intensity'), (high_intensity, 'high_intensity')]:
        for experiment in experiments:
            baseline_name = f"baseline_{'_'.join(experiment.split('_')[4:])}"
            dataset_name = ' '.join(baseline_name.split('_')[1:])
            for technique in techniques:
                for rate in rates:
                    with open(os.path.join('augmentation-results', experiment, f'{technique}_{rate}_results.json'), 'r') as f:
                        results = json.load(f)
                        result_dict = {
                            'technique': technique,
                            'dataset': dataset_name.capitalize(),
                            'difference': results['binary_balanced_accuracy'] - baseline_results[baseline_name]
                        }
                        results_list.append(result_dict)

        df = pd.DataFrame.from_records(results_list)
        plot = sns.boxplot(data=df, x='difference', y='technique', hue='dataset')
        fig = plot.get_figure()
        fig.savefig(os.path.join('figures', f'{plot_name}_boxlpot.eps'))
        plt.close()


if __name__ == "__main__":
    main()