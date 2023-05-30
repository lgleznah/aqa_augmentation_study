import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from experiment_lists import baselines, low_intensity, high_intensity, techniques, rates

scale = 2
palette = ["#0011ff","#ff7b00","#ff8800","#ff9500","#ffa200","#ffaa00","#ffb700","#ffc300","#ffd000","#ffdd00","#ffea00"]
palette_agg = ["#0011ff","#ff7b00","#ffb700","#ffea00"]

def main():
    results_list_ava = []
    results_list_celeba = []
    results_list_aggregated = []
    results_list_full_aggregated = []
    baseline_results = {}

    low_intensity_ava, high_intensity_ava = low_intensity['ava-photozilla'], high_intensity['ava-photozilla']
    low_intensity_cel, high_intensity_cel = low_intensity['celeba'], high_intensity['celeba']
    baselines_ava, baselines_cel = baselines['ava-photozilla'], baselines['celeba']

    sns.set(rc={
        "figure.figsize":(8.27*scale, 11.69*scale), 
        "font.size":25, 
        "axes.labelsize":25,
        "xtick.labelsize":20,
        "ytick.labelsize":20,
    })
    sns.set_palette(palette)

    for baseline in baselines_ava + baselines_cel:
        with open(os.path.join('augmentation-results', baseline, 'baseline_results.json'), 'r') as f:
            results = json.load(f)
            baseline_results[baseline] = results['binary_balanced_accuracy']

    for experiments, plot_name in [(low_intensity_ava, 'low_intensity'), (high_intensity_ava, 'high_intensity')]:
        results_list_ava = []
        for experiment in experiments:
            baseline_name = f"baseline_{'_'.join(experiment.split('_')[4:])}"
            dataset_name = ' '.join(baseline_name.split('_')[1:])
            for technique in techniques:
                for rate in rates:
                    with open(os.path.join('augmentation-results', experiment, f'{technique}_{rate}_results.json'), 'r') as f:
                        results = json.load(f)
                        result_dict = {
                            'Augmentation technique': technique,
                            'Dataset': dataset_name.capitalize(),
                            'Balanced accuracy difference': results['binary_balanced_accuracy'] - baseline_results[baseline_name]
                        }
                        results_list_ava.append(result_dict)

        df = pd.DataFrame.from_records(results_list_ava)
        plot = sns.boxplot(
            data=df, 
            x='Balanced accuracy difference', 
            y='Augmentation technique', 
            hue='Dataset', 
            medianprops=dict(color="#008000", alpha=1.0),
            boxprops={'zorder':10},
            whiskerprops={'zorder':10},
            zorder=10
        )
        plot.axvline(x=0, ymin=0, ymax=1, color='black', zorder=1, linestyle=":")
        fig = plot.get_figure()
        fig.savefig(os.path.join('figures', f'{plot_name}_boxplot_ava_photozilla.eps'))
        plt.close()

    for experiments, plot_name in [(low_intensity_cel, 'low_intensity'), (high_intensity_cel, 'high_intensity')]:
        results_list_celeba = []
        for experiment in experiments:
            baseline_name = f"baseline_{'_'.join(experiment.split('_')[4:])}"
            dataset_name = ' '.join(baseline_name.split('_')[1:])
            for technique in techniques:
                for rate in rates:
                    with open(os.path.join('augmentation-results', experiment, f'{technique}_{rate}_results.json'), 'r') as f:
                        results = json.load(f)
                        result_dict = {
                            'Augmentation technique': technique,
                            'Dataset': dataset_name.capitalize(),
                            'Balanced accuracy difference': results['binary_balanced_accuracy'] - baseline_results[baseline_name]
                        }
                        results_list_celeba.append(result_dict)

        df = pd.DataFrame.from_records(results_list_celeba)
        plot = sns.boxplot(
            data=df, 
            x='Balanced accuracy difference', 
            y='Augmentation technique', 
            hue='Dataset', 
            medianprops=dict(color="#008000", alpha=1.0),
            boxprops={'zorder':10},
            whiskerprops={'zorder':10},
            zorder=10
        )
        plot.axvline(x=0, ymin=0, ymax=1, color='black', zorder=1, linestyle=":")
        fig = plot.get_figure()
        fig.savefig(os.path.join('figures', f'{plot_name}_boxplot_celeba.eps'))
        plt.close()

    sns.set_palette(palette_agg)
    for experiments, plot_name in [(low_intensity_ava + low_intensity_cel, 'low_intensity'), (high_intensity_ava + high_intensity_cel, 'high_intensity')]:
        results_list_aggregated = []
        for experiment in experiments:
            baseline_name = f"baseline_{'_'.join(experiment.split('_')[4:])}"
            dataset_name = ' '.join(baseline_name.split('_')[1:])
            if dataset_name not in ['celeba photos', 'ava small']:
                if dataset_name[0] == 'p':
                    dataset_name = 'Photozilla'
                elif dataset_name[0] == 'c':
                    dataset_name = 'CelebA objective'
                else:
                    raise ValueError(f'Unexpected dataset! Got {dataset_name}')
                
            for technique in techniques:
                for rate in rates:
                    with open(os.path.join('augmentation-results', experiment, f'{technique}_{rate}_results.json'), 'r') as f:
                        results = json.load(f)
                        result_dict = {
                            'Augmentation technique': technique,
                            'Dataset': dataset_name.capitalize(),
                            'Balanced accuracy difference': results['binary_balanced_accuracy'] - baseline_results[baseline_name]
                        }
                        results_list_aggregated.append(result_dict)

        df = pd.DataFrame.from_records(results_list_aggregated)
        plot = sns.boxplot(
            data=df, 
            x='Balanced accuracy difference', 
            y='Augmentation technique', 
            hue='Dataset', 
            medianprops=dict(color="#008000", alpha=1.0),
            boxprops={'zorder':10},
            whiskerprops={'zorder':10},
            zorder=10
        )
        plot.axvline(x=0, ymin=0, ymax=1, color='black', zorder=1, linestyle=":")
        fig = plot.get_figure()
        fig.savefig(os.path.join('figures', f'{plot_name}_boxplot_aggregated.eps'))
        plt.close()

    for experiment in low_intensity_ava + low_intensity_cel + high_intensity_ava + high_intensity_cel:
        baseline_name = f"baseline_{'_'.join(experiment.split('_')[4:])}"
        dataset_name = ' '.join(baseline_name.split('_')[1:])
        if dataset_name not in ['celeba photos', 'ava small']:
            if dataset_name[0] == 'p':
                dataset_name = 'Photozilla'
            elif dataset_name[0] == 'c':
                dataset_name = 'CelebA objective'
            else:
                raise ValueError(f'Unexpected dataset! Got {dataset_name}')
            
        for technique in techniques:
            for rate in rates:
                with open(os.path.join('augmentation-results', experiment, f'{technique}_{rate}_results.json'), 'r') as f:
                    results = json.load(f)
                    result_dict = {
                        'Augmentation technique': technique,
                        'Dataset': dataset_name.capitalize(),
                        'Balanced accuracy difference': results['binary_balanced_accuracy'] - baseline_results[baseline_name]
                    }
                    results_list_full_aggregated.append(result_dict)

    df = pd.DataFrame.from_records(results_list_full_aggregated)
    plot = sns.boxplot(
        data=df, 
        x='Balanced accuracy difference', 
        y='Augmentation technique', 
        hue='Dataset', 
        medianprops=dict(color="#008000", alpha=1.0),
        boxprops={'zorder':10},
        whiskerprops={'zorder':10},
        zorder=10
    )
    plot.axvline(x=0, ymin=0, ymax=1, color='black', zorder=1, linestyle=":")
    fig = plot.get_figure()
    fig.savefig(os.path.join('figures', f'boxplot_full_aggregated.eps'))
    plt.close()

if __name__ == "__main__":
    main()