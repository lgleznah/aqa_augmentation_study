import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from experiment_lists import baselines, low_intensity, high_intensity, techniques, rates

scale = 2
palette = ["#0011ff","#ff7b00","#ff8800","#ff9500","#ffa200","#ffaa00","#ffb700","#ffc300","#ffd000","#ffdd00","#ffea00"]
palette_agg = ["#0011ff","#0077ff","#ff7b00","#ffb700","#ffea00"]

def main():
    results_list_ava = []
    results_list_celeba = []
    results_list_ava_regression = []
    results_list_aggregated = []
    results_list_full_aggregated = []
    results_list_full_aggregated_regression = []
    baseline_results = {}

    low_intensity_ava, high_intensity_ava = low_intensity['ava'], high_intensity['ava']
    low_intensity_photozilla, high_intensity_photozilla = low_intensity['photozilla'], high_intensity['photozilla']
    low_intensity_cel, high_intensity_cel = low_intensity['celeba'], high_intensity['celeba']
    low_intensity_ava_reg, high_intensity_ava_reg = low_intensity['ava-regression'], high_intensity['ava-regression']
    baselines_ava, baselines_photozilla, baselines_cel, baselines_ava_reg = baselines['ava'], baselines['photozilla'], baselines['celeba'], baselines['ava-regression']

    sns.set(rc={
        "figure.figsize":(8.27*scale, 11.69*scale), 
        "font.size":20, 
        "axes.labelsize":20,
        "xtick.labelsize":20,
        "ytick.labelsize":20,
    })
    sns.set_palette(palette)

    for baseline in baselines_ava + baselines_photozilla + baselines_cel + baselines_ava_reg:
        with open(os.path.join('augmentation-results', baseline, 'baseline_results.json'), 'r') as f:
            results = json.load(f)
            baseline_results[baseline] = results['binary_balanced_accuracy']

    for experiments, plot_name in [(low_intensity_ava + low_intensity_photozilla, 'low_intensity'), (high_intensity_ava + high_intensity_photozilla, 'high_intensity')]:
        results_list_ava = []
        for experiment in experiments:
            baseline_name = f"baseline_{'_'.join(experiment.split('_')[4:])}"
            dataset_name = ' '.join(baseline_name.split('_')[1:])
            for technique in techniques:
                # AutoAugment is only considered as a high intensity augmentation. Skip in low.
                if (technique == 'autoaugment' and plot_name == 'low_intensity'):
                    continue
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
        plot.set_xlim(-0.5, 0.5)
        fig = plot.get_figure()
        fig.savefig(os.path.join('figures', f'{plot_name}_boxplot_ava_photozilla.eps'))
        plt.close()

    for experiments, plot_name in [(low_intensity_cel, 'low_intensity'), (high_intensity_cel, 'high_intensity')]:
        results_list_celeba = []
        for experiment in experiments:
            baseline_name = f"baseline_{'_'.join(experiment.split('_')[4:])}"
            dataset_name = ' '.join(baseline_name.split('_')[1:])
            for technique in techniques:
                # AutoAugment is only considered as a high intensity augmentation. Skip in low.
                if (technique == 'autoaugment' and plot_name == 'low_intensity'):
                    continue
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
        plot.set_xlim(-0.5, 0.5)
        fig = plot.get_figure()
        fig.savefig(os.path.join('figures', f'{plot_name}_boxplot_celeba.eps'))
        plt.close()

    sns.set_palette(palette_agg)
    # Generate two boxplots in the same figure, for the aggreagated boxplots
    two_agg_boxplots, axes = plt.subplots(1,2, figsize=(8.27*scale, 11.69*scale))
    plt.subplots_adjust(wspace=0.05, hspace=0)
    plt.close()
    i = 0

    for experiments, plot_name in [(low_intensity_ava + low_intensity_ava_reg + low_intensity_photozilla + low_intensity_cel, 'low_intensity'), (high_intensity_ava + high_intensity_ava_reg + high_intensity_photozilla + high_intensity_cel, 'high_intensity')]:
        results_list_aggregated = []
        for experiment in experiments:
            baseline_name = f"baseline_{'_'.join(experiment.split('_')[4:])}"
            dataset_name = ' '.join(baseline_name.split('_')[1:])
            if dataset_name not in ['celeba attractive']:
                if dataset_name[0] == 'p':
                    dataset_name = 'Photozilla'
                elif dataset_name[0] == 'c':
                    dataset_name = 'CelebA objective'
                elif "regression" in dataset_name:
                    dataset_name = 'AVA small (regression)'
                elif dataset_name == "ava small":
                    dataset_name = 'AVA small (classification)'
                else:
                    raise ValueError(f'Unexpected dataset! Got {dataset_name}')
                
            for technique in techniques:
                for rate in rates:
                    # AutoAugment is only considered as a high intensity augmentation. Skip in low.
                    if (technique == 'autoaugment' and plot_name == 'low_intensity'):
                        result_dict = {
                            'Augmentation technique': technique,
                            'Dataset': dataset_name.capitalize(),
                            'Balanced accuracy difference': 0
                        }
                    else:
                        with open(os.path.join('augmentation-results', experiment, f'{technique}_{rate}_results.json'), 'r') as f:
                            results = json.load(f)
                            result_dict = {
                                'Augmentation technique': technique,
                                'Dataset': dataset_name.capitalize(),
                                'Balanced accuracy difference': results['binary_balanced_accuracy'] - baseline_results[baseline_name]
                            }
                    results_list_aggregated.append(result_dict)

        df = pd.DataFrame.from_records(results_list_aggregated)
        plot_sf = sns.boxplot(
            data=df, 
            x='Balanced accuracy difference', 
            y='Augmentation technique', 
            hue='Dataset', 
            medianprops=dict(color="#008000", alpha=1.0),
            boxprops={'zorder':10},
            whiskerprops={'zorder':10},
            zorder=10,
            ax=axes[i]
        )
        plot_sf.legend().set_visible(False)
        plot_sf.set_title(plot_name.replace('_', ' ').capitalize(), fontsize=20)

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
        i += 1

        plot.axvline(x=0, ymin=0, ymax=1, color='black', zorder=1, linestyle=":")
        plot.set_xlim(-0.5, 0.5)
        plot_sf.axvline(x=0, ymin=0, ymax=1, color='black', zorder=1, linestyle=":")
        plot_sf.set_xlim(-0.5, 0.5)
        fig = plot.get_figure()
        fig.savefig(os.path.join('figures', f'{plot_name}_boxplot_aggregated.eps'))
        plt.close()

    axes[1].set_yticklabels([])
    axes[1].set_ylabel('')
    two_agg_boxplots.suptitle('Balanced accuracy difference', y = 0.09)
    axes[0].set_xlabel('')
    axes[1].set_xlabel('')
    two_agg_boxplots.legend(handles=axes[0].get_legend_handles_labels()[0], loc='center', bbox_to_anchor=(0.5, 0.93), fontsize=20, ncol=4)
    two_agg_boxplots.savefig(os.path.join('figures', 'two_boxplot_aggregated.eps'))
    plt.close()

    # Generate boxplots for AVA-small-regression
    for experiments, plot_name in [(low_intensity_ava_reg, 'low_intensity'), (high_intensity_ava_reg, 'high_intensity')]:
        results_list_ava_regression = []
        for experiment in experiments:
            baseline_name = f"baseline_{'_'.join(experiment.split('_')[4:])}"
            dataset_name = ' '.join(baseline_name.split('_')[1:])
            for technique in techniques:
                # AutoAugment is only considered as a high intensity augmentation. Skip in low.
                if (technique == 'autoaugment' and plot_name == 'low_intensity'):
                    continue
                for rate in rates:
                    with open(os.path.join('augmentation-results', experiment, f'{technique}_{rate}_results.json'), 'r') as f:
                        results = json.load(f)
                        result_dict = {
                            'Augmentation technique': technique,
                            'Dataset': dataset_name.capitalize(),
                            'Balanced accuracy difference': results['binary_balanced_accuracy'] - baseline_results[baseline_name]
                        }
                        results_list_ava_regression.append(result_dict)

        df = pd.DataFrame.from_records(results_list_ava_regression)
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
        plot.set_xlim(-0.5, 0.5)
        fig = plot.get_figure()
        fig.savefig(os.path.join('figures', f'{plot_name}_boxplot_ava_regression.eps'))
        plt.close()

    # Generate fully-aggregated boxplot, for all intensities and probabilities
    for experiment in low_intensity_ava + low_intensity_ava_reg + low_intensity_photozilla + low_intensity_cel + high_intensity_ava + high_intensity_ava_reg + high_intensity_photozilla + high_intensity_cel:
        baseline_name = f"baseline_{'_'.join(experiment.split('_')[4:])}"
        dataset_name = ' '.join(baseline_name.split('_')[1:])
        if dataset_name not in ['celeba attractive']:
            if dataset_name[0] == 'p':
                dataset_name = 'Photozilla'
            elif dataset_name[0] == 'c':
                dataset_name = 'CelebA objective'
            elif "regression" in dataset_name:
                dataset_name = 'AVA-small (regression)'
            elif dataset_name == "ava small":
                    dataset_name = 'AVA small (classification)'
            else:
                raise ValueError(f'Unexpected dataset! Got {dataset_name}')
            
        for technique in techniques:
        # AutoAugment is only considered as a high intensity augmentation. Skip in low.
            if (technique == 'autoaugment' and 'low_intensity' in experiment):
                continue
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
    plot.legend().set_visible(False)
    plot.legend(loc='center', bbox_to_anchor=(0.5, 1.05), fontsize=20, ncol=4)
    plot.axvline(x=0, ymin=0, ymax=1, color='black', zorder=1, linestyle=":")
    plot.set_xlim(-0.5, 0.5)
    plot.set_xlabel('Balanced accuracy difference', fontsize=25)
    fig = plot.get_figure()
    fig.savefig(os.path.join('figures', f'boxplot_full_aggregated.eps'))
    plt.close()

    # Generate fully-aggregated boxplots for AVA regression experiments
    for experiment in low_intensity_ava_reg + high_intensity_ava_reg:
        baseline_name = f"baseline_{'_'.join(experiment.split('_')[4:])}"
        dataset_name = ' '.join(baseline_name.split('_')[1:])
        for technique in techniques:
            # AutoAugment is only considered as a high intensity augmentation. Skip in low.
            if (technique == 'autoaugment' and 'low_intensity' in experiment):
                continue
            for rate in rates:
                with open(os.path.join('augmentation-results', experiment, f'{technique}_{rate}_results.json'), 'r') as f:
                    results = json.load(f)
                    result_dict = {
                        'Augmentation technique': technique,
                        'Dataset': dataset_name.capitalize(),
                        'Balanced accuracy difference': results['binary_balanced_accuracy'] - baseline_results[baseline_name]
                    }
                    results_list_full_aggregated_regression.append(result_dict)

    df = pd.DataFrame.from_records(results_list_full_aggregated_regression)
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
    plot.set_xlim(-0.5, 0.5)
    fig = plot.get_figure()
    fig.savefig(os.path.join('figures', f'boxplot_full_aggregated_regression.eps'))
    plt.close()

if __name__ == "__main__":
    main()