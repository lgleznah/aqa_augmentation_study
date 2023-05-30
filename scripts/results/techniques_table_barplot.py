import os
import json
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    table_template = """\\begin{{tabular}}{{c||c|c|c|c|c|c}}
         & Brightness & Contrast & Flip & Rotation & Translation & Zoom  \\\\ \\hhline{{=#=#======}}
         AVA & {AVAB:.4f} & {AVAC:.4f} & {AVAF:.4f} & {AVAR:.4f} & {AVAT:.4f} & {AVAZ:.4f} \\\\ \\hline
         Photozilla & {PHB:.4f} & {PHC:.4f} & {PHF:.4f} & {PHR:.4f} & {PHT:.4f} & {PHZ:.4f} \\\\ \\hline
         CelebA attractiveness & {CELSB:.4f} & {CELSC:.4f} & {CELSF:.4f} & {CELSR:.4f} & {CELST:.4f} & {CELSZ:.4f} \\\\ \\hline
         CelebA objective tasks & {CELOB:.4f} & {CELOC:.4f} & {CELOF:.4f} & {CELOR:.4f} & {CELOT:.4f} & {CELOZ:.4f} \\\\ \\hline
    \\end{{tabular}}"""

    palette = ["#b5838d", "#ffb4a2", "#132a13", "#31572c", "#90a955", "#ecf39e"]

    datasets_per_group = {
        'AVA': ['ava_small'],
        'PH': [
            'photozilla_ovr_aerial',
            'photozilla_ovr_architecture',
            'photozilla_ovr_event',
            'photozilla_ovr_fashion',
            'photozilla_ovr_food',
            'photozilla_ovr_nature',
            'photozilla_ovr_sports',
            'photozilla_ovr_street',
            'photozilla_ovr_wedding',
            'photozilla_ovr_wildlife'
        ],
        'CELS': ['celeba_photos'],
        'CELO': [
            'celeba_photos',
            'celeba_earrings',
            'celeba_eyeglasses',
            'celeba_hair',
            'celeba_hat',
            'celeba_male',
            'celeba_necklace',
            'celeba_necktie',
            'celeba_strhair',
            'celeba_wavyhair',
            'celeba_young'
        ]
    }

    augmentations = {
        'brightness': 'B',
        'contrast': 'C',
        'flip': 'F',
        'rotation': 'R',
        'translation': 'T',
        'zoom': 'Z'
    }

    intensities = ['one_techs_low_intensity', 'one_techs_high_intensity']

    percentages = ['25', '50', '75', '100']

    results_path = os.path.join(os.environ['AQA_AUGMENT_ROOT'], 'augmentation-results')
    baseline_results = {}
    experiment_results = {inte: {} for inte in intensities}
    
    # Get baseline models results
    for group in datasets_per_group:
        for dataset in datasets_per_group[group]:
            experiment_group_name = f'baseline_{dataset}'
            experiment_path = os.path.join(results_path, experiment_group_name, 'baseline_results.json')
            with open(experiment_path, 'r') as f:
                result_dict = json.load(f)
            baseline_results.update({dataset: result_dict['binary_balanced_accuracy']})

    # Get aggregated results
    for intensity in intensities:
        for group in datasets_per_group:
            for augmentation in augmentations:
                group_abbreviation = group
                augmentation_abbreviation = augmentations[augmentation]
                abbreviation = f'{group_abbreviation}{augmentation_abbreviation}'

                average_balacc_delta = 0
                for dataset in datasets_per_group[group]:
                        for percentage in percentages:
                            experiment_group_name = f'{intensity}_{dataset}'
                            experiment_path = os.path.join(results_path, experiment_group_name, f'{augmentation}_{percentage}_results.json')
                            with open(experiment_path, 'r') as f:
                                result_dict = json.load(f)
                            average_balacc_delta += (result_dict['binary_balanced_accuracy'] - baseline_results[dataset])
                
                average_balacc_delta /= (4 * len(datasets_per_group[group]))
                experiment_results[intensity].update({abbreviation: average_balacc_delta})

        print(f'Results for {intensity}')
        print(table_template.format(**experiment_results[intensity]))

    # Get final barplot
    total_averages = []
    for augmentation in augmentations:
        avg_balacc = 0
        for intensity in intensities:
            for group in datasets_per_group:
                avg_balacc += experiment_results[intensity][f'{group}{augmentations[augmentation]}']
        
        avg_balacc /= 8
        total_averages.append(avg_balacc)

    sns.set_palette(palette)
    ax = sns.barplot(x=['Brightness','Contrast','Flip','Rotation','Translation','Zoom'], y=total_averages)
    ax.set(xlabel="Augmentation technique", ylabel="Average balanced accuracy difference")
    fig = ax.get_figure()
    fig.savefig(os.path.join('figures', f'averages_barplot.eps'))
    plt.close()


if __name__ == "__main__":
    main()