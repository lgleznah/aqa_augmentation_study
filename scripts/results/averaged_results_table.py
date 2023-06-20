import os
import json

def main():

    table_template = """\\begin{{tabular}}{{c||c||c|c|c|c|c|c}}
         & \\textbf{{Baseline}} & Brightness & Contrast & Flip & Rotation & Translation & Zoom  \\\\ \\hhline{{=#=#======}}
         AVA & {AVABASE:.4f} & {AVAB:.4f} & {AVAC:.4f} & {AVAF:.4f} & {AVAR:.4f} & {AVAT:.4f} & {AVAZ:.4f} \\\\ \\hhline{{=#=#======}}
         Photozilla-aerial & {PHAEBASE:.4f} & {PHAEB:.4f} & {PHAEC:.4f} & {PHAEF:.4f} & {PHAER:.4f} & {PHAET:.4f} & {PHAEZ:.4f} \\\\ \\hline
         Photozilla-architecture & {PHARBASE:.4f} & {PHARB:.4f} & {PHARC:.4f} & {PHARF:.4f} & {PHARR:.4f} & {PHART:.4f} & {PHARZ:.4f} \\\\ \\hline
         Photozilla-event & {PHEVBASE:.4f} & {PHEVB:.4f} & {PHEVC:.4f} & {PHEVF:.4f} & {PHEVR:.4f} & {PHEVT:.4f} & {PHEVZ:.4f} \\\\ \\hline
         Photozilla-fashion & {PHFABASE:.4f} & {PHFAB:.4f} & {PHFAC:.4f} & {PHFAF:.4f} & {PHFAR:.4f} & {PHFAT:.4f} & {PHFAZ:.4f} \\\\ \\hline
         Photozilla-food & {PHFOBASE:.4f} & {PHFOB:.4f} & {PHFOC:.4f} & {PHFOF:.4f} & {PHFOR:.4f} & {PHFOT:.4f} & {PHFOZ:.4f} \\\\ \\hline
         Photozilla-nature & {PHNABASE:.4f} & {PHNAB:.4f} & {PHNAC:.4f} & {PHNAF:.4f} & {PHNAR:.4f} & {PHNAT:.4f} & {PHNAZ:.4f} \\\\ \\hline
         Photozilla-sports & {PHSPBASE:.4f} & {PHSPB:.4f} & {PHSPC:.4f} & {PHSPF:.4f} & {PHSPR:.4f} & {PHSPT:.4f} & {PHSPZ:.4f} \\\\ \\hline
         Photozilla-street & {PHSTBASE:.4f} & {PHSTB:.4f} & {PHSTC:.4f} & {PHSTF:.4f} & {PHSTR:.4f} & {PHSTT:.4f} & {PHSTZ:.4f} \\\\ \\hline
         Photozilla-wedding & {PHWEBASE:.4f} & {PHWEB:.4f} & {PHWEC:.4f} & {PHWEF:.4f} & {PHWER:.4f} & {PHWET:.4f} & {PHWEZ:.4f} \\\\ \\hline
         Photozilla-wildlife & {PHWIBASE:.4f} & {PHWIB:.4f} & {PHWIC:.4f} & {PHWIF:.4f} & {PHWIR:.4f} & {PHWIT:.4f} & {PHWIZ:.4f} \\\\ \\hhline{{=#=#======}}
         CelebA-attractive & {CLATBASE:.4f} & {CLATB:.4f} & {CLATC:.4f} & {CLATF:.4f} & {CLATR:.4f} & {CLATT:.4f} & {CLATZ:.4f} \\\\ \\hhline{{=#=#======}}
         CelebA-earrings & {CLERBASE:.4f} & {CLERB:.4f} & {CLERC:.4f} & {CLERF:.4f} & {CLERR:.4f} & {CLERT:.4f} & {CLERZ:.4f} \\\\ \\hline
         CelebA-eyeglasses & {CLEYBASE:.4f} & {CLEYB:.4f} & {CLEYC:.4f} & {CLEYF:.4f} & {CLEYR:.4f} & {CLEYT:.4f} & {CLEYZ:.4f} \\\\ \\hline
         CelebA-hair & {CLHABASE:.4f} & {CLHAB:.4f} & {CLHAC:.4f} & {CLHAF:.4f} & {CLHAR:.4f} & {CLHAT:.4f} & {CLHAZ:.4f} \\\\ \\hline
         CelebA-hat & {CLHTBASE:.4f} & {CLHTB:.4f} & {CLHTC:.4f} & {CLHTF:.4f} & {CLHTR:.4f} & {CLHTT:.4f} & {CLHTZ:.4f} \\\\ \\hline
         CelebA-male & {CLMABASE:.4f} & {CLMAB:.4f} & {CLMAC:.4f} & {CLMAF:.4f} & {CLMAR:.4f} & {CLMAT:.4f} & {CLMAZ:.4f} \\\\ \\hline
         CelebA-necklace & {CLNLBASE:.4f} & {CLNLB:.4f} & {CLNLC:.4f} & {CLNLF:.4f} & {CLNLR:.4f} & {CLNLT:.4f} & {CLNLZ:.4f} \\\\ \\hline
         CelebA-necktie & {CLNTBASE:.4f} & {CLNTB:.4f} & {CLNTC:.4f} & {CLNTF:.4f} & {CLNTR:.4f} & {CLNTT:.4f} & {CLNTZ:.4f} \\\\ \\hline
         CelebA-straight hair & {CLSTBASE:.4f} & {CLSTB:.4f} & {CLSTC:.4f} & {CLSTF:.4f} & {CLSTR:.4f} & {CLSTT:.4f} & {CLSTZ:.4f} \\\\ \\hline
         CelebA-wavy hair & {CLWABASE:.4f} & {CLWAB:.4f} & {CLWAC:.4f} & {CLWAF:.4f} & {CLWAR:.4f} & {CLWAT:.4f} & {CLWAZ:.4f} \\\\ \\hline
         CelebA-young & {CLYOBASE:.4f} & {CLYOB:.4f} & {CLYOC:.4f} & {CLYOF:.4f} & {CLYOR:.4f} & {CLYOT:.4f} & {CLYOZ:.4f} \\\\ \\hline
    \\end{{tabular}}"""

    datasets = {
        'ava_small': 'AVA',
        'photozilla_ovr_aerial': 'PHAE',
        'photozilla_ovr_architecture': 'PHAR',
        'photozilla_ovr_event': 'PHEV',
        'photozilla_ovr_fashion': 'PHFA',
        'photozilla_ovr_food': 'PHFO',
        'photozilla_ovr_nature': 'PHNA',
        'photozilla_ovr_sports': 'PHSP',
        'photozilla_ovr_street': 'PHST',
        'photozilla_ovr_wedding': 'PHWE',
        'photozilla_ovr_wildlife': 'PHWI',
        'celeba_attractive': 'CLAT',
        'celeba_earrings': 'CLER',
        'celeba_eyeglasses': 'CLEY',
        'celeba_hair': 'CLHA',
        'celeba_hat': 'CLHT',
        'celeba_male': 'CLMA',
        'celeba_necklace': 'CLNL',
        'celeba_necktie': 'CLNT',
        'celeba_strhair': 'CLST',
        'celeba_wavyhair': 'CLWA',
        'celeba_young': 'CLYO'
    }

    augmentations = {
        'brightness': 'B',
        'contrast': 'C',
        'flip': 'F',
        'rotation': 'R',
        'translation': 'T',
        'zoom': 'Z'
    }

    experiments = ['one_techs_low_intensity', 'one_techs_high_intensity']

    percentages = ['25', '50', '75', '100']

    results_path = os.path.join(os.environ['AQA_AUGMENT_ROOT'], 'augmentation-results')
    baseline_results = {}
    experiment_results = {exp: {} for exp in experiments}

    # Get baseline models results
    for dataset in datasets:
        abbreviation = f'{datasets[dataset]}BASE'
        experiment_group_name = f'baseline_{dataset}'
        experiment_path = os.path.join(results_path, experiment_group_name, 'baseline_results.json')
        with open(experiment_path, 'r') as f:
            result_dict = json.load(f)
        baseline_results.update({abbreviation: result_dict['binary_balanced_accuracy']})

    # Get aggregated results for experiments
    for experiment in experiments:

        for dataset in datasets:
            dataset_abbreviated = datasets[dataset]

            for augmentation in augmentations:
                augmentation_abbreviated = augmentations[augmentation]
                abbreviation = f'{dataset_abbreviated}{augmentation_abbreviated}'
                experiment_group_name = f'{experiment}_{dataset}'

                average_balacc = 0
                for percentage in percentages:
                    experiment_path = os.path.join(results_path, experiment_group_name, f'{augmentation}_{percentage}_results.json')
                    with open(experiment_path, 'r') as f:
                        result_dict = json.load(f)
                    average_balacc += result_dict['binary_balanced_accuracy']
                average_balacc /= 4
                experiment_results[experiment].update({abbreviation: average_balacc})

        print(f'Results for {experiment}')
        print(table_template.format(**({**baseline_results, **experiment_results[experiment]})))

if __name__ == "__main__":
    main()