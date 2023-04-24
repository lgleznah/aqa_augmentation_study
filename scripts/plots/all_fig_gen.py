import os
import matplotlib.pyplot as plt
import matplotlib
from one_techs_rates_plot import fig_gen

baselines = [
    'baseline_ava_small',
    'baseline_photozilla_ovr_aerial',
    'baseline_photozilla_ovr_architecture',
    'baseline_photozilla_ovr_event',
    'baseline_photozilla_ovr_fashion',
    'baseline_photozilla_ovr_food',
    'baseline_photozilla_ovr_nature',
    'baseline_photozilla_ovr_sports',
    'baseline_photozilla_ovr_street',
    'baseline_photozilla_ovr_wedding',
    'baseline_photozilla_ovr_wildlife',
]

low_intensity = [
    'one_techs_low_intensity_ava_small',
    'one_techs_low_intensity_photozilla_ovr_aerial',
    'one_techs_low_intensity_photozilla_ovr_architecture',
    'one_techs_low_intensity_photozilla_ovr_event',
    'one_techs_low_intensity_photozilla_ovr_fashion',
    'one_techs_low_intensity_photozilla_ovr_food',
    'one_techs_low_intensity_photozilla_ovr_nature',
    'one_techs_low_intensity_photozilla_ovr_sports',
    'one_techs_low_intensity_photozilla_ovr_street',
    'one_techs_low_intensity_photozilla_ovr_wedding',
    'one_techs_low_intensity_photozilla_ovr_wildlife',
]

high_intensity = [
    'one_techs_high_intensity_ava_small',
    'one_techs_high_intensity_photozilla_ovr_aerial',
    'one_techs_high_intensity_photozilla_ovr_architecture',
    'one_techs_high_intensity_photozilla_ovr_event',
    'one_techs_high_intensity_photozilla_ovr_fashion',
    'one_techs_high_intensity_photozilla_ovr_food',
    'one_techs_high_intensity_photozilla_ovr_nature',
    'one_techs_high_intensity_photozilla_ovr_sports',
    'one_techs_high_intensity_photozilla_ovr_street',
    'one_techs_high_intensity_photozilla_ovr_wedding',
    'one_techs_high_intensity_photozilla_ovr_wildlife',
]

rates = [25, 50, 75, 100]

scale = 3
margin_frac = 0.05

def main():

    matplotlib.rc('font', size=22)

    for experiments, plot_name in [(low_intensity, 'low_intensity'), (high_intensity, 'high_intensity')]:
        fig = plt.figure(figsize=(8.27*scale, 11.69*scale))
        axes = {
            'ax0': fig.add_axes([0.25 + margin_frac, 0.85 - margin_frac, 0.5 - margin_frac * 2, 0.15 - margin_frac]),
            'ax1': fig.add_axes([0 + margin_frac, 0.7 - margin_frac, 0.5 - margin_frac * 2, 0.15 - margin_frac]),
            'ax2': fig.add_axes([0.5 + margin_frac, 0.7 - margin_frac, 0.5 - margin_frac * 2, 0.15 - margin_frac]),
            'ax3': fig.add_axes([0 + margin_frac, 0.55 - margin_frac, 0.5 - margin_frac * 2, 0.15 - margin_frac]),
            'ax4': fig.add_axes([0.5 + margin_frac, 0.55 - margin_frac, 0.5 - margin_frac * 2, 0.15 - margin_frac]),
            'ax5': fig.add_axes([0 + margin_frac, 0.4 - margin_frac, 0.5 - margin_frac * 2, 0.15 - margin_frac]),
            'ax6': fig.add_axes([0.5 + margin_frac, 0.4 - margin_frac, 0.5 - margin_frac * 2, 0.15 - margin_frac]),
            'ax7': fig.add_axes([0 + margin_frac, 0.25 - margin_frac, 0.5 - margin_frac * 2, 0.15 - margin_frac]),
            'ax8': fig.add_axes([0.5 + margin_frac, 0.25 - margin_frac, 0.5 - margin_frac * 2, 0.15 - margin_frac]),
            'ax9': fig.add_axes([0 + margin_frac, 0.1 - margin_frac, 0.5 - margin_frac * 2, 0.15 - margin_frac]),
            'ax10': fig.add_axes([0.5 + margin_frac, 0.1 - margin_frac, 0.5 - margin_frac * 2, 0.15 - margin_frac])
        }

        # Plot AVA small part
        fig_gen(experiments[0], 'binary', baselines[0], rates, ax=axes['ax0'], metric='binary_balanced_accuracy', legend=False)

        # Plot Photozilla experiments
        for exp_number, experiment in enumerate(experiments[1:], 1):
            fig_gen(experiment, 'binary', baselines[exp_number], rates, ax=axes[f'ax{exp_number}'], metric='binary_balanced_accuracy', legend=False)

        fig.legend(axes['ax0'].lines, ['Brightness', 'Contrast', 'Flip', 'Rotation', 'Translation', 'Zoom', 'Baseline'])
        plt.savefig(os.path.join('figures', f'{plot_name}.eps'))
        plt.close()

if __name__ == "__main__":
    main()