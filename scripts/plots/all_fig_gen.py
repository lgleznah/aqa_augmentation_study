import os
import sys
import matplotlib.pyplot as plt
import matplotlib
from one_techs_rates_plot import fig_gen as rates_fig_gen
from one_techs_intensities_plot import fig_gen as intensities_fig_gen
from experiment_lists import baselines, low_intensity, high_intensity, low_probability, high_probability, rates

scale = 3
margin_frac = 0.05

def main():

    experiment_supergroup = sys.argv[1]
    base = baselines[experiment_supergroup]
    low, high = low_intensity[experiment_supergroup], high_intensity[experiment_supergroup]

    matplotlib.rc('font', size=22)

    for experiments, plot_name in [(low, 'low_intensity'), (high, 'high_intensity')]:
        fig = plt.figure(figsize=(8.27*scale, 11.69*scale))
        w, h = 0.5 - margin_frac * 2, 0.15 - margin_frac
        axes = {
            'ax0': fig.add_axes([0.25 + margin_frac, 0.85 - margin_frac, w, h]),
            'ax1': fig.add_axes([0 + margin_frac, 0.7 - margin_frac, w, h]),
            'ax2': fig.add_axes([0.5 + margin_frac, 0.7 - margin_frac, w, h]),
            'ax3': fig.add_axes([0 + margin_frac, 0.55 - margin_frac, w, h]),
            'ax4': fig.add_axes([0.5 + margin_frac, 0.55 - margin_frac, w, h]),
            'ax5': fig.add_axes([0 + margin_frac, 0.4 - margin_frac, w, h]),
            'ax6': fig.add_axes([0.5 + margin_frac, 0.4 - margin_frac, w, h]),
            'ax7': fig.add_axes([0 + margin_frac, 0.25 - margin_frac, w, h]),
            'ax8': fig.add_axes([0.5 + margin_frac, 0.25 - margin_frac, w, h]),
            'ax9': fig.add_axes([0 + margin_frac, 0.1 - margin_frac, w, h]),
            'ax10': fig.add_axes([0.5 + margin_frac, 0.1 - margin_frac, w, h])
        }

        # Plot AVA small part
        rates_fig_gen(experiments[0], 'binary', base[0], rates, ax=axes['ax0'], metric='binary_balanced_accuracy', legend=False)

        # Plot Photozilla experiments
        for exp_number, experiment in enumerate(experiments[1:], 1):
            rates_fig_gen(experiment, 'binary', base[exp_number], rates, ax=axes[f'ax{exp_number}'], metric='binary_balanced_accuracy', legend=False)

        fig.legend(axes['ax0'].lines, ['Brightness', 'Contrast', 'Flip', 'Rotation', 'Translation', 'Zoom', 'Autoaugment', 'Baseline'])
        plt.savefig(os.path.join('figures', f'{experiment_supergroup}_{plot_name}.eps'))
        plt.close()

    '''
    for experiments, plot_name in [(low_probability, 'low_probability'), (high_probability, 'high_probability')]:
        fig = plt.figure(figsize=(8.27*scale, 11.69*scale))
        w, h = (0.5 - margin_frac * 2) / 2, 0.15 - margin_frac
        axes = {
            'ax0': (
                fig.add_axes([0.25 + margin_frac, 0.85 - margin_frac, w, h]),
                fig.add_axes([0.25 + margin_frac * 1.5 + w, 0.85 - margin_frac, w, h])),
            'ax1': (
                fig.add_axes([0 + margin_frac, 0.7 - margin_frac, w, h]),
                fig.add_axes([0 + margin_frac * 1.5 + w, 0.7 - margin_frac, w, h])),
            'ax2': (
                fig.add_axes([0.5 + margin_frac, 0.7 - margin_frac, w, h]),
                fig.add_axes([0.5 + margin_frac * 1.5 + w, 0.7 - margin_frac, w, h])),
            'ax3': (
                fig.add_axes([0 + margin_frac, 0.55 - margin_frac, w, h]),
                fig.add_axes([0 + margin_frac * 1.5 + w, 0.55 - margin_frac, w, h])),
            'ax4': (
                fig.add_axes([0.5 + margin_frac, 0.55 - margin_frac, w, h]),
                fig.add_axes([0.5 + margin_frac * 1.5 + w, 0.55 - margin_frac, w, h])),
            'ax5': (
                fig.add_axes([0 + margin_frac, 0.4 - margin_frac, w, h]),
                fig.add_axes([0 + margin_frac * 1.5 + w, 0.4 - margin_frac, w, h])),
            'ax6': (
                fig.add_axes([0.5 + margin_frac, 0.4 - margin_frac, w, h]),
                fig.add_axes([0.5 + margin_frac * 1.5 + w, 0.4 - margin_frac, w, h])),
            'ax7': (
                fig.add_axes([0 + margin_frac, 0.25 - margin_frac, w, h]),
                fig.add_axes([0 + margin_frac * 1.5 + w, 0.25 - margin_frac, w, h])),
            'ax8': (
                fig.add_axes([0.5 + margin_frac, 0.25 - margin_frac, w, h]),
                fig.add_axes([0.5 + margin_frac * 1.5 + w, 0.25 - margin_frac, w, h])),
            'ax9': (
                fig.add_axes([0 + margin_frac, 0.1 - margin_frac, w, h]),
                fig.add_axes([0 + margin_frac * 1.5 + w, 0.1 - margin_frac, w, h])),
            'ax10': (
                fig.add_axes([0.5 + margin_frac, 0.1 - margin_frac, w, h]),
                fig.add_axes([0.5 + margin_frac * 1.5 + w, 0.1 - margin_frac, w, h]))
        }

        # Plot AVA small part
        intensities_fig_gen(experiments[0], 'binary', baselines[0], rates, ax1=axes['ax0'][0], ax2=axes['ax0'][1], metric='binary_balanced_accuracy', legend=False)
        axes['ax0'][1].set_yticks([])

        # Plot Photozilla experiments
        for exp_number, experiment in enumerate(experiments[1:], 1):
            intensities_fig_gen(experiment, 'binary', baselines[exp_number], rates, ax1=axes[f'ax{exp_number}'][0], ax2=axes[f'ax{exp_number}'][1], metric='binary_balanced_accuracy', legend=False)
            axes[f'ax{exp_number}'][1].set_yticks([])

        fig.legend(axes['ax0'][0].lines, ['Brightness', 'Contrast', 'Rotation', 'Translation', 'Zoom', 'Baseline'])
        plt.savefig(os.path.join('figures', f'{plot_name}.eps'))
        plt.close()
        '''

if __name__ == "__main__":
    main()