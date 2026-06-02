import json
from codes.utils.misc.stats import *
import seaborn as sns
import matplotlib.pyplot as plt
from codes.utils.misc.plot_on_grid import plot_grid_on_allen
from codes.utils.misc.fig_saving import save_fig
from codes.utils.misc.table_saving import save_table


def figure2cde(data_table, trial_table, saving_path, supp_saving_path, saving_formats):
    data_table['shuffle_dist_sub'] = data_table['shuffle_dist_sub'].apply(json.loads)

    avg_df = data_table.groupby(by=['context', 'trial_type', 'opto_grid_ml', 'opto_grid_ap']).agg(
        data=('data_mean', list),
        data_sub=('data_mean_sub', list),
        data_mean=('data_mean', 'mean'),
        data_mean_sub=(
            'data_mean_sub', 'mean'),
        shuffle_dist=('shuffle_dist', 'sum'),
        shuffle_dist_sub=(
            'shuffle_dist_sub', 'sum'),
        percentile_avg=('percentile', 'mean'),
        percentile_avg_sub=(
            'percentile_sub', 'mean'),
        n_sigma_avg=('n_sigma', 'mean'),
        n_sigma_avg_sub=(
            'n_sigma_sub', 'mean'))

    avg_df['d_sub'] = avg_df.apply(lambda x: abs(cohen_d(x.shuffle_dist_sub, x.data_sub)), axis=1)
    avg_df = avg_df.reset_index()

    dprime_palette = 'binary'
    seismic_palette = sns.diverging_palette(265, 10, s=100, l=40, sep=60, n=200, center="light", as_cmap=True)

    fig, axes = plt.subplots(2, 3, figsize=(8, 6))
    fig.suptitle(f'Opto grid control subtracted')
    fig1, axes1 = plt.subplots(2, 3, figsize=(8, 6))
    fig1.suptitle(f'd prime subtracted')
    fig2, axes2 = plt.subplots(2, 3, figsize=(8, 6))
    fig2.suptitle(f'Opto grid raw P lick')
    fig3, axes3 = plt.subplots(2, 3, figsize=(8, 6))
    fig3.suptitle(f'Opto grid density')

    density_grids = []
    for name, group in avg_df.groupby(by=['context', 'trial_type']):
        if 'whisker_trial' in name:
            outcome = 'outcome_w'
            col = 2
        elif 'auditory_trial' in name:
            outcome = 'outcome_a'
            col = 1
        else:
            outcome = 'outcome_n'
            col = 0

        row = 0 if name[0] == 'rewarded' else 1

        group.rename(columns={'opto_grid_ml': 'x', 'opto_grid_ap': 'y'}, inplace=True)
        data_trial = trial_table.groupby(by=['context', 'trial_type']).get_group((0 if name[0] == 'non-rewarded' else 1, name[1]))
        stim = data_trial.loc[data_trial.opto_stim == 1].drop_duplicates()
        density_grid = stim.groupby(by=['opto_grid_ml', 'opto_grid_ap'])[outcome].count().reset_index()
        density_grid.rename(columns={'opto_grid_ml': 'x', 'opto_grid_ap': 'y'}, inplace=True)
        density_grid['context'] = name[0]
        density_grid['trial_type'] = name[1]
        density_grids.append(density_grid)

        # CONTROL SUBTRACTED GRID
        fig, axes[row, col] = plot_grid_on_allen(group, outcome=f"data_mean_sub", palette=seismic_palette, facecolor=None,
                                                 edgecolor='black', vmin=-0.3,
                                                 vmax=0.3, dotsize=230, fig=fig, ax=axes[row, col], result_path=None)

        # D' FROM CONTROL SUBTRACTED GRID
        fig1, axes1[row, col] = plot_grid_on_allen(group, outcome="d_sub", palette=dprime_palette,
                                                   vmin=0.5, facecolor=None, edgecolor='black',
                                                   vmax=2.2, dotsize=230, fig=fig1,
                                                   ax=axes1[row, col], result_path=None)

        # RAW P LICK
        fig2, axes2[row, col] = plot_grid_on_allen(group, outcome="data_mean", palette='viridis',
                                                   vmin=0, facecolor=None, edgecolor='black',
                                                   vmax=1, dotsize=230, fig=fig2,
                                                   ax=axes2[row, col], result_path=None)

        # TRIAL DENSITY
        fig3, axes3[row, col] = plot_grid_on_allen(density_grid, outcome=outcome, palette='viridis',
                                                   vmin=0, facecolor=None, edgecolor='black',
                                                   vmax=90,
                                                   # vmax=0.8 * density_grid[outcome].max(),
                                                   dotsize=230, fig=fig3,
                                                   ax=axes3[row, col], result_path=None)

    cols = ['No stim', 'Auditory', 'Whisker']
    rows_labels = ['W+', 'W-']

    for ax_grid in [axes, axes1, axes2, axes3]:
        for a, col in zip(ax_grid[0], cols):
            a.set_title(col)

    for panel in [fig, fig1, fig2, fig3]:
            panel.tight_layout()
            panel.subplots_adjust(left=0.1)
            for i, row_label in enumerate(rows_labels):
                panel.text(0.07, 0.75 - i * 0.5, row_label, va='center', rotation='vertical',
                        fontsize=12, transform=panel.transFigure)

    names = ['Figure2CDE_delta_plick', 'Figure2CDE_dprime']
    supp_names = ['Figure2_supp1B', 'Figure2_supp1A']
    for idx, panel in enumerate([fig, fig1]):
        save_fig(panel, saving_path, names[idx], formats=saving_formats)
    for idx, panel in enumerate([fig2, fig3]):
        save_fig(panel, supp_saving_path, supp_names[idx], formats=saving_formats)

    plt.close('all')

    density_grids = pd.concat(density_grids)
    density_grids.rename(columns={'outcome_a': 'auditory_count',
                                  'outcome_n': 'catch_count',
                                  'outcome_w': 'whisker_count'}, inplace=True)
    density_grids = density_grids.drop(['ml_wf', 'ap_wf'], axis=1)
    avg_table = avg_df.copy()
    avg_table.rename(columns={'opto_grid_ml': 'x', 'opto_grid_ap': 'y'}, inplace=True)
    avg_table = avg_table[['context', 'trial_type', 'x', 'y', 'data_mean', 'data_mean_sub', 'd_sub']]
    stats_df = pd.merge(avg_table, density_grids, on=['context', 'x', 'y', 'trial_type'], how='right')

    save_table(stats_df, saving_path, f'optogrid_results', format=['csv'])
