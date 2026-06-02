import os
import json
import ast
import itertools
import scipy
import glob
from codes.utils.misc.stats import *
import matplotlib.pyplot as plt
from codes.utils.misc.plot_on_grid import plot_grid_on_allen
from codes.utils.misc.fig_saving import save_fig
from codes.utils.misc.plot_utils import *


def plot_figure2_supp1de(data_table, saving_path, saving_formats):
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
        fig, axes[row, col] = plot_grid_on_allen(group, outcome=f"data_mean_sub", palette=seismic_palette,
                                                 facecolor=None,
                                                 edgecolor='black', vmin=-0.3,
                                                 vmax=0.3, dotsize=200, fig=fig, ax=axes[row, col], result_path=None)
        fig.tight_layout()

        fig1, axes1[row, col] = plot_grid_on_allen(group, outcome="d_sub", palette=dprime_palette,
                                                   vmin=0.5, facecolor=None, edgecolor='black',
                                                   vmax=2.2, dotsize=200, fig=fig1,
                                                   ax=axes1[row, col], result_path=None)
        fig1.tight_layout()

    cols = ['No stim', 'Auditory', 'Whisker']
    rows_labels = ['W+', 'W-']

    for ax_grid in [axes, axes1]:
        for a, col in zip(ax_grid[0], cols):
            a.set_title(col)

    for panel in [fig, fig1]:
            panel.tight_layout()
            panel.subplots_adjust(left=0.1)
            for i, row_label in enumerate(rows_labels):
                panel.text(0.07, 0.75 - i * 0.5, row_label, va='center', rotation='vertical',
                        fontsize=12, transform=panel.transFigure)


    names = ['Figure2_supp1D', 'Figure2_supp1E']
    for idx, panel in enumerate([fig, fig1]):
        save_fig(panel, saving_path, names[idx], formats=saving_formats)

    plt.close('all')


def plot_figure2_supp2abc(muscimol, ringer, saving_path, sites, names, saving_formats):
    for idx, site in enumerate(sites):
        muscimol_path = glob.glob(os.path.join(muscimol, site, '*', 'context_days_full_table.csv'))[0]
        ringer_path = glob.glob(os.path.join(ringer, site, '*', 'context_days_full_table.csv'))[0]
        muscimol_df = pd.read_csv(muscimol_path)
        ringer_df = pd.read_csv(ringer_path)

        muscimol_df['drug'] = 'Muscimol'
        ringer_df['drug'] = 'Ringer'
        df = pd.concat((muscimol_df, ringer_df))

        df = df.loc[df.artificial_day != 2]
        df = df.drop(['Unnamed: 0', 'session_id', 'day', 'context', 'context_background'], axis=1)

        hue_name = ['Rewarded', 'Non-Rewarded']
        context_palette = {
            'catch_palette': {hue_name[0]: 'darkgray', hue_name[1]: 'lightgrey'},
            'wh_palette': {hue_name[0]: 'green', hue_name[1]: 'darkmagenta'},
            'aud_palette': {hue_name[0]: 'mediumblue', hue_name[1]: 'cornflowerblue'}
        }

        figure, (ax0, ax1) = plt.subplots(1, 2, figsize=(2, 3), sharey=True)
        for outcome, palette_key in zip(['outcome_n', 'outcome_a', 'outcome_w'],
                                        ['catch_palette', 'aud_palette', 'wh_palette']):
            plot_with_point_and_strip(data=df.loc[df.drug == 'Muscimol'], x_name='artificial_day', y_name=outcome,
                                      hue='context_rwd_str', palette=context_palette, ax=ax1, palette_key=palette_key,
                                      link_mice=False)

            plot_with_point_and_strip(data=df.loc[df.drug == 'Ringer'], x_name='artificial_day', y_name=outcome,
                                      hue='context_rwd_str', palette=context_palette, ax=ax0, palette_key=palette_key,
                                      link_mice=False)
        ax0.set_title('Ringer')
        ax1.set_title('Muscimol')
        for ax in [ax0, ax1]:
            ax.get_legend().set_visible(False)
            ax_set(ax, ylim=[-0.1, 1.05], xlabel='Day', ylabel='Lick probability')
        figure.suptitle(site)
        figure.tight_layout()

        save_fig(figure, saving_path, f'Figure2_supp1{names[idx]}_left', formats=saving_formats)

        # Separate artificial_day 0 and 1 into different DataFrames
        day_0 = df[df['artificial_day'] == 0].set_index(['mouse_id', 'context_rwd_str', 'drug'])
        day_1 = df[df['artificial_day'] == 1].set_index(['mouse_id', 'context_rwd_str', 'drug'])

        # Ensure the indices are aligned before subtraction
        difference = day_1[['outcome_a', 'outcome_w', 'outcome_n']] - day_0[['outcome_a', 'outcome_w', 'outcome_n']]

        # Reset the index and rename columns
        difference = difference.reset_index()
        difference.columns = ['mouse_id', 'context_rwd_str', 'drug', 'outcome_a_diff', 'outcome_w_diff',
                              'outcome_n_diff']

        figure, axes = plt.subplots(1, 3, figsize=(5, 4), sharey=False)
        ax = 0
        titles = ['catch trials', 'auditory trials', 'whisker trials']
        for outcome, palette_key in zip(['outcome_n_diff', 'outcome_a_diff', 'outcome_w_diff'],
                                        ['wh_palette', 'wh_palette', 'wh_palette']):
            sns.stripplot(data=difference, x='drug', order=['Ringer', 'Muscimol'], y=outcome, hue='context_rwd_str',
                          palette=context_palette[palette_key], legend=False, dodge=True, ax=axes.flatten()[ax],
                          size=8)
            sns.pointplot(data=difference, x='drug', order=['Ringer', 'Muscimol'], y=outcome, hue='context_rwd_str',
                          palette=context_palette[palette_key],
                          legend=False, ax=axes.flatten()[ax], linestyle='none', estimator=np.nanmean, alpha=0.5,
                          dodge=True)
            axes.flatten()[ax].set_ylim(-1, 1)
            axes.flatten()[ax].set_title(titles[ax])
            axes.flatten()[ax].set_ylabel('Delta lick probability')
            axes.flatten()[ax].axhline(y=0, c='k', linestyle='--')
            ax += 1
        sns.despine()
        figure.suptitle(site)
        figure.tight_layout()
        save_fig(figure, saving_path, f'Figure2_supp1{names[idx]}_right', formats=saving_formats)


def plot_figure2_supp3a(data_table, saving_path, name, saving_formats):
    data_table = data_table.loc[data_table.trial_type == 'whisker_trial']

    selected_spots = ['(-1.5, 3.5)', '(-1.5, 4.5)', '(1.5, 1.5)', '(-1.5, 0.5)',
                      '(2.5, 2.5)', '(0.5, 4.5)']
    contexts = [1, 0]
    bodyparts = ['jaw_y']
    combinations = list(itertools.product(contexts, bodyparts, selected_spots))

    # fig, axes = plt.subplots(6, 6, figsize=(6, 3), sharex=True)
    fig, axes = plt.subplots(2, 6, figsize=(6, 3), sharex=True, sharey=True)
    for ax_idx, ax in enumerate(axes.flatten()):
        df_subplot = data_table.loc[(data_table.opto_stim_coord == combinations[ax_idx][2]) &
                                    (data_table.context == combinations[ax_idx][0])].copy()
        df_subplot[combinations[ax_idx][1]] = df_subplot[combinations[ax_idx][1]].apply(
            lambda s: np.fromstring(s.strip('[]'), sep=' '))
        time_vector = np.linspace(-1, 1.5, 250)
        df_subplot['time'] = [time_vector] * len(df_subplot)
        df_subplot = df_subplot[['time', 'mouse_id', combinations[ax_idx][1]]]
        df_long = df_subplot.explode([combinations[ax_idx][1], 'time'], ignore_index=True)
        df_long = df_long.loc[(df_long.time > -0.07) & (df_long.time < 0.15)]
        if combinations[ax_idx][1] == 'whisker_velocity':
            df_long[combinations[ax_idx][1]] = np.abs(df_long[combinations[ax_idx][1]])
        sns.lineplot(df_long, x='time', y=combinations[ax_idx][1], color='royalblue', ax=ax)

        df_subplot_ctrl = data_table.loc[(data_table.opto_stim_coord == '(-5.0, 5.0)') &
                                         (data_table.context == combinations[ax_idx][0])].copy()
        df_subplot_ctrl[combinations[ax_idx][1]] = df_subplot_ctrl[combinations[ax_idx][1]].apply(
            lambda s: np.fromstring(s.strip('[]'), sep=' '))
        df_subplot_ctrl['time'] = [time_vector] * len(df_subplot_ctrl)
        df_subplot_ctrl = df_subplot_ctrl[['time', 'mouse_id', combinations[ax_idx][1]]]
        df_subplot_ctrl_long = df_subplot_ctrl.explode([combinations[ax_idx][1], 'time'], ignore_index=True)
        df_subplot_ctrl_long = df_subplot_ctrl_long.loc[(df_subplot_ctrl_long.time > -0.07)
                                                        & (df_subplot_ctrl_long.time < 0.15)]
        if combinations[ax_idx][1] == 'whisker_velocity':
            df_subplot_ctrl_long[combinations[ax_idx][1]] = np.abs(df_subplot_ctrl_long[combinations[ax_idx][1]])
        sns.lineplot(df_subplot_ctrl_long, x='time', y=combinations[ax_idx][1], color='grey', ax=ax)

        sns.despine()
        ax.axvline(x=0, ymin=0, ymax=1, linestyle='--', c='k')
        ax.set_title(f'{"W+" if combinations[ax_idx][0] == 1 else "W-"}\n'
                     f'{combinations[ax_idx][2]}')

    for ax in axes.flatten():
        ax.set_ylim(-0.16, 2)

    fig.suptitle(f'Whisker trials')
    fig.tight_layout()

    save_fig(fig, saving_path, name, formats=saving_formats)


def plot_figure2_supp3b(data_table, saving_path, name, saving_formats):
    data_table = data_table.loc[data_table.trial_type == 'whisker_trial']

    selected_spots = ['(-5.0, 5.0)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(1.5, 1.5)', '(-1.5, 0.5)', '(2.5, 2.5)',
                      '(0.5, 4.5)']
    sub_df = data_table.loc[data_table.opto_stim_coord.isin(selected_spots)].copy()

    bodyparts_to_plot = ['jaw_y']

    for context in sub_df.context.unique():
        sub_df_ctx = sub_df.loc[sub_df.context == context].copy()
        for bpart in bodyparts_to_plot:
            col_to_keep = [bpart, 'mouse_id', 'context', 'trial_type', 'opto_stim_coord']
            df_plot = sub_df_ctx[col_to_keep].copy()
            df_plot[bpart] = df_plot[bpart].apply(lambda s: np.fromstring(s.strip('[]'), sep=' '))
            time_vector = np.linspace(-1, 1.5, 250)
            df_plot['time'] = [time_vector] * len(df_plot)
            auc = []
            for i in range(len(df_plot)):
                auc.append(scipy.integrate.simpson(df_plot.iloc[i][bpart][101: 116], np.arange(0, 15)))
            df_plot[f'auc'] = auc

            fig, ax = plt.subplots(1, 1, figsize=(4, 2))
            sns.barplot(data=df_plot, x='opto_stim_coord', y='auc', fill=False, color='black', order=selected_spots,
                        ax=ax)
            sns.scatterplot(data=df_plot, x='opto_stim_coord', y='auc', color='grey', alpha=0.5, ax=ax)
            sns.despine()
            ax.set_ylabel('AUC')
            ax.set_xlabel('Opto target')
            ax.set_title(f'{bpart} AUC {"W-" if context == 0 else "W+"}')
            fig.tight_layout()

            save_fig(fig, saving_path, f'{name}_{bpart}_{"W-" if context == 0 else "W+"}', formats=saving_formats)

            # Stats
            spot_list = []
            mean_list = []
            std_list = []
            mean_diff = []
            dprime = []
            t_values = []
            p_values = []
            ctrl_data = df_plot.loc[df_plot.opto_stim_coord == '(-5.0, 5.0)', 'auc'].values
            for target in selected_spots:
                data = df_plot.loc[df_plot.opto_stim_coord == target, 'auc'].values
                spot_list.append(target)
                mean_list.append(np.nanmean(data))
                std_list.append(np.nanstd(data))
                mean_diff.append(np.nanmean(data) - np.nanmean(ctrl_data))
                if target != '(-5.0, 5.0)':
                    dprime.append((np.nanmean(data) - np.nanmean(ctrl_data)) / 0.5 * (
                                np.nanstd(data) ** 2 + np.nanstd(ctrl_data) ** 2))
                    stat_res = st.ttest_rel(data, ctrl_data)
                    t_values.append(stat_res[0])
                    p_values.append(stat_res[1])
                else:
                    dprime.append(np.nan)
                    t_values.append(np.nan)
                    p_values.append(np.nan)
            stats_results = pd.DataFrame()
            stats_results['Spot'] = spot_list
            stats_results['Mean'] = mean_list
            stats_results['STD'] = std_list
            stats_results['MeanDiff'] = mean_diff
            stats_results['Dprime'] = dprime
            stats_results['T'] = t_values
            stats_results['p'] = p_values
            stats_results['significant'] = stats_results['p'] < 0.05 / 6
            stats_results.to_csv(
                os.path.join(saving_path, f'{name}_{bpart}_{"W-" if context == 0 else "W+"}_stat_results.csv'))


def plot_figure2_supp3c(data_table, saving_path, name, saving_formats):
    data_table = data_table.loc[data_table.trial_type == 'whisker_trial']

    bodyparts_to_plot = ['jaw_y']

    seismic_palette = sns.diverging_palette(265, 10, s=100, l=40, sep=30, n=200, center="light", as_cmap=True)

    for context in data_table.context.unique():
        sub_df = data_table.loc[data_table.context == context]
        for bpart in bodyparts_to_plot:
            col_to_keep = [bpart, 'mouse_id', 'context', 'trial_type', 'opto_stim_coord']
            df_plot = sub_df[col_to_keep].copy()
            df_plot[bpart] = df_plot[bpart].apply(lambda s: np.fromstring(s.strip('[]'), sep=' '))
            time_vector = np.linspace(-1, 1.5, 250)
            df_plot['time'] = [time_vector] * len(df_plot)
            auc = []
            for i in range(len(df_plot)):
                auc.append(scipy.integrate.simpson(df_plot.iloc[i][bpart][101: 116], np.arange(0, 15)))
            df_plot[f'auc'] = auc
            reference = df_plot[df_plot['opto_stim_coord'] == '(-5.0, 5.0)'][['mouse_id', 'auc']].rename(
                columns={'auc': 'ctrl_auc'})
            df_plot = df_plot.merge(reference, on='mouse_id', how='left')
            df_plot[f'delta_auc'] = df_plot[f'auc'] - df_plot[f'ctrl_auc']

            full_avg_df = df_plot.copy().drop(['mouse_id', bpart, 'context', 'trial_type', 'time'], axis=1).groupby(
                ['opto_stim_coord'], as_index=False).agg('mean').reset_index()
            full_avg_df['y'] = full_avg_df['opto_stim_coord'].apply(lambda x: ast.literal_eval(x)[0])
            full_avg_df['x'] = full_avg_df['opto_stim_coord'].apply(lambda x: ast.literal_eval(x)[1])

            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            plot_grid_on_allen(full_avg_df, outcome='delta_auc', palette=seismic_palette, facecolor=None,
                               edgecolor='black',
                               vmin=-(np.max(np.abs(full_avg_df.delta_auc))),
                               vmax=np.max(np.abs(full_avg_df.delta_auc)), result_path=None, dotsize=350, fig=fig,
                               ax=ax)
            ax.set_title(f'{bpart} -  "all_trials"\n'
                         f'{"W+" if context == 1 else "W-"}')
            fig.tight_layout()

            save_fig(fig, saving_path, f'{name}_{bpart}_{"W-" if context == 0 else "W+"}', formats=saving_formats)


def figure2supp_dplick_barplots(data_table, saving_path, name, saving_formats):
    df_sel = data_table.loc[data_table.trial_type == 'whisker_trial']

    cols = ['index', 'opto_grid_ml', 'opto_grid_ap', 'data_mean_sub', 'context', 'trial_type', 'mouse_id']
    df_sel = df_sel[cols].reset_index(drop=True)

    df_sel['coord'] = '(' + df_sel['opto_grid_ap'].astype(str) + ', ' + df_sel['opto_grid_ml'].astype(str) + ')'

    selected_spots = ['(-1.5, 3.5)', '(-1.5, 4.5)', '(1.5, 1.5)', '(-1.5, 0.5)', '(2.5, 2.5)', '(0.5, 4.5)']
    data_to_plot = df_sel.loc[df_sel.coord.isin(selected_spots)]
    data_to_plot = data_to_plot.drop(['opto_grid_ml', 'opto_grid_ap'], axis=1).reset_index(drop=True)

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    sns.barplot(data_to_plot, x='coord', y='data_mean_sub', hue='context',
                hue_order=['non-rewarded', 'rewarded'], palette=['darkmagenta', 'green'], legend=False, ax=ax)
    sns.stripplot(data_to_plot, x='coord', y='data_mean_sub', hue='context', dodge=True,
                  hue_order=['non-rewarded', 'rewarded'], palette=['darkmagenta', 'green'], legend=False, ax=ax)
    sns.despine()
    ax.set_xlabel('Area')
    ax.set_ylabel('Delta P-Lick')
    fig.tight_layout()
    save_fig(fig, saving_path, f'{name}', formats=saving_formats)
