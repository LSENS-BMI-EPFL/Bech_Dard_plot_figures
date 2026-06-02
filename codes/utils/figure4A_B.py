import os
import sys
sys.path.append(os.getcwd())
import yaml
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from itertools import combinations
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import ttest_1samp, ttest_rel
from tqdm import tqdm

from codes.utils.misc.plot_on_grid import get_wf_scalebar, reduce_im_dimensions, plot_grid_on_allen, generate_reduced_image_df
from codes.utils.misc.plot_on_allen import plot_wf_single_frame


def preprocess_corr_results(file):

    df = pd.read_parquet(file.replace("\\", "/"), use_threads=False)

    df['block_id'] = np.abs(np.diff(df.context.values, prepend=0)).cumsum()
    df['trial_count'] = np.empty(len(df), dtype=int)
    df.loc[df.trial_type == 'whisker_trial', 'trial_count'] = df.loc[df.trial_type == 'whisker_trial'].groupby(
        'block_id').cumcount()
    df.loc[df.trial_type == 'auditory_trial', 'trial_count'] = df.loc[
        df.trial_type == 'auditory_trial'].groupby(
        'block_id').cumcount()
    df.loc[df.trial_type == 'no_stim_trial', 'trial_count'] = df.loc[df.trial_type == 'no_stim_trial'].groupby(
        'block_id').cumcount()

    df = df.melt(id_vars=['mouse_id', 'session_id', 'context', 'context_background', 'block_id', 'correct_trial'],
            value_vars=['(-1.5, 0.5)_r', '(-1.5, 0.5)_shuffle_mean', '(-1.5, 0.5)_shuffle_std', '(-1.5, 0.5)_percentile', '(-1.5, 0.5)_nsigmas', 
                        '(-1.5, 3.5)_r', '(-1.5, 3.5)_shuffle_mean', '(-1.5, 3.5)_shuffle_std', '(-1.5, 3.5)_percentile', '(-1.5, 3.5)_nsigmas',
                        '(-1.5, 4.5)_r', '(-1.5, 4.5)_shuffle_mean', '(-1.5, 4.5)_shuffle_std', '(-1.5, 4.5)_percentile', '(-1.5, 4.5)_nsigmas', 
                        '(1.5, 3.5)_r', '(1.5, 3.5)_shuffle_mean', '(1.5, 3.5)_shuffle_std', '(1.5, 3.5)_percentile', '(1.5, 3.5)_nsigmas', 
                        '(0.5, 4.5)_r', '(0.5, 4.5)_shuffle_mean', '(0.5, 4.5)_shuffle_std', '(0.5, 4.5)_percentile', '(0.5, 4.5)_nsigmas', 
                        '(1.5, 1.5)_r', '(1.5, 1.5)_shuffle_mean', '(1.5, 1.5)_shuffle_std', '(1.5, 1.5)_percentile', '(1.5, 1.5)_nsigmas',
                        '(2.5, 2.5)_r', '(2.5, 2.5)_shuffle_mean', '(2.5, 2.5)_shuffle_std', '(2.5, 2.5)_percentile', '(2.5, 2.5)_nsigmas'])

    avg_df = df.groupby(by=['mouse_id', 'session_id', 'context', 'correct_trial', 'variable'])[
        'value'].apply(lambda x: np.array(x.tolist()).mean(axis=0)).reset_index()

    return avg_df


def Figure4_supp1_A(df, roi, main_save_path, save_path, vmin=-0.1, vmax=0.1):
    total_avg = df.groupby(by=['context', 'variable'])['value'].apply(lambda x: np.array(x.tolist()).mean(axis=0)).reset_index()

    seismic_palette = sns.diverging_palette(265, 10, s=100, l=40, sep=30, n=200, center="light", as_cmap=True)

    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    fig.suptitle(f"R - shuffle")

    im_r = total_avg.loc[(total_avg.context == 1) & (total_avg.variable == f'{roi}_r'), 'value'].values[0] - \
            total_avg.loc[(total_avg.context == 1) & (total_avg.variable == f'{roi}_shuffle_mean'), 'value'].values[0]
    im_nor = total_avg.loc[(total_avg.context == 0) & (total_avg.variable == f'{roi}_r'), 'value'].values[0] - \
            total_avg.loc[(total_avg.context == 0) & (total_avg.variable == f'{roi}_shuffle_mean'), 'value'].values[0]

    plot_wf_single_frame(im_r, title='Rewarded', figure=fig, ax_to_plot=ax[0], suptitle='', saving_path='',
                         save_formats=[],
                         colormap='viridis',
                         vmin=0.3, vmax=0.6)
    plot_wf_single_frame(im_nor, title='Non-Rewarded', figure=fig, ax_to_plot=ax[1], suptitle='', saving_path='',
                         save_formats=[],
                         colormap='viridis',
                         vmin=0.3, vmax=0.6)
    plot_wf_single_frame(im_r - im_nor, title='R+ - R-', figure=fig, ax_to_plot=ax[2], suptitle='', saving_path='',
                         save_formats=[],
                         colormap='viridis',
                         vmin=vmin, vmax=vmax)
    fig.savefig(os.path.join(save_path, f"{roi}_shuffle_avg.png"))

    d_palette = sns.color_palette("gnuplot2", 50)
    dprime_palette = LinearSegmentedColormap.from_list("Custom", d_palette[:-2])

    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    fig.suptitle(f"{roi} block average")

    im = total_avg.loc[(total_avg.context == 1) & (total_avg.variable == f"{roi}_nsigmas"), 'value'].values[0] - \
            total_avg.loc[(total_avg.context == 0) & (total_avg.variable == f"{roi}_nsigmas"), 'value'].values[0]

    plot_wf_single_frame(total_avg.loc[(total_avg.context == 1) & (total_avg.variable == f"{roi}_nsigmas"), 'value'].values[0],
                         title='Rewarded', figure=fig, ax_to_plot=ax[0], suptitle='', saving_path='',
                         save_formats=[],
                         colormap=dprime_palette, vmin=1.8, vmax=3)
    plot_wf_single_frame(total_avg.loc[(total_avg.context == 0) & (total_avg.variable == f"{roi}_nsigmas"), 'value'].values[0],
                         title='Non-Rewarded', figure=fig, ax_to_plot=ax[1], suptitle='', saving_path='',
                         save_formats=[],
                         colormap=dprime_palette, vmin=1.8, vmax=3)

    plot_wf_single_frame(im, title='R+ - R-', figure=fig, ax_to_plot=ax[2], suptitle='', saving_path='',
                         save_formats=[],
                         colormap=seismic_palette,
                         vmin=-0.3, vmax=0.3)
    fig.savefig(os.path.join(save_path, f"{roi}_shuffle_nsigmas.png"))

    mask_r = np.where(total_avg.loc[(total_avg.context == 1) & (total_avg.variable == f"{roi}_nsigmas"), 'value'].values[0]>=1.8, im_r, np.nan)
    mask_non_r = np.where(total_avg.loc[(total_avg.context == 0) & (total_avg.variable == f"{roi}_nsigmas"), 'value'].values[0]>=1.8, im_nor, np.nan)
    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    fig.suptitle(f"{roi} block average")

    plot_wf_single_frame(mask_r,
                         title='Rewarded',
                         colormap='viridis', vmin=0.3, vmax=0.6, figure=fig, ax_to_plot=ax[0],
                         suptitle='', saving_path='',
                         save_formats=[])
    plot_wf_single_frame(mask_non_r,
                         title='Non-Rewarded',
                         colormap='viridis', vmin=0.3, vmax=0.6, suptitle='', saving_path='',
                         save_formats=[], figure=fig, ax_to_plot=ax[1])

    plot_wf_single_frame(mask_r-mask_non_r, title='R+ - R-',
                         colormap=seismic_palette, vmin=-0.1, vmax=0.1, suptitle='', saving_path='',
                         save_formats=[], figure=fig, ax_to_plot=ax[2])

    fig.savefig(os.path.join(save_path, f"{roi}_significant_pairs.png"))
    fig.savefig(os.path.join(main_save_path, f"{roi}_significant_pairs.png"))

    plt.close('all')


def Figure4_supp1_B_C(data, output_path):
        
    redim_df = []
    for i, row in data.iterrows():
        redim, coords = reduce_im_dimensions(row['value'][np.newaxis])
        df = generate_reduced_image_df(redim, coords)
        df['context'] = row.context
        df['mouse_id'] = row.mouse_id
        df['correct_trial'] = row.correct_trial
        df['variable'] = row.variable
        redim_df+=[df]
        
    redim_df = pd.concat(redim_df).rename(columns={'dff0': 'value'}).reset_index(drop=True)

    redim_df['seed'] = redim_df.apply(lambda x: x.variable.split("_")[0], axis=1)
    redim_df['coord'] = redim_df.apply(lambda x: f"({x.y}, {x.x})", axis=1)
    drop = []
    for i, row in redim_df.reset_index(names='old_index').iterrows():
        if row.seed == row.coord:
            drop += [i]
        elif row.seed == "(-0.5, 0.5)" or row.coord == "(-0.5, 0.5)":
            drop += [i]
    redim_df = redim_df.drop(drop)

    r_df = redim_df.groupby(by=['mouse_id', 'context', 'correct_trial', 'coord', 'seed', 'y', 'x']).apply(
        lambda x: np.nan_to_num(x.loc[x.variable.str.contains("_r"), 'value'].values[0]) - np.nan_to_num(x.loc[x.variable.str.contains("_shuffle_mean"), 'value'].values[0])).reset_index().rename(columns={0:'r'})

    std_df = redim_df.groupby(by=['mouse_id', 'context', 'correct_trial', 'coord', 'seed', 'y', 'x']).apply(
        lambda x: np.nan_to_num(x.loc[x.variable.str.contains("sigma"), 'value'].values[0])).reset_index().rename(columns={0:'std'})

    r_df = r_df.merge(std_df, on=['mouse_id', 'context', 'correct_trial', 'coord', 'seed', 'y', 'x'])

    redim_df = redim_df.groupby(by=['mouse_id', 'context', 'correct_trial', 'coord', 'seed']).apply(lambda x: np.nan_to_num(x.loc[x.variable.str.contains("_r"), 'value'].values[0]) - np.nan_to_num(x.loc[x.variable.str.contains("_shuffle_mean"), 'value'].values[0])).reset_index().rename(columns={0:'r'})

    os.path.join(output_path, 'figure4A_B')
    choice_all_rois_stats, choice_all_rois_stats_correct_vs_incorrect, choice_selected_rois_stats, choice_selected_rois_stats_correct_vs_incorrect = compute_stats_barplot_choice(redim_df, os.path.join(output_path, 'figure4A_B'))
    all_rois_stats, all_rois_stats_correct_vs_incorrect, selected_rois_stats, selected_rois_stats_correct_vs_incorrect = compute_stats_barplot_context(redim_df, os.path.join(output_path, 'figure4A_B'))
    

    save_path = os.path.join(output_path, 'figure4_supp1')
    agg_df = []

    for outcome, subgroup in redim_df.groupby(by=['context', 'correct_trial']):

        iterator = combinations(subgroup.seed.unique(), 2)
        
        for seed, dest in iterator:
            if seed == dest:
                continue
                
            sub_r = subgroup[(subgroup.seed.isin([seed, dest])) & (subgroup.coord.isin([seed, dest]))]
            sub_r = sub_r[sub_r.seed != sub_r.coord]
            sub_r = sub_r.groupby(by=['mouse_id', 'correct_trial', 'context']).agg({'r': 'mean'}).reset_index()
            sub_r['seed'] = seed
            sub_r['coord'] = dest
            agg_df += [sub_r]

            sub_r = subgroup[(subgroup.seed.isin([seed, dest])) & (subgroup.coord.isin([seed, dest]))]
            sub_r = sub_r[sub_r.seed != sub_r.coord]
            sub_r = sub_r.groupby(by=['mouse_id', 'correct_trial', 'context']).agg({'r': 'mean'}).reset_index()
            sub_r['seed'] = dest
            sub_r['coord'] = seed
            agg_df += [sub_r]

    agg_df = pd.concat(agg_df)
    agg_df = agg_df.groupby(by=['mouse_id', 'correct_trial', 'coord', 'seed']).apply(lambda x: x.loc[x.context==1, 'r'].values[0] - x.loc[x.context==0, 'r'].values[0]).reset_index().rename(columns={0:'r'})

    ## Plot corrected correlation r substracted R+ - R-  with correct trials
    g = sns.catplot(
        x="coord",
        y="r",
        palette=['#032b22'],
        col="seed",
        order=["(-1.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(1.5, 3.5)", "(0.5, 4.5)"],
        data=agg_df.loc[(agg_df.correct_trial==1) & (agg_df.coord.isin(["(-1.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(1.5, 3.5)", "(0.5, 4.5)"]))],
        kind="bar",
        errorbar = ('ci', 95),
        edgecolor="black",
        errcolor="black",
        errwidth=1.5,
        capsize = 0.1,
        height=4,
        aspect=0.8,
        alpha=0.5)
    
    g.map(sns.stripplot, 'coord', 'r', 'correct_trial', hue_order=[1], order=["(-1.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(1.5, 3.5)", "(0.5, 4.5)"], palette=['#032b22', '#da4e02'], dodge=True, alpha=0.6, edgecolor='k', linewidth=1)
    g.set_ylabels('R- <-- r-shuffle --> R+')
    g.tick_params(axis='x', rotation=30)
    for ax in g.axes.flat:
        ax.set_ylim([-0.15, 0.15])
        seed = ax.get_title('center').split("= ")[-1]
        stats = selected_rois_stats.loc[(selected_rois_stats.seed==seed) & (selected_rois_stats.correct_trial==1)]
        ax.scatter(stats.loc[stats.significant, 'coord'].to_list(), stats.loc[stats.significant, 'significant'].map({True:1}).to_numpy()*0.1, marker='*', s=100, c='k')
        for label in ax.get_xticklabels():
            label.set_horizontalalignment('right')
    g.figure.tight_layout()                
    g.figure.savefig(os.path.join(save_path, 'Figure4_supp1_C_correct.png'))
    g.figure.savefig(os.path.join(save_path, 'Figure4_supp1_C_correct.svg'))

    ## Plot corrected correlation r substracted R+ - R-  with incorrect trials

    g = sns.catplot(
        x="coord",
        y="r",
        palette=['#da4e02'],
        col="seed",
        order=["(-1.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(1.5, 3.5)", "(0.5, 4.5)"],
        data=agg_df.loc[(agg_df.correct_trial==0) & (agg_df.coord.isin(["(-1.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(1.5, 3.5)", "(0.5, 4.5)"]))],
        kind="bar",
        errorbar = ('ci', 95),
        edgecolor="black",
        errcolor="black",
        errwidth=1.5,
        capsize = 0.1,
        height=4,
        aspect=0.8,
        alpha=0.5)
    
    g.map(sns.stripplot, 'coord', 'r', 'correct_trial', hue_order=[0], order=["(-1.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(1.5, 3.5)", "(0.5, 4.5)"], palette=['#da4e02'], dodge=True, alpha=0.6, edgecolor='k', linewidth=1)
    g.set_ylabels('R- <-- r-shuffle --> R+')
    g.tick_params(axis='x', rotation=30)
    for ax in g.axes.flat:
        ax.set_ylim([-0.15, 0.15])
        seed = ax.get_title('center').split("= ")[-1]
        stats = selected_rois_stats.loc[(selected_rois_stats.seed==seed) & (selected_rois_stats.correct_trial==0)]
        ax.scatter(stats.loc[stats.significant, 'coord'].to_list(), stats.loc[stats.significant, 'significant'].map({True:1}).to_numpy()*0.1, marker='*', s=100, c='k')
        for label in ax.get_xticklabels():
            label.set_horizontalalignment('right')
    g.figure.tight_layout()
    g.figure.savefig(os.path.join(save_path, 'Figure4_supp1_C_incorrect.png'))
    g.figure.savefig(os.path.join(save_path, 'Figure4_supp1_C_incorrect.svg'))

    ## compute         
    total_avg = data.groupby(by=['context', 'correct_trial', 'variable'])['value'].apply(
        lambda x: np.nanmean(np.array(x.tolist()),axis=0)).reset_index()
    total_df = []
    group = total_avg.reset_index(drop=True)
    group['seed'] = group.apply(lambda x: x.variable.split("_")[0], axis=1)

    group['masked_data'] = group.groupby(by=['context', 'correct_trial', 'seed']).apply(
        lambda x: x.apply(
            lambda y: np.where(x.loc[x.variable.str.contains('sigmas'), 'value'].values[0]>=1.8, y.value, np.nan), axis=1)).reset_index()[0]

    for i, row in group.iterrows():
        if row.seed == '(-0.5, 0.5)':
            continue
        redim, coords = reduce_im_dimensions(row['masked_data'][np.newaxis])
        df = generate_reduced_image_df(redim, coords)
        df['context'] = row.context
        df['seed'] = row.seed
        df['correct_trial'] = row.correct_trial
        df['variable'] = row.variable
        total_df+=[df]
        
    total_df = pd.concat(total_df).rename(columns={'dff0': 'value'})
    total_df['coord'] = total_df.apply(lambda x: f"({x.y}, {x.x})", axis=1)
    total_df = total_df[total_df.coord.isin(total_df.seed.unique())]

    sigma_df = total_df.groupby(by=[
        'context', 'correct_trial', 'coord', 'seed']).apply(
            lambda x: np.nan_to_num(x.loc[x.variable.str.contains("sigma"), 'value'].values[0])).reset_index().rename(columns={0:'sigma'})

    for outcome, subgroup in r_df.groupby('correct_trial'):

        r_df_total_avg = sigma_df.loc[sigma_df.correct_trial==outcome].groupby(by=['context', 'coord', 'seed']).mean().reset_index().rename(columns={'sigma': 'std'})
        r_df_total_avg = r_df_total_avg.loc[r_df_total_avg.seed!=r_df_total_avg.coord]
        agg_df = []
        for c in subgroup.context.unique():
            iterator = combinations(subgroup.seed.unique(), 2)
            
            for seed, dest in iterator:
                if seed == dest:
                    continue
                    
                sub_r = subgroup[(subgroup.context==c) & (subgroup.seed.isin([seed, dest])) & (subgroup.coord.isin([seed, dest]))]
                sub_r = sub_r[sub_r.seed != sub_r.coord]
                sub_r = sub_r.groupby(by=['mouse_id', 'context', 'correct_trial']).agg({'r': 'mean'}).reset_index()
                sub_r['seed'] = seed
                sub_r['coord'] = dest
                sub_r['threshold'] = 0.8 if r_df_total_avg.loc[(r_df_total_avg.context==c) & (r_df_total_avg.seed.isin([seed, dest])) & (r_df_total_avg.coord.isin([seed, dest]))] ['std'].mean()>=1.8 else np.nan
                agg_df += [sub_r]

                sub_r = subgroup[(subgroup.context==c) & (subgroup.seed.isin([seed, dest])) & (subgroup.coord.isin([seed, dest]))]
                sub_r = sub_r[sub_r.seed != sub_r.coord]
                sub_r = sub_r.groupby(by=['mouse_id', 'context', 'correct_trial']).agg({'r': 'mean'}).reset_index()
                sub_r['seed'] = dest
                sub_r['coord'] = seed
                sub_r['threshold'] = 0.8 if r_df_total_avg.loc[(r_df_total_avg.context==c) & (r_df_total_avg.seed.isin([seed, dest])) & (r_df_total_avg.coord.isin([seed, dest]))] ['std'].mean()>=1.8 else np.nan
                agg_df += [sub_r]

        agg_df = pd.concat(agg_df)

        g = sns.catplot(
            x="r",
            y="coord",
            order=["(-1.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(1.5, 3.5)", "(0.5, 4.5)"],
            hue='context',
            hue_order=[0,1],
            palette = ['purple', 'green'],
            col="seed",
            col_order=["(-1.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(1.5, 3.5)", "(0.5, 4.5)"],
            data=agg_df.loc[agg_df.coord.isin(["(-1.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(1.5, 3.5)", "(0.5, 4.5)"])],
            kind="bar",
            errorbar = ('ci', 95),
            edgecolor="black",
            errcolor="black",
            errwidth=1.5,
            capsize = 0.1,
            height=4,
            aspect=0.3,
            alpha=0.5)

        g.map(sns.stripplot, 'r', 'coord', 'context', order=["(-1.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(1.5, 3.5)", "(0.5, 4.5)"], 
                hue_order=[0, 1], palette=['purple', 'green'], dodge=True, alpha=0.6, edgecolor='k', linewidth=1)

        coord_order = ["(-1.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(1.5, 3.5)", "(0.5, 4.5)"]
        hue_order = [0, 1]

        for ax, seed_coord in zip(g.axes.flat, coord_order):
            # Filter data for this subplot's seed
            seed_data = agg_df[agg_df["seed"] == seed_coord]

            for i, coord in enumerate(coord_order):
                for j, context in enumerate(hue_order):
                    subset = seed_data[(seed_data["coord"] == coord) & (seed_data["context"] == context)]

                    # Check if this pair is significant (adjust 'significant' to your column name)
                    if subset.empty or not subset["threshold"].any():
                        continue

                    # Get the mean r value to position the star just beyond the bar tip
                    mean_r = 0.7
                    offset = 0.05 * (1 if mean_r >= 0 else -1)

                    # j offsets the two hue bars: -0.2 and +0.2 for two contexts
                    hue_offset = -0.2 + j * 0.4
                    y_pos = i + hue_offset

                    ax.text(
                        mean_r + offset,  # x: just past the bar tip
                        y_pos,            # y: aligned with the bar
                        "*",
                        ha="center",
                        va="center",
                        fontsize=12,
                        color="black",
                        fontweight="bold"
                    )

        g.figure.savefig(os.path.join(save_path, f'Figure4_supp1_B_{"correct" if outcome==1 else "incorrect"}.png'))
        g.figure.savefig(os.path.join(save_path, f'Figure4_supp1_B_{"correct" if outcome==1 else "incorrect"}.svg'))



def Figure4_B(total_avg, output_path):
    seismic_palette = sns.diverging_palette(265, 10, s=100, l=40, sep=30, n=200, center="light", as_cmap=True)
    viridis_palette = cm.get_cmap('viridis')
    color_dict = {
        "(-1.5, 0.5)": '#6495ED', 
        '(-1.5, 3.5)': '#FF8C00', 
        '(-1.5, 4.5)': '#FFA500', 
        '(1.5, 1.5)': '#0000FF', 
        '(2.5, 2.5)': '#FF0000',
        '(1.5, 3.5)': '#F800FF', 
        '(0.5, 4.5)': '#BA55D3', 
        '(-5.0, 5.0)': 'k',
    }

    total_df = []
    group = total_avg.reset_index(drop=True)
    group['seed'] = group.apply(lambda x: x.variable.split("_")[0], axis=1)

    group['masked_data'] = group.groupby(by=['context', 'correct_trial', 'seed']).apply(
        lambda x: x.apply(
            lambda y: np.where(x.loc[x.variable.str.contains('sigmas'), 'value'].values[0]>=1.8, y.value, np.nan), axis=1)).reset_index()[0]

    for i, row in group.iterrows():
        if row.seed == '(-0.5, 0.5)':
            continue
        redim, coords = reduce_im_dimensions(row['masked_data'][np.newaxis])
        df = generate_reduced_image_df(redim, coords)
        df['context'] = row.context
        df['seed'] = row.seed
        df['correct_trial'] = row.correct_trial
        df['variable'] = row.variable
        df['opto_stim_coord'] = '(-5.0, 5.0)'
        df['color'] = color_dict[row.seed]
        total_df+=[df]
            
    total_df = pd.concat(total_df).rename(columns={'dff0': 'value'})
    total_df['coord'] = total_df.apply(lambda x: f"({x.y}, {x.x})", axis=1)
    total_df = total_df[total_df.coord.isin(total_df.seed.unique())]
    total_df['y_dest'] = total_df.apply(lambda x: eval(x.coord)[0], axis=1)
    total_df['x_dest'] = total_df.apply(lambda x: eval(x.coord)[1], axis=1)
    total_df['y_source'] = total_df.apply(lambda x: eval(x.seed)[0], axis=1)
    total_df['x_source'] = total_df.apply(lambda x: eval(x.seed)[1], axis=1)

    r_df = total_df.groupby(by=[
        'context', 'opto_stim_coord', 'color', 'correct_trial', 'coord', 'seed', 'y_dest', 'x_dest', 'y_source', 'x_source']).apply(
            lambda x: np.nan_to_num(x.loc[x.variable.str.contains("_r"), 'value'].values[0]) - np.nan_to_num(x.loc[x.variable.str.contains("_shuffle_mean"), 'value'].values[0])).reset_index().rename(columns={0:'r'})
    sigma_df = total_df.groupby(by=[
        'context', 'opto_stim_coord', 'color', 'correct_trial', 'coord', 'seed', 'y_dest', 'x_dest', 'y_source', 'x_source']).apply(
            lambda x: np.nan_to_num(x.loc[x.variable.str.contains("sigma"), 'value'].values[0])).reset_index().rename(columns={0:'sigma'})
    delta_r_df = r_df.groupby(by=['opto_stim_coord', 'color', 'correct_trial', 'coord', 'seed', 'y_dest', 'x_dest', 'y_source', 'x_source']).apply(
        lambda x: x.loc[x.context==1, 'r'].values[0] - x.loc[x.context==0, 'r'].values[0]).reset_index().rename(columns={0:'r'})

    for coord in total_df['opto_stim_coord'].unique():
        stats = pd.read_csv(os.path.join(output_path, [f"{coord}_stim" if 'opto' in output_path else ''][0], 'context_pairwise_selected_rois_stats.csv'))
        stats['norm_d'] = np.round(np.clip((stats.d_prime.values - 0.8)/(2 - 0.8), 0, 1), 2)

        for outcome in total_df.correct_trial.unique():
            r = r_df[(r_df.correct_trial==outcome) & (r_df.opto_stim_coord==coord)]
            r = r[r.seed != coord]
            r['norm_r'] = np.round(np.clip((r.r.values- 0.3)/(0.6 - 0.3), 0, 1), 2)

            sigma = sigma_df[(sigma_df.correct_trial==outcome) & (sigma_df.opto_stim_coord==coord)]
            sigma = sigma[sigma.seed != coord]

            delta = delta_r_df[(delta_r_df.correct_trial==outcome) & (delta_r_df.opto_stim_coord==coord)]
            delta = delta[delta.seed != coord]
            delta['norm_r'] = np.round(np.clip((delta.r.values- -0.05)/(0.05 - -0.05), 0, 1), 2)

            save_path = os.path.join(output_path, "Figure4_B")

            for c in total_df.context.unique():
                fig, ax = plt.subplots(figsize=(4,4))
                fig.suptitle(f"{coord} stim r between rois")
                im=ax.scatter(r.loc[r.coord==r.seed, 'x_source'], r.loc[r.coord==r.seed, 'y_source'], s=200, c='black')
                if coord != '(-5.0, 5.0)':
                    ax.scatter(eval(coord)[1], eval(coord)[0], c='gray', s=200)

                ax.scatter(0, 0, marker='+', c='gray', s=100)

                iterator = combinations(r.seed.unique(), 2)

                for seed, dest in iterator:
                    if seed == dest:
                        continue
                    
                    sub_r = r[(r.context==c) & (r.seed.isin([seed, dest])) & (r.coord.isin([seed, dest]))]
                    sub_r = sub_r[sub_r.seed != sub_r.coord]
                    sub_sigma = sigma[(sigma.context==c) & (sigma.seed.isin([seed, dest])) & (sigma.coord.isin([seed, dest]))]
                    sub_sigma = sub_sigma[sub_sigma.seed != sub_sigma.coord]

                    if sub_sigma.sigma.mean()>=1.8:
                        ax.plot([sub_r.x_source.unique(), sub_r.x_dest.unique()], 
                                [sub_r.y_source.unique(), sub_r.y_dest.unique()], 
                                c=viridis_palette(sub_r.norm_r.mean()), linewidth=8)       

                ax.grid(True)
                ax.set_xticks(np.linspace(0.5,5.5,6))
                ax.set_xlim([-0.25, 6])
                ax.set_yticks(np.linspace(-3.5, 2.5,7))
                ax.set_ylim([-3.75, 2.75])
                ax.invert_xaxis()

                sm = plt.cm.ScalarMappable(cmap=viridis_palette, norm=plt.Normalize(vmin=0, vmax=1))
                fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
                fig.savefig(os.path.join(output_path, f'Figure4_B_{["rewarded" if c else "non-rewarded"][0]}_{"correct" if outcome==1 else "incorrect"}.png'))
                fig.savefig(os.path.join(output_path, f'Figure4_B_{["rewarded" if c else "non-rewarded"][0]}_{"correct" if outcome==1 else "incorrect"}.svg'))

            fig, ax = plt.subplots(figsize=(4,4))
            fig.suptitle(f"{coord} stim r between rois")
            im=ax.scatter(delta.loc[delta.coord==delta.seed, 'x_source'], delta.loc[delta.coord==delta.seed, 'y_source'], s=200, c='black')
            if coord != '(-5.0, 5.0)':
                ax.scatter(eval(coord)[1], eval(coord)[0], c='gray', s=200)

            ax.scatter(0, 0, marker='+', c='gray', s=100)

            iterator = zip(["(-1.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(2.5, 2.5)", "(1.5, 3.5)"], 
                            ["(1.5, 1.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(1.5, 3.5)", "(1.5, 3.5)", "(0.5, 4.5)", "(0.5, 4.5)"])                
            for seed, dest in iterator:
                if seed == dest:
                    continue
                
                sub_delta = delta[(delta.seed.isin([seed, dest])) & (delta.coord.isin([seed, dest]))]
                sub_delta = sub_delta[sub_delta.seed != sub_delta.coord]
                sub_sigma = sigma[(sigma.seed.isin([seed, dest])) & (sigma.coord.isin([seed, dest]))]
                sub_sigma = sub_sigma[sub_sigma.seed != sub_sigma.coord]

                d = stats.loc[(stats.correct_trial==outcome) & (stats.seed==seed) & (stats.coord==dest), 'd_prime'].values[0]
                norm_d = stats.loc[(stats.correct_trial==outcome) & (stats.seed==seed) & (stats.coord==dest), 'norm_d'].values[0]

                if norm_d>0:      
                    ax.plot([sub_delta.x_source.unique(), sub_delta.x_dest.unique()], [sub_delta.y_source.unique(), sub_delta.y_dest.unique()], 
                            c=seismic_palette(sub_delta.norm_r.mean()), linewidth=d*2)  
                else:
                    ax.plot([sub_delta.x_source.unique(), sub_delta.x_dest.unique()], [sub_delta.y_source.unique(), sub_delta.y_dest.unique()], 
                            c='gray', linewidth=1)  
                        
            ax.grid(True)
            ax.set_xticks(np.linspace(0.5,5.5,6))
            ax.set_xlim([-0.25, 6])
            ax.set_yticks(np.linspace(-3.5, 2.5,7))
            ax.set_ylim([-3.75, 2.75])
            ax.invert_xaxis()
            
            sm = plt.cm.ScalarMappable(cmap=seismic_palette, norm=plt.Normalize(vmin=0, vmax=1))
            fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            fig.savefig(os.path.join(output_path, f'Figure4_B_delta_context_{outcome}.png'))
            fig.savefig(os.path.join(output_path, f'Figure4_B_delta_context_{outcome}.svg'))


    r_df = total_df.groupby(by=['context',  'color', 'correct_trial', 'coord', 'seed', 'y_dest', 'x_dest', 'y_source', 'x_source']).apply(lambda x: np.nan_to_num(x.loc[x.variable.str.contains("_r"), 'value'].values[0]) - np.nan_to_num(x.loc[x.variable.str.contains("_shuffle_mean"), 'value'].values[0])).reset_index().rename(columns={0:'r'})
    sigma_df = total_df.groupby(by=['context', 'color', 'correct_trial', 'coord', 'seed', 'y_dest', 'x_dest', 'y_source', 'x_source']).apply(lambda x: np.nan_to_num(x.loc[x.variable.str.contains("sigma"), 'value'].values[0])).reset_index().rename(columns={0:'sigma'})
    delta_r_df = r_df.groupby(by=[ 'context', 'color', 'coord', 'seed', 'y_dest', 'x_dest', 'y_source', 'x_source']).apply(lambda x: x.loc[x.correct_trial==1, 'r'].values[0] - x.loc[x.correct_trial==0, 'r'].values[0]).reset_index().rename(columns={0:'r'})


    stats = pd.read_csv(os.path.join(output_path, 'choice_pairwise_selected_rois_stats.csv'))
    stats['norm_d'] = np.round(np.clip((stats.d_prime.values - 0.8)/(2 - 0.8), 0, 1), 2)

    for c in total_df.context.unique():
        r = r_df[(r_df.context==c)]
        r = r[r.seed != coord]
        r['norm_r'] = np.round(np.clip((r.r.values- 0.3)/(0.6 - 0.3), 0, 1), 2)

        sigma = sigma_df[(sigma_df.context==c)]
        sigma = sigma[sigma.seed != coord]

        delta = delta_r_df[(delta_r_df.context==c)]
        delta = delta[delta.seed != coord]
        delta['norm_r'] = np.round(np.clip((delta.r.values- -0.05)/(0.05 - -0.05), 0, 1), 2)

        fig, ax = plt.subplots(figsize=(4,4))
        fig.suptitle(f"{coord} stim r between rois")
        im=ax.scatter(delta.loc[delta.coord==delta.seed, 'x_source'], delta.loc[delta.coord==delta.seed, 'y_source'], s=200, c='black')
        if coord != '(-5.0, 5.0)':
            ax.scatter(eval(coord)[1], eval(coord)[0], c='gray', s=100)

        ax.scatter(0, 0, marker='+', c='gray', s=100)
        iterator = zip(["(-1.5, 0.5)", "(-1.5, 3.5)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(2.5, 2.5)", "(1.5, 3.5)"],
                        ["(1.5, 1.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(1.5, 1.5)", "(2.5, 2.5)", "(1.5, 3.5)", "(1.5, 3.5)", "(0.5, 4.5)", "(0.5, 4.5)"])

        for seed, dest in iterator:
            if seed == dest:
                continue

            sub_delta = delta[(delta.seed.isin([seed, dest])) & (delta.coord.isin([seed, dest]))]
            sub_delta = sub_delta[sub_delta.seed != sub_delta.coord]
            sub_sigma = sigma[(sigma.seed.isin([seed, dest])) & (sigma.coord.isin([seed, dest]))]
            sub_sigma = sub_sigma[sub_sigma.seed != sub_sigma.coord]

            d = stats.loc[(stats.context==c) & (stats.seed==seed) & (stats.coord==dest), 'd_prime'].values[0]
            norm_d = stats.loc[(stats.context==c) & (stats.seed==seed) & (stats.coord==dest), 'norm_d'].values[0]

            if norm_d>0:
                ax.plot([sub_delta.x_source.unique(), sub_delta.x_dest.unique()], [sub_delta.y_source.unique(), sub_delta.y_dest.unique()],
                        c=seismic_palette(sub_delta.norm_r.mean()), linewidth=d*2)
            else:
                ax.plot([sub_delta.x_source.unique(), sub_delta.x_dest.unique()], [sub_delta.y_source.unique(), sub_delta.y_dest.unique()],
                        c='gray', linewidth=1)
        ax.grid(True)
        ax.set_xticks(np.linspace(0.5,5.5,6))
        ax.set_xlim([-0.25, 6])
        ax.set_yticks(np.linspace(-3.5, 2.5,7))
        ax.set_ylim([-3.75, 2.75])
        ax.invert_xaxis()

        sm = plt.cm.ScalarMappable(cmap=seismic_palette, norm=plt.Normalize(vmin=0, vmax=1))
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        fig.savefig(os.path.join(output_path, f'Figure4_B_delta_choice_{c}.png'))
        fig.savefig(os.path.join(output_path, f'Figure4_B_delta_choice_{c}.svg'))



def compute_stats_barplot_context(df, output_path):
    df=df[df.coord!='(2.5, 5.5)']
    all_rois_stats =[]
    for name, group in df.groupby(by=['correct_trial', 'seed', 'coord']):

        dof = group.mouse_id.unique().shape[0]-1,
        t, p = ttest_rel(group.loc[group.context==1, 'r'].to_numpy(), group.loc[group.context==0, 'r'].to_numpy())
        mean_diff = (group.loc[group.context==1, 'r'].mean() - group.loc[group.context==0, 'r'].mean())
        std_diff = np.std(group.loc[group.context==1, 'r'].to_numpy() - group.loc[group.context==0, 'r'].to_numpy())
        d = abs(mean_diff/std_diff)

        results = {
         'correct_trial': name[0],
         'seed': name[1],
         'coord': name[2],
         'dof': dof,
         'mean_rew': group.loc[group.context==1, 'r'].mean(),
         'std_rew': group.loc[group.context==1, 'r'].std(),
         'mean_no_rew': group.loc[group.context==0, 'r'].mean(),
         'std_no_rew': group.loc[group.context==0, 'r'].std(),
         't': t,
         'p': p,
         'p_corr': p*df.coord.unique().shape[0],
         'alpha': 0.05,
         'alpha_corr': 0.05/df.coord.unique().shape[0],
         'significant': p<(0.05/df.coord.unique().shape[0]),
        'd_prime': d
         }
        
        all_rois_stats += [results]
    all_rois_stats = pd.DataFrame(all_rois_stats)
    all_rois_stats.to_csv(os.path.join(output_path, 'context_pairwise_all_rois_stats.csv'))

    all_rois_stats_correct_vs_incorrect =[]
    for name, group in df.groupby(by=['seed', 'coord']):
        context_diff = group.groupby(by=['mouse_id', 'correct_trial']).apply(
            lambda x: float(np.mean(
                x.loc[x.context == 1, 'r'].to_numpy() - x.loc[x.context == 0, 'r'].to_numpy()
            ))
        ).reset_index().rename(columns={0: 'r'})
        t, p = ttest_rel(context_diff.loc[context_diff.correct_trial==1, 'r'], context_diff.loc[context_diff.correct_trial==0, 'r'])
        mean_diff = (context_diff.loc[context_diff.correct_trial==1, 'r'].mean() - context_diff.loc[context_diff.correct_trial==0, 'r'].mean())
        std_diff = np.std(context_diff.loc[context_diff.correct_trial==1, 'r'].to_numpy() - context_diff.loc[context_diff.correct_trial==0, 'r'].to_numpy())

        results = {
        'seed': name[0],
        'coord': name[1],
        'dof': context_diff.mouse_id.unique().shape[0]-1,
        'mean_correct': context_diff.loc[context_diff.correct_trial==1, 'r'].mean(),
        'std_correct': context_diff.loc[context_diff.correct_trial==1, 'r'].std(),
        'mean_incorrect': context_diff.loc[context_diff.correct_trial==0, 'r'].mean(),
        'std_incorrect': context_diff.loc[context_diff.correct_trial==0, 'r'].std(),
        't': t,
        'p': p,
        'p_corr': p*df.coord.unique().shape[0],
        'alpha': 0.05,
        'alpha_corr': 0.05/df.coord.unique().shape[0],
        'significant': p<(0.05/df.coord.unique().shape[0]),
        'd_prime': abs(mean_diff/std_diff)
        }
        all_rois_stats_correct_vs_incorrect += [results]
    all_rois_stats_correct_vs_incorrect = pd.DataFrame(all_rois_stats_correct_vs_incorrect)
    all_rois_stats_correct_vs_incorrect.to_csv(os.path.join(output_path, 'context_pairwise_all_rois_stats_correct_vs_incorrect.csv'))

    # df = df[df.seed!='(1.5, 3.5)']
    selected_rois_stats =[]
    for name, group in df.loc[df.coord.isin(df.seed.unique())].groupby(by=['correct_trial', 'seed', 'coord']):

        dof = group.mouse_id.unique().shape[0]-1,
        t, p = ttest_rel(group.loc[group.context==1, 'r'].to_numpy(), group.loc[group.context==0, 'r'].to_numpy())
        mean_diff = (group.loc[group.context==1, 'r'].mean() - group.loc[group.context==0, 'r'].mean())
        std_diff = np.std(group.loc[group.context==1, 'r'].to_numpy() - group.loc[group.context==0, 'r'].to_numpy())
        d = abs(mean_diff/std_diff)

        results = {
         'correct_trial': name[0],
         'seed': name[1],
         'coord': name[2],
         'dof': dof,
         'mean_rew': group.loc[group.context==1, 'r'].mean(),
         'std_rew': group.loc[group.context==1, 'r'].std(),
         'mean_no_rew': group.loc[group.context==0, 'r'].mean(),
         'std_no_rew': group.loc[group.context==0, 'r'].std(),
         't': t,
         'p': p,
         'p_corr': p*df.loc[df.coord.isin(df.seed.unique()), 'coord'].unique().shape[0],
         'alpha': 0.05,
         'alpha_corr': 0.05/df.loc[df.coord.isin(df.seed.unique()), 'coord'].unique().shape[0],
         'significant': p<(0.05/df.loc[df.coord.isin(df.seed.unique()), 'coord'].unique().shape[0]),
         'd_prime': abs(mean_diff/std_diff)

         }
        
        selected_rois_stats += [results]
    selected_rois_stats = pd.DataFrame(selected_rois_stats)
    selected_rois_stats.to_csv(os.path.join(output_path, 'context_pairwise_selected_rois_stats.csv'))

    selected_rois_stats_correct_vs_incorrect =[]
    for name, group in df.loc[df.coord.isin(df.seed.unique())].groupby(by=['seed', 'coord']):
        context_diff = (
            group.groupby(by=['mouse_id', 'correct_trial'])
            .apply(lambda x: (
                    x.loc[x.context == 1, 'r'].to_numpy() - x.loc[x.context == 0, 'r'].to_numpy()
            ).item(), include_groups=False)
            .reset_index()
            .rename(columns={0: 'r'})
        )
        t, p = ttest_rel(context_diff.loc[context_diff.correct_trial==1, 'r'], context_diff.loc[context_diff.correct_trial==0, 'r'])
        mean_diff = (context_diff.loc[context_diff.correct_trial==1, 'r'].mean() - context_diff.loc[context_diff.correct_trial==0, 'r'].mean())
        std_diff = np.std(context_diff.loc[context_diff.correct_trial==1, 'r'].to_numpy() - context_diff.loc[context_diff.correct_trial==0, 'r'].to_numpy())
        results = {
        'seed': name[0],
        'coord': name[1],
        'dof': context_diff.mouse_id.unique().shape[0]-1,
        'mean_correct': context_diff.loc[context_diff.correct_trial==1, 'r'].mean(),
        'std_correct': context_diff.loc[context_diff.correct_trial==1, 'r'].std(),
        'mean_incorrect': context_diff.loc[context_diff.correct_trial==0, 'r'].mean(),
        'std_incorrect': context_diff.loc[context_diff.correct_trial==0, 'r'].std(),
        't': t,
        'p': p,
        'p_corr': p*df.loc[df.coord.isin(df.seed.unique()), 'coord'].unique().shape[0],
        'alpha': 0.05,
        'alpha_corr': 0.05/df.loc[df.coord.isin(df.seed.unique()), 'coord'].unique().shape[0],
        'significant': p<(0.05/df.loc[df.coord.isin(df.seed.unique()), 'coord'].unique().shape[0]),
        'd_prime': abs(mean_diff/std_diff)
        }
        selected_rois_stats_correct_vs_incorrect += [results]
    selected_rois_stats_correct_vs_incorrect = pd.DataFrame(selected_rois_stats_correct_vs_incorrect)
    selected_rois_stats_correct_vs_incorrect.to_csv(os.path.join(output_path, 'context_pairwise_selected_rois_stats_correct_vs_incorrect.csv'))

    return all_rois_stats, all_rois_stats_correct_vs_incorrect, selected_rois_stats, selected_rois_stats_correct_vs_incorrect


def compute_stats_barplot_choice(mouse_avg, output_path):

    save_path = os.path.join(output_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    df = mouse_avg[~mouse_avg.coord.isin(['(1.5, 4.5)', '(2.5, 4.5)', '(2.5, 5.5)'])]
    all_rois_stats = []
    for name, group in df.groupby(by=['context', 'seed', 'coord']):
        t, p = ttest_rel(group.loc[group.correct_trial == 1, 'r'].to_numpy(),
                         group.loc[group.correct_trial == 0, 'r'].to_numpy())
        mean_diff = (group.loc[group.correct_trial == 1, 'r'].mean() - group.loc[group.correct_trial == 0, 'r'].mean())
        std_diff = np.std(
            group.loc[group.correct_trial == 1, 'r'].to_numpy() - group.loc[group.correct_trial == 0, 'r'].to_numpy())

        results = {
            'context': name[0],
            'seed': name[1],
            'coord': name[2],
            'dof': group.mouse_id.unique().shape[0] - 1,
            'mean_rew': group.loc[group.correct_trial == 1, 'r'].mean(),
            'std_rew': group.loc[group.correct_trial == 1, 'r'].std(),
         'mean_no_rew': group.loc[group.correct_trial==0, 'r'].mean(),
         'std_no_rew': group.loc[group.correct_trial==0, 'r'].std(),
         't': t,
         'p': p,
         'p_corr': p*df.coord.unique().shape[0],
         'alpha': 0.05,
         'alpha_corr': 0.05/df.coord.unique().shape[0],
         'significant': p<(0.05/df.coord.unique().shape[0]),
        'd_prime': abs(mean_diff/std_diff)
         }
        
        all_rois_stats += [results]
    all_rois_stats = pd.DataFrame(all_rois_stats)
    all_rois_stats.to_csv(os.path.join(output_path, 'choice_pairwise_all_rois_stats.csv'))

    all_rois_stats_rew_vs_norew =[]
    for name, group in df.groupby(by=['seed', 'coord']):
        choice_diff = group.groupby(by=['mouse_id', 'context']).apply(
            lambda x: (x.loc[x.correct_trial == 1, 'r'].to_numpy() - x.loc[x.correct_trial == 0, 'r'].to_numpy()).item()
        ).reset_index().rename(columns={0: 'r'})
        t, p = ttest_rel(choice_diff.loc[choice_diff.context==1, 'r'].to_numpy(), choice_diff.loc[choice_diff.context==0, 'r'].to_numpy())
        mean_diff = choice_diff.loc[choice_diff.context==1, 'r'].mean() - choice_diff.loc[choice_diff.context==0, 'r'].mean()
        std_diff = np.std(choice_diff.loc[choice_diff.context==1, 'r'].to_numpy() - choice_diff.loc[choice_diff.context==0, 'r'].to_numpy())

        results = {
        'seed': name[0],
        'coord': name[1],
        'dof': choice_diff.mouse_id.unique().shape[0]-1,
        'mean_correct': choice_diff.loc[choice_diff.context==1, 'r'].mean(),
        'std_correct': choice_diff.loc[choice_diff.context==1, 'r'].std(),
        'mean_incorrect': choice_diff.loc[choice_diff.context==0, 'r'].mean(),
        'std_incorrect': choice_diff.loc[choice_diff.context==0, 'r'].std(),
        't': t,
        'p': p,
        'p_corr': p*df.coord.unique().shape[0],
        'alpha': 0.05,
        'alpha_corr': 0.05/df.coord.unique().shape[0],
        'significant': p<(0.05/df.coord.unique().shape[0]),
        'd_prime': abs(mean_diff/std_diff)
        }
        all_rois_stats_rew_vs_norew += [results]
    all_rois_stats_rew_vs_norew = pd.DataFrame(all_rois_stats_rew_vs_norew)
    all_rois_stats_rew_vs_norew.to_csv(os.path.join(output_path, 'choice_pairwise_all_rois_stats_rew_vs_norew.csv'))

    # df = df[df.seed!='(1.5, 3.5)']
    selected_rois_stats =[]
    for name, group in df.loc[df.coord.isin(df.seed.unique())].groupby(by=['context', 'seed', 'coord']):
        t, p = ttest_rel(group.loc[group.correct_trial==1, 'r'].to_numpy(), group.loc[group.correct_trial==0, 'r'].to_numpy())
        mean_diff = group.loc[group.correct_trial==1, 'r'].mean() - group.loc[group.correct_trial==0, 'r'].mean()
        std_diff = np.std(group.loc[group.correct_trial==1, 'r'].to_numpy() - group.loc[group.correct_trial==0, 'r'].to_numpy())

        results = {
         'context': name[0],
         'seed': name[1],
         'coord': name[2],
         'dof': group.mouse_id.unique().shape[0]-1,
         'mean_rew': group.loc[group.correct_trial==1, 'r'].mean(),
         'std_rew': group.loc[group.correct_trial==1, 'r'].std(),
         'mean_no_rew': group.loc[group.correct_trial==0, 'r'].mean(),
         'std_no_rew': group.loc[group.correct_trial==0, 'r'].std(),
         't': t,
         'p': p,
         'p_corr': p*df.loc[df.coord.isin(df.seed.unique()), 'coord'].unique().shape[0],
         'alpha': 0.05,
         'alpha_corr': 0.05/df.loc[df.coord.isin(df.seed.unique()), 'coord'].unique().shape[0],
         'significant': p<(0.05/df.loc[df.coord.isin(df.seed.unique()), 'coord'].unique().shape[0]),
         'd_prime': abs(mean_diff/std_diff)

         }
        
        selected_rois_stats += [results]
    selected_rois_stats = pd.DataFrame(selected_rois_stats)
    selected_rois_stats.to_csv(os.path.join(output_path, 'choice_pairwise_selected_rois_stats.csv'))

    selected_rois_stats_rew_vs_norew =[]
    for name, group in df.loc[df.coord.isin(df.seed.unique())].groupby(by=['seed', 'coord']):
        choice_diff = (
            group.groupby(by=['mouse_id', 'context'])
            .apply(lambda x: (
                    x.loc[x.correct_trial == 1, 'r'].to_numpy() - x.loc[x.correct_trial == 0, 'r'].to_numpy()
            ).item(), include_groups=False)
            .reset_index()
            .rename(columns={0: 'r'})
        )
        t, p = ttest_rel(choice_diff.loc[choice_diff.context==1, 'r'].to_numpy(), choice_diff.loc[choice_diff.context==0, 'r'].to_numpy())
        mean_diff = choice_diff.loc[choice_diff.context==1, 'r'].mean() - choice_diff.loc[choice_diff.context==0, 'r'].mean()
        std_diff = np.std(choice_diff.loc[choice_diff.context==1, 'r'].to_numpy() - choice_diff.loc[choice_diff.context==0, 'r'].to_numpy())
        results = {
        'seed': name[0],
        'coord': name[1],
        'dof': choice_diff.mouse_id.unique().shape[0]-1,
        'mean_correct': choice_diff.loc[choice_diff.context==1, 'r'].mean(),
        'std_correct': choice_diff.loc[choice_diff.context==1, 'r'].std(),
        'mean_incorrect': choice_diff.loc[choice_diff.context==0, 'r'].mean(),
        'std_incorrect': choice_diff.loc[choice_diff.context==0, 'r'].std(),
        't': t,
        'p': p,
        'p_corr': p*df.loc[df.coord.isin(df.seed.unique()), 'coord'].unique().shape[0],
        'alpha': 0.05,
        'alpha_corr': 0.05/df.loc[df.coord.isin(df.seed.unique()), 'coord'].unique().shape[0],
        'significant': p<(0.05/df.loc[df.coord.isin(df.seed.unique()), 'coord'].unique().shape[0]),
        'd_prime': abs(mean_diff/std_diff)
        }
        selected_rois_stats_rew_vs_norew += [results]
    selected_rois_stats_rew_vs_norew = pd.DataFrame(selected_rois_stats_rew_vs_norew)
    selected_rois_stats_rew_vs_norew.to_csv(os.path.join(output_path, 'choice_pairwise_selected_rois_stats_rew_vs_norew.csv'))

    return all_rois_stats, all_rois_stats_rew_vs_norew, selected_rois_stats, selected_rois_stats_rew_vs_norew


def main(data, output_path):

    if not os.path.exists(os.path.join(output_path, 'figure4A_B')):
        os.makedirs(os.path.join(output_path, 'figure4A_B'))
        os.makedirs(os.path.join(output_path, 'figure4_supp1'))

    data.value = data.apply(lambda x: x.value[0] if 'percentile' not in x.variable else x.value, axis=1)
    mouse_avg = data.groupby(by=['mouse_id',  'context', 'correct_trial', 'variable'])['value'].apply(
        lambda x: np.nanmean(np.array(x.tolist()),axis=0)).reset_index()
    mouse_avg['value'] = mouse_avg['value'].apply(lambda x: np.array(x).reshape(125, -1))

    total_avg = mouse_avg.groupby(by=['context', 'correct_trial', 'variable'])['value'].apply(
        lambda x: np.nanmean(np.array(x.tolist()),axis=0)).reset_index()

    # plot total avg
    for roi in ['(-1.5, 0.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(1.5, 3.5)', '(0.5, 4.5)', '(1.5, 1.5)', '(2.5, 2.5)']:

        print(f"Plotting total averages for roi {roi}")
        save_path = os.path.join(output_path, 'figure4_supp1', 'figure4_supp1_A', roi)
        main_save_path = os.path.join(output_path, 'figure4A_B', 'figure4', roi)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(main_save_path):
            os.makedirs(main_save_path)
        Figure4_supp1_A(total_avg.loc[total_avg.correct_trial==1], roi, main_save_path, save_path)

    Figure4_supp1_B_C(mouse_avg, output_path)
    Figure4_B(total_avg, os.path.join(output_path, 'figure4A_B'))

