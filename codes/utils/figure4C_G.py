import itertools
import os
import re
import sys
sys.path.append(os.getcwd())
import glob
import warnings

import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
import yaml
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, rgb2hex, hex2color
from matplotlib.lines import Line2D            
from itertools import product

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics.pairwise import paired_distances, cosine_similarity

from codes.utils.misc.plot_on_grid import get_wf_scalebar, reduce_im_dimensions, plot_grid_on_allen, generate_reduced_image_df
from codes.utils.misc.plot_on_allen import plot_wf_single_frame

from pathlib import Path
from scipy.stats import ttest_1samp, linregress

import random
np.random.seed(982874)
random.seed(982874)

def load_wf_opto_data(data_path):
    total_df = []
    
    data_files = glob.glob(os.path.join(data_path, "*", "results.parquet.gzip")) 
    for file in data_files:
        df = [pd.read_parquet(Path(file, compression='gzip'))]
        total_df += df

    return pd.concat(total_df, ignore_index=True)


def load_opto_data(opto_result_path):
    single_mouse_result_files = glob.glob(os.path.join(opto_result_path, "*", "opto_data.json"))
    mice=[]
    for file in single_mouse_result_files:
        mice += [os.path.basename(os.path.dirname(file))]

    opto_df = []
    for file in single_mouse_result_files:
        d= pd.read_json(file)
        d['mouse_id'] = [os.path.basename(os.path.dirname(file)) for i in range(d.shape[0])]
        opto_df += [d]
    opto_df = pd.concat(opto_df)
    opto_df = opto_df.loc[opto_df.opto_grid_ap!=3.5]

    opto_df = opto_df.reset_index(drop=True)
    opto_df['opto_stim_coord'] = opto_df.apply(lambda x: tuple([x.opto_grid_ap, x.opto_grid_ml]), axis=1)
    return opto_df.loc[opto_df.mouse_id.isin(mice)]

    
def Figure4C(data_path, result_path):

    name_dict={
        "(-5.0, 5.0)": 'Control', 
        "(-1.5, 3.5)": 'wS1', 
        "(1.5, 1.5)": 'wM1',
        "(-1.5, 0.5)": 'RSC'
    }
    avg = pd.read_pickle(data_path)

    for c, group in avg.groupby('context'):
        if c==1:
            loc_list = ["(-5.0, 5.0)", "(-1.5, 3.5)", "(1.5, 1.5)"]
        else:
            loc_list = ["(-1.5, 0.5)"]

        for loc in loc_list:
            print(c, loc)
            im_seq = group.loc[(group.trial_type=='whisker_trial') & (group.opto_stim_coord==loc), 'wf_image_sub'].to_numpy()[0]
            save_path = os.path.join(result_path, 'Figure4C_images', f"{name_dict[loc]}_stim")
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            for i in range(9, 16):
                fig, ax0 = plt.subplots(1, 1)
                frame = im_seq[i]
                if frame.ndim == 1:
                    frame = frame.reshape(125, -1)
                plot_wf_single_frame(frame, f"{(i-10)*20}ms", figure=fig, ax_to_plot=ax0, colormap='hotcold', vmin=-0.03, vmax=0.03, saving_path='', save_formats=[], suptitle='')
                fig.savefig(os.path.join(save_path, f'whisker_stim_frame_{(i-10)*20}.png'))


def Figure4_supp2_A(data_path, result_path): 

    name_dict={
        "(-5.0, 5.0)": 'Control', 
        "(-1.5, 3.5)": 'wS1', 
        "(1.5, 1.5)": 'wM1',
        "(-1.5, 0.5)": 'RSC'
    }

    avg = pd.read_pickle(data_path)

    for loc in ["(-1.5, 3.5)", "(1.5, 1.5)"]:
        print(loc)
        im_seq = avg.loc[(avg.context==1) & (avg.trial_type=='whisker_trial') & (avg.opto_stim_coord==loc), 'wf_image_sub'].to_numpy()[0]
        save_path = os.path.join(result_path, f"{name_dict[loc]}_stim")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for i in range(9, 16):
            fig, ax0 = plt.subplots()
            frame = im_seq[i]
            if frame.ndim == 1:
                frame = frame.reshape(125, -1)
            plot_wf_single_frame(frame, f"{(i-10)*20}ms",  figure=fig, ax_to_plot=ax0, colormap='hotcold', vmin=-0.03, vmax=0.03, saving_path='', save_formats=[], suptitle='')
            fig.savefig(os.path.join(save_path, f'whisker_stim_frame_{(i-10)*20}.png'))


def Figure4_D_Figure4_supp2_D(control_df, pc_df, result_path):

    roi_list = ['(-1.5, 3.5)', '(-1.5, 4.5)', '(1.5, 1.5)', '(-1.5, 0.5)', '(2.5, 2.5)', '(0.5, 4.5)']

    fig, ax = plt.subplots(2,6, figsize=(24,8))
    fig.suptitle('PC1')
    fig1, ax1 = plt.subplots(2,6, figsize=(24,8))
    fig1.suptitle('PC2')
    fig2, ax2 = plt.subplots(2,6, figsize=(24,8))
    fig2.suptitle('PC3')

    for i, stim in enumerate(roi_list):
        ax[0, i].set_title(stim)
        ax1[0, i].set_title(stim)
        ax2[0, i].set_title(stim)

        for j, (name, subgroup) in enumerate(control_df.groupby('context')):
            if name=='rewarded':
                whisker = ['#83f28f', '#348A18']
                idx=0
            else:
                whisker = ['#D9C4EC', '#6E188A']
                idx=1

            trial = 'whisker_trial'
            group = subgroup[subgroup.trial_type == 'whisker_trial']

            sns.lineplot(group, 
                            x='time', 
                            y='PC 1', 
                            hue='legend', 
                            hue_order = ['(-5.0, 5.0) - no lick', '(-5.0, 5.0) - lick'], 
                            palette=whisker, estimator='mean', errorbar=('ci', 95), ax=ax[idx, i])
            sns.lineplot(group, 
                            x='time', 
                            y='PC 2', 
                            hue='legend', 
                            hue_order = ['(-5.0, 5.0) - no lick', '(-5.0, 5.0) - lick'], 
                            palette=whisker, estimator='mean', errorbar=('ci', 95), ax=ax1[idx, i])                    
            sns.lineplot(group, 
                            x='time', 
                            y='PC 3', 
                            hue='legend', 
                            hue_order = ['(-5.0, 5.0) - no lick', '(-5.0, 5.0) - lick'], 
                            palette=whisker, estimator='mean', errorbar=('ci', 95), ax=ax2[idx, i])                    

            group = pc_df.loc[(pc_df.context==name) & (pc_df.trial_type==trial) & (pc_df.opto_stim_coord==stim)]


            sns.lineplot(group, 
                            x='time', 
                            y='PC 1', 
                            color='royalblue', estimator='mean', errorbar=('ci', 95), ax=ax[idx, i])
            sns.lineplot(group, 
                            x='time', 
                            y='PC 2', 
                            color='royalblue', estimator='mean', errorbar=('ci', 95), ax=ax1[idx, i])                    
            sns.lineplot(group, 
                            x='time', 
                            y='PC 3', 
                           color='royalblue', estimator='mean', errorbar=('ci', 95), ax=ax2[idx, i]) 
                           
            ax[idx,i].set_ylim(-35,35)
            ax[idx,i].set_ylabel('PC 1')

            ax1[idx,i].set_ylim(-15,10)
            ax1[idx,i].set_ylabel('PC 2')

            ax2[idx,i].set_ylim(-15,5)
            ax2[idx,i].set_ylabel('PC 3')

    save_path = os.path.join(result_path, 'figure4_supp2')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    fig.tight_layout()
    fig.savefig(os.path.join(save_path, f"Figure4_supp2_D_top.png"))
    fig.savefig(os.path.join(save_path, f"Figure4_supp2_D_top.svg"))
    fig1.tight_layout()
    fig1.savefig(os.path.join(save_path, f"Figure4_supp2_D_mid.png"))
    fig1.savefig(os.path.join(save_path, f"Figure4_supp2_D_mid.svg"))
    fig2.tight_layout()
    fig2.savefig(os.path.join(save_path, f"Figure4_supp2_D_bot.png"))
    fig2.savefig(os.path.join(save_path, f"Figure4_supp2_D_bot.svg"))

    fig2.savefig(os.path.join(result_path, 'figure4C_G', f"Figure4_D.png"))
    fig2.savefig(os.path.join(result_path, 'figure4C_G', f"Figure4_D.svg"))
    plt.close('all')

def compute_angle_stim_lick(control_df, pc_df, result_path):

    control = control_df.loc[(control_df.time>=0) & (control_df.trial_type=='whisker_trial')].drop(
        columns=['mouse_id', 'legend']
    ).groupby(by=['context', 'trial_type', 'opto_stim_coord', 'lick_flag', 'time']).apply('mean').reset_index()
    stim = pc_df.loc[(pc_df.time>=0) & (pc_df.trial_type=='whisker_trial') & (pc_df.opto_stim_coord!='(-5.0, 5.0)')]
    
    for pc in ['PC 3']:
        control[pc] = control[pc] + stim[pc].min()
        stim[pc] = stim[pc] + stim[pc].min()

    result_df = []
    for name, group in stim.groupby(by=['mouse_id', 'context', 'opto_stim_coord']):
        context = group.context.unique()[0]
        lick_sim = np.diag(cosine_similarity(group['PC 3'].reset_index(drop=True).reset_index().to_numpy(), control.loc[(control.context==context) & (control.lick_flag==1), 'PC 3'].reset_index(drop=True).reset_index().to_numpy()))
        nolick_sim = np.diag(cosine_similarity(group['PC 3'].reset_index(drop=True).reset_index().to_numpy(), control.loc[(control.context==context) & (control.lick_flag==0), 'PC 3'].reset_index(drop=True).reset_index().to_numpy()))

        for pc in ['PC 3']:
            v1 = group[pc].to_numpy().flatten()/np.linalg.norm(group[pc].to_numpy().flatten())
            v2 = control.loc[(control.context==context) & (control.lick_flag==1), pc].to_numpy().flatten()/np.linalg.norm(control.loc[(control.context==context) & (control.lick_flag==1), pc].to_numpy().flatten())
            v3 = control.loc[(control.context==context) & (control.lick_flag==0), pc].to_numpy().flatten()/np.linalg.norm(control.loc[(control.context==context) & (control.lick_flag==0), pc].to_numpy().flatten())
            angle_lick = np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))
            angle_nolick = np.degrees(np.arccos(np.clip(np.dot(v1, v3), -1.0, 1.0)))
            angle_control = np.degrees(np.arccos(np.clip(np.dot(v2, v3), -1.0, 1.0)))

            result={
                'mouse_id': name[0],
                'pc': pc,
                'context': context,
                'opto_stim_coord': group.opto_stim_coord.unique()[0],
                'PC3_similarity': (lick_sim - nolick_sim).sum(),
                'angle_lick': angle_lick,
                'angle_nolick': angle_nolick,
                'angle_diff': angle_lick - angle_nolick,
                'angle_control': angle_control
            }
            result_df += [result]
    result_df = pd.DataFrame(result_df)
    save_path = os.path.join(result_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    result_df.to_csv(os.path.join(save_path, 'results_angle.csv'))
    return result_df


def Figure4_E(angle_df, result_path):
    save_path = os.path.join(result_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    roi_list = ['(-1.5, 0.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(0.5, 4.5)', '(1.5, 1.5)', '(2.5, 2.5)']
    group = angle_df[(angle_df.opto_stim_coord.isin(roi_list)) & (angle_df.pc=="PC 3")]
    pc = "PC 3"

    fig, ax = plt.subplots(figsize=(4,3))
    sns.barplot(
        data=group,
        x='opto_stim_coord',
        y='angle_lick',
        order=['(-1.5, 3.5)', '(-1.5, 4.5)', '(1.5, 1.5)', '(-1.5, 0.5)', '(2.5, 2.5)', '(0.5, 4.5)'],
        hue='context',
        hue_order=['non-rewarded', 'rewarded'],
        palette=['purple', 'green'],
        alpha=0.5,
        estimator='mean',
        errorbar=('ci', 95),
        err_kws=dict(color='black', lw=1),
        ax=ax,
    )
    sns.stripplot(
        data=group,
        x='opto_stim_coord',
        y='angle_lick',
        order=['(-1.5, 3.5)', '(-1.5, 4.5)', '(1.5, 1.5)', '(-1.5, 0.5)', '(2.5, 2.5)', '(0.5, 4.5)'],
        hue='context',
        hue_order=['non-rewarded', 'rewarded'],
        palette=['purple', 'green'],
        s= 5,  
        jitter=True, 
        dodge=True,
        ax=ax,
    )
    sns.despine()
    ax.set_xticks(['(-1.5, 3.5)', '(-1.5, 4.5)', '(1.5, 1.5)', '(-1.5, 0.5)', '(2.5, 2.5)', '(0.5, 4.5)'], ['wS1', 'wS2', 'wM', 'RSC', 'ALM', 'tjS1'])
    ax.legend_.remove()
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, f"Figure4_E.png"))
    fig.savefig(os.path.join(save_path, f"Figure4_E.svg"))
    plt.close('all')


def Figure4_F_G_correlations(opto_df, angle_df, result_path):
    mouse_color = ["#003049", "#d62828", "#f77f00", "#fcbf49"]
    roi_color = {
        '(-1.5, 3.5)': '#ff8c00', 
        '(-1.5, 4.5)': "#ffa500", 
        '(1.5, 1.5)': "#0000ff", 
        '(-1.5, 0.5)': '#6495ed', 
        '(2.5, 2.5)': "#ff0000", 
        '(0.5, 4.5)': "#ba55d3"
    }
    save_path = os.path.join(result_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)    
    opto_df = opto_df[opto_df.trial_type=='whisker_trial']
    opto_df = opto_df[opto_df.mouse_id.isin(angle_df.mouse_id.unique())]

    angle_df['colors'] = angle_df.apply(lambda x: roi_color[x.opto_stim_coord] if x.opto_stim_coord in roi_color.keys() else "#808080", axis=1)
    
    avg_angle_df = angle_df.drop(columns=['mouse_id']).groupby(by=['pc', 'context', 'opto_stim_coord', 'colors'], as_index=False, sort=False).agg('mean')
    avg_opto_df = opto_df.loc[opto_df.trial_type=='whisker_trial'].drop(
        columns=['mouse_id', 'index', 'opto_grid_ml', 'opto_grid_ap', 'shuffle_mean', 'shuffle_std', 'context_background',
       'shuffle_mean_sub', 'shuffle_std_sub', 'shuffle_dist', 'shuffle_dist_sub', 'data_mean', 'percentile',
       'percentile_sub', 'n_sigma', 'n_sigma_sub', 'p', 'p_corr', 'p_sub', 'p_corr_sub', 'trial_type',]).groupby(
        by=['context', 'opto_stim_coord'], as_index=False, sort=False).agg('mean')

    for pc in ['PC 3']:
        fig, ax = plt.subplots(1,2, figsize=(8,4))
        fig.suptitle(f"Average {pc} Plick correlation\n n=4 mice")
        for i, c in enumerate(avg_angle_df.context.unique()):
            ax[i].set_title(c)

            colorlist = avg_angle_df.sort_values(['context', 'opto_stim_coord']).loc[(avg_angle_df.pc==pc) & (angle_df.context==c), "colors"]
            angle = avg_angle_df.sort_values(['context', 'opto_stim_coord']).loc[(avg_angle_df.pc==pc) & (angle_df.context==c), "angle_lick"]
            opto = avg_opto_df.sort_values(['context', 'opto_stim_coord']).loc[avg_opto_df.context==c, "data_mean_sub"]
            z = np.polyfit(angle, opto, 1)  
            y_hat = np.poly1d(z)(angle)
            model = linregress(angle, opto)
            text = f"$R^2 = {model.rvalue ** 2:0.3f}$\n$p={model.pvalue:0.3e}$"

            ax[i].scatter(angle, opto, color=colorlist)
            ax[i].plot(angle, y_hat, c='k', ls='-', lw=2)
            ax[i].text(0.05, 0.95, text, transform=ax[i].transAxes, fontsize=10, verticalalignment='top')

            ax[i].set_ylim([-0.5, 0.3])
            ax[i].set_xlim([0, 15])
            ax[i].spines[['right', 'top']].set_visible(False)
        fig.savefig(os.path.join(save_path, f'Figure4_F_G_right.png'))
        fig.savefig(os.path.join(save_path, f'Figure4_F_G_right.svg'))
        plt.close('all')


def Figure4_F_G_map(avg_angle_df, result_path):
    save_path = os.path.join(result_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for name, group in avg_angle_df.groupby('context'):
        for pc in ['PC 3']:
            fig, ax = plt.subplots(figsize=(4,4))
            im_df = generate_reduced_image_df(group.loc[group.pc==pc, 'angle_lick'].to_numpy()[np.newaxis, :], [eval(coord) for coord in group.loc[group.pc==pc, 'opto_stim_coord']])
            im_df = im_df.rename(columns={'dff0': 'angle'})
            plot_grid_on_allen(im_df, outcome='angle', palette='viridis', facecolor=None, edgecolor=None, dotsize=350, result_path=None, vmin=0, vmax=10, fig=fig, ax=ax)
            ax.set_axis_off()
            fig.savefig(os.path.join(save_path, f'Figure4_{"F" if name=="rewarded" else "G"}_left.png'), dpi=400)
            plt.close('all')


def Figure4_supp2_BC(pca, result_path):
    
    # Explained variance ratio
    exp_var = [val * 100 for val in pca.explained_variance_ratio_]
    plot_y = [sum(exp_var[:i+1]) for i in range(len(exp_var))]
    plot_x = range(1, len(plot_y) + 1)
    fig,ax = plt.subplots(figsize=(7,4))
    ax.plot(plot_x, plot_y, marker="o", color="#9B1D20")
    for x, y in zip(plot_x, plot_y):
        plt.text(x, y + 3, f"{y:.1f}%", ha="center", va="bottom")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Cumulative percentage of variance explained")
    ax.set_ylim([50,100])
    ax.set_xticks(plot_x)
    ax.grid(axis="y")
    ax.spines[['top', 'right']].set_visible(False)
    for ext in ['.png', '.svg']:
        fig.savefig(Path(result_path, f'Figure4_supp2_B{ext}'))

    coeff = np.transpose(pca.components_)

    ## Plot loadings in allen ccf
    labels=['(-0.5, 0.5)', '(-0.5, 1.5)', '(-0.5, 2.5)', '(-0.5, 3.5)',
       '(-0.5, 4.5)', '(-0.5, 5.5)', '(-1.5, 0.5)', '(-1.5, 1.5)',
       '(-1.5, 2.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(-1.5, 5.5)',
       '(-2.5, 0.5)', '(-2.5, 1.5)', '(-2.5, 2.5)', '(-2.5, 3.5)',
       '(-2.5, 4.5)', '(-2.5, 5.5)', '(-3.5, 0.5)', '(-3.5, 1.5)',
       '(-3.5, 2.5)', '(-3.5, 3.5)', '(-3.5, 4.5)', '(-3.5, 5.5)',
       '(0.5, 0.5)', '(0.5, 1.5)', '(0.5, 2.5)', '(0.5, 3.5)', '(0.5, 4.5)',
       '(0.5, 5.5)', '(1.5, 0.5)', '(1.5, 1.5)', '(1.5, 2.5)', '(1.5, 3.5)',
       '(1.5, 4.5)', '(2.5, 0.5)', '(2.5, 1.5)', '(2.5, 2.5)', '(2.5, 3.5)']
    fig, ax = plt.subplots(1, 3, figsize=(7, 12))
    fig.suptitle("PC1-3 loadings")

    for i in range(3):
        im_pca = generate_reduced_image_df(coeff[np.newaxis, :, i], [eval(label) for label in labels])
        im_pca.drop(im_pca[(im_pca.x==5.5)&(im_pca.y==2.5)].index, inplace=True)        
        plot_grid_on_allen(im_pca, outcome='dff0', palette='seismic', facecolor=None, edgecolor=None, result_path=None, dotsize=220, vmin=-im_pca.dff0.abs().max(), vmax=im_pca.dff0.abs().max(), norm=None, fig=fig, ax= ax.flat[i])
        ax.flat[i].set_axis_off()
        ax.flat[i].set_title(f"PC {i+1}")
    fig.savefig(os.path.join(result_path, f"Figure4_supp2_C.png"))
    plt.close('all')


def enforce_sign_consistency(pca, reference_feature_idx=9):
    """
    Checks PCA coefficient signs and aligns them to the paper, since they can flip arbitrarily between runs due to the nature of PCA.
    Args:
        pca: The fitted PCA object containing the components.
        reference_feature_idx: The index of the reference feature (default is 9, corresponding to '(-1.5, 3.5)').
    """
    for i in range(pca.n_components):
        if i==0 and pca.components_[i, reference_feature_idx] < 0:
            pca.components_[i, :] *= -1
        elif i!=0 and pca.components_[i, reference_feature_idx] > 0:
            pca.components_[i, :] *= -1
        else:
            continue

    return pca
    

def Figure4_DG_supp2_BD(data_path, opto_data_path, output_path):

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    coords_list = {'wS1': "(-1.5, 3.5)", 'wS2': "(-1.5, 4.5)", 'wM1': "(1.5, 1.5)", 'wM2': "(2.5, 1.5)", 'RSC': "(-0.5, 0.5)", "RSC_2": "(-1.5, 0.5)",
            'ALM': "(2.5, 2.5)", 'tjS1':"(0.5, 4.5)", 'tjM1':"(1.5, 3.5)", 'control': "(-5.0, 5.0)"}

    opto_df = load_opto_data(opto_data_path)
    opto_df = opto_df[~opto_df.opto_stim_coord.astype(str).isin(["(1.5, 5.5)", "(2.5, 4.5)", "(2.5, 5.5)"])]
    
    total_df = load_wf_opto_data(data_path)
    total_df = total_df[~total_df.opto_stim_coord.isin(["(1.5, 5.5)", "(2.5, 4.5)", "(2.5, 5.5)"])]
    total_df.context = total_df.context.map({0:'non-rewarded', 1:'rewarded'})
    total_df['time'] = [[np.linspace(-1,3.98,250)] for i in range(total_df.shape[0])]
    total_df['legend']= total_df.apply(lambda x: f"{x.opto_stim_coord} - {'lick' if x.lick_flag==1 else 'no lick'}",axis=1)

    d = {c: lambda x: x.unique()[0] for c in ['opto_stim_loc', 'legend']}
    d['time'] = lambda x: list(x)[0][0]
    for c in ['(-0.5, 0.5)', '(-0.5, 1.5)', '(-0.5, 2.5)', '(-0.5, 3.5)', '(-0.5, 4.5)', '(-0.5, 5.5)', '(-1.5, 0.5)',
       '(-1.5, 1.5)', '(-1.5, 2.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(-1.5, 5.5)', '(-2.5, 0.5)', '(-2.5, 1.5)', '(-2.5, 2.5)',
       '(-2.5, 3.5)', '(-2.5, 4.5)', '(-2.5, 5.5)', '(-3.5, 0.5)', '(-3.5, 1.5)', '(-3.5, 2.5)', '(-3.5, 3.5)', '(-3.5, 4.5)',
       '(-3.5, 5.5)', '(0.5, 0.5)', '(0.5, 1.5)', '(0.5, 2.5)', '(0.5, 3.5)', '(0.5, 4.5)', '(0.5, 5.5)', '(1.5, 0.5)', 
       '(1.5, 1.5)', '(1.5, 2.5)', '(1.5, 3.5)', '(1.5, 4.5)', '(2.5, 0.5)', '(2.5, 1.5)', '(2.5, 2.5)',
        '(2.5, 3.5)']:
        d[f"{c}"]= lambda x: np.nanmean(np.stack(x), axis=0)
          
    mouse_df = total_df.groupby(by=['mouse_id', 'context', 'trial_type', 'opto_stim_coord', 'lick_flag']).agg(d).reset_index()
    mouse_df = mouse_df.melt(id_vars=['mouse_id', 'context', 'trial_type', 'legend', 'opto_stim_coord', 'lick_flag', 'time'],
                                 value_vars=['(-0.5, 0.5)', '(-0.5, 1.5)', '(-0.5, 2.5)', '(-0.5, 3.5)', '(-0.5, 4.5)', '(-0.5, 5.5)', '(-1.5, 0.5)',
       '(-1.5, 1.5)', '(-1.5, 2.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(-1.5, 5.5)', '(-2.5, 0.5)', '(-2.5, 1.5)', '(-2.5, 2.5)',
       '(-2.5, 3.5)', '(-2.5, 4.5)', '(-2.5, 5.5)', '(-3.5, 0.5)', '(-3.5, 1.5)', '(-3.5, 2.5)', '(-3.5, 3.5)', '(-3.5, 4.5)',
       '(-3.5, 5.5)', '(0.5, 0.5)', '(0.5, 1.5)', '(0.5, 2.5)', '(0.5, 3.5)', '(0.5, 4.5)', '(0.5, 5.5)', '(1.5, 0.5)', 
       '(1.5, 1.5)', '(1.5, 2.5)', '(1.5, 3.5)', '(1.5, 4.5)', '(2.5, 0.5)', '(2.5, 1.5)', '(2.5, 2.5)', '(2.5, 3.5)'],
                                 var_name='roi',
                                 value_name='dff0').explode(['time', 'dff0'])
    mouse_df = mouse_df[(mouse_df.time>=-0.15)&(mouse_df.time<=0.15)]
    
    # Use control stim location, whisker and catch trials, with lick and no-lick separate to compute the pc space
    avg_df = mouse_df.drop(columns=['mouse_id']).groupby(by=['context', 'trial_type', 'lick_flag', 'legend', 'opto_stim_coord', 'roi', 'time']).agg(lambda x: np.nanmean(x)).reset_index()
    avg_df.time = avg_df.time.round(2)
    subset_df = avg_df[(avg_df.trial_type.isin(['whisker_trial', 'no_stim_trial'])) & (avg_df.opto_stim_coord=="(-5.0, 5.0)")].pivot(index=['context','trial_type', 'legend', 'time'], columns='roi', values='dff0')
    avg_data_for_pca = subset_df.to_numpy()

    # Standardize average data for training: Based on trials with light on control location 
    scaler = StandardScaler()
    fit_scaler = scaler.fit(avg_data_for_pca)
    avg_data_for_pca = fit_scaler.transform(avg_data_for_pca)
    pca = PCA(n_components=15)
    results = pca.fit(np.nan_to_num(avg_data_for_pca))

    # Enforce coefficient sign
    pca = enforce_sign_consistency(pca, reference_feature_idx=9) ## 9 corresponds to the index in the roi list for wS1

    # Plot coefficients and variance explained
    Figure4_supp2_BC(pca, os.path.join(output_path, 'figure4_supp2'))

    subset_df = mouse_df[
        (mouse_df.trial_type.isin(['whisker_trial', 'no_stim_trial'])) & 
        (mouse_df.opto_stim_coord=="(-5.0, 5.0)")].pivot(
        index=['mouse_id', 'context', 'trial_type', 'lick_flag', 'legend', 'opto_stim_coord', 'time'], columns='roi', values='dff0')
    avg_data_for_pca = subset_df.to_numpy()
    avg_data_for_pca = fit_scaler.transform(avg_data_for_pca)
    principal_components = pca.transform(np.nan_to_num(avg_data_for_pca))

    control_df = pd.DataFrame(data=principal_components, index=subset_df.index)
    control_df.columns = [f"PC {i+1}" for i in range(0, principal_components.shape[1])]
    control_df = control_df.reset_index()

    # Project whisker and catch trials 
    mouse_df = total_df.groupby(by=['mouse_id', 'context', 'trial_type', 'opto_stim_coord']).agg(d).reset_index()
    mouse_df = mouse_df.melt(id_vars=['mouse_id', 'context', 'trial_type', 'opto_stim_coord', 'time'],
                                value_vars=['(-0.5, 0.5)', '(-0.5, 1.5)', '(-0.5, 2.5)', '(-0.5, 3.5)', '(-0.5, 4.5)', '(-0.5, 5.5)', '(-1.5, 0.5)',
    '(-1.5, 1.5)', '(-1.5, 2.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(-1.5, 5.5)', '(-2.5, 0.5)', '(-2.5, 1.5)', '(-2.5, 2.5)',
    '(-2.5, 3.5)', '(-2.5, 4.5)', '(-2.5, 5.5)', '(-3.5, 0.5)', '(-3.5, 1.5)', '(-3.5, 2.5)', '(-3.5, 3.5)', '(-3.5, 4.5)',
    '(-3.5, 5.5)', '(0.5, 0.5)', '(0.5, 1.5)', '(0.5, 2.5)', '(0.5, 3.5)', '(0.5, 4.5)', '(0.5, 5.5)', '(1.5, 0.5)', 
    '(1.5, 1.5)', '(1.5, 2.5)', '(1.5, 3.5)', '(1.5, 4.5)', '(2.5, 0.5)', '(2.5, 1.5)', '(2.5, 2.5)', '(2.5, 3.5)'],
                                var_name='roi',
                                value_name='dff0').explode(['time', 'dff0'])

    mouse_df = mouse_df.reset_index()
    mouse_df = mouse_df[(mouse_df.time>=-0.15)&(mouse_df.time<=0.15)]
    subset_df = mouse_df[mouse_df.trial_type.isin(['whisker_trial', 'no_stim_trial'])].pivot(
        index=['mouse_id', 'context','trial_type', 'opto_stim_coord', 'time'], columns='roi', values='dff0')

    avg_data_for_pca = subset_df.to_numpy()
    avg_data_for_pca = fit_scaler.transform(avg_data_for_pca)
    principal_components = pca.transform(np.nan_to_num(avg_data_for_pca))

    pc_df = pd.DataFrame(data=principal_components, index=subset_df.index)
    pc_df.columns = [f"PC {i+1}" for i in range(0, principal_components.shape[1])]
    subset_df = subset_df.join(pc_df).reset_index()
    pc_df = pc_df.reset_index()

    Figure4_D_Figure4_supp2_D(control_df, pc_df[pc_df.opto_stim_coord!="(-5.0, 5.0)"], output_path)

    angle_df = compute_angle_stim_lick(control_df, pc_df[pc_df.opto_stim_coord!="(-5.0, 5.0)"], output_path)
    Figure4_E(angle_df, os.path.join(output_path, 'figure4C_G'))

    avg_angle_df = angle_df.drop(columns='mouse_id').groupby(by=['pc', 'context', 'opto_stim_coord'], as_index=False, sort=False).agg('mean')
    Figure4_F_G_map(avg_angle_df, os.path.join(output_path, 'figure4C_G'))
    Figure4_F_G_correlations(opto_df, angle_df, os.path.join(output_path, 'figure4C_G'))


def main(data_path_4C, data_path_4DG, data_path_4_supp, opto_data_path, output_path):

    if not os.path.exists(os.path.join(output_path, 'figure4C_G')):
        os.makedirs(os.path.join(output_path, 'figure4C_G'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'figure4_supp2'), exist_ok=True)

    Figure4C(data_path_4C, os.path.join(output_path, 'figure4C_G'))
    Figure4_supp2_A(data_path_4_supp, os.path.join(output_path, 'figure4_supp2', '2A'))

    Figure4_DG_supp2_BD(data_path=data_path_4DG, opto_data_path=opto_data_path, output_path=output_path)

