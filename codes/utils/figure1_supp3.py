from pathlib import Path
import sys
import types
import yaml
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

from codes.utils.figure1KL import BehavioralAnalysisGBM

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'


def plot_figure1_supp3a(load_dir: Path, saving_path: Path, name: str = 'Figure1_supp3A', saving_formats: list = ['png', 'svg']):
    load_dir = Path(load_dir)
    saving_path = Path(saving_path)
    saving_path.mkdir(exist_ok=True, parents=True)

    analyzer, results = BehavioralAnalysisGBM.load_results(load_dir)

    for tree_idx in range(10):
        save_fig = saving_path / f'{name}_tree_{tree_idx}'
        for ext in saving_formats:
            if ext.lower() == 'svg':
                g = xgb.to_graphviz(analyzer.model, tree_idx=tree_idx)
                svg_content = g.pipe(format='svg', encoding='utf-8')
                with open(save_fig.with_suffix(f'.{ext}'), 'w', encoding='utf-8') as f:
                    f.write(svg_content)
            else:
                xgb.plot_tree(analyzer.model, tree_idx=tree_idx)
                fig = plt.gcf()
                fig.savefig(save_fig.with_suffix(f'.{ext}'), dpi=300)
                plt.close(fig)


def _plot_correlation_matrix(corr_matrix: pd.DataFrame, save_path: Path, saving_formats: list):
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                fmt='.2f', square=True, ax=ax)
    fig.tight_layout()
    for ext in saving_formats:
        fig.savefig(save_path.with_suffix(f'.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def _plot_clustered_correlation(corr_matrix: pd.DataFrame, save_path: Path, saving_formats: list):
    distance_matrix = 1 - np.abs(corr_matrix)
    condensed_dist = squareform(distance_matrix)
    linkage_matrix = linkage(condensed_dist, method='ward')
    cluster_assignments = fcluster(linkage_matrix, 3, criterion='maxclust')
    cluster_order = np.argsort(cluster_assignments)
    corr_clustered = corr_matrix.iloc[cluster_order, cluster_order]

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_clustered, dtype=bool))
    sns.heatmap(corr_clustered, mask=mask, annot=False, cmap='RdBu_r', center=0,
                fmt='.2f', square=True, ax=ax)
    fig.tight_layout()
    for ext in saving_formats:
        fig.savefig(save_path.with_suffix(f'.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def _prepare_features_for_correlation(X: pd.DataFrame) -> pd.DataFrame:
    """Encode boolean and categorical columns for correlation computation."""
    X_prep = X.copy()
    bool_cols = X_prep.select_dtypes(include=['bool']).columns
    X_prep[bool_cols] = X_prep[bool_cols].astype(int)
    for col in X_prep.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X_prep[col] = le.fit_transform(X_prep[col])
    return X_prep


def plot_figure1_supp3b(
    load_dir: Path,
    saving_path: Path,
    name: str = 'Figure1_supp3B',
    saving_formats: list = ['png', 'svg'],
):
    load_dir = Path(load_dir)
    saving_path = Path(saving_path)
    saving_path.mkdir(exist_ok=True, parents=True)

    analyzer, _ = BehavioralAnalysisGBM.load_results(load_dir)

    X = pd.concat([analyzer.X_train, analyzer.X_test])
    y = pd.concat([analyzer.y_train, analyzer.y_test])

    X_prep = _prepare_features_for_correlation(X)
    X_prep['Lick flag'] = y.values

    corr_matrix = X_prep.corr()

    _plot_clustered_correlation(corr_matrix, saving_path / name, saving_formats)


def plot_figure1_supp3c(load_dir: Path, saving_path: Path, name: str = 'Figure1_supp3C', saving_formats: list = ['png', 'svg']):
    load_dir = Path(load_dir)
    saving_path = Path(saving_path)
    saving_path.mkdir(exist_ok=True, parents=True)

    analyzer, results = BehavioralAnalysisGBM.load_results(load_dir)

    X = analyzer.X_train
    # np.random.seed(42)
    # trials = np.random.randint(0, X.shape[0], 10)
    trials = [7270]
    for trial in trials:
        shap.waterfall_plot(
            analyzer.explainer(X.iloc[[trial]])[0],
            show=False,
        )
        fig = plt.gcf()
        fig.tight_layout()
        for ext in saving_formats:
            fig.savefig(saving_path / f'{name}_{trial}.{ext}', dpi=300, bbox_inches='tight')
        plt.close(fig)


def plot_figure1_supp3d(load_dir: Path, saving_path: Path, name: str = 'Figure1_supp3D', saving_formats: list = ['png', 'svg']):
    load_dir = Path(load_dir)
    saving_path = Path(saving_path)
    saving_path.mkdir(exist_ok=True, parents=True)

    with open(load_dir / 'config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    analyzer, results = BehavioralAnalysisGBM.load_results(load_dir)

    df_test = pd.read_csv(load_dir / 'df_test.csv')
    df_test = df_test[config['whisker_features']]
    df_test = df_test.rename(columns=dict(zip(df_test.columns, config['feature_labels'])))

    df_use = df_test.copy()
    cat_mapping = results['cat_mappings']
    for col in cat_mapping:
        if col in df_use.columns:
            df_use[col] = df_use[col].map(cat_mapping[col])

    X = df_use.drop(columns=[config['outcome_column']])
    y = df_use[config['outcome_column']]

    y_pred = analyzer.model.predict(X)
    bal_acc = balanced_accuracy_score(y, y_pred)
    print(f'Balanced accuracy: {bal_acc}')

    X['y_true'] = y
    X['y_pred'] = y_pred
    X['context_change'] = (X['Context'] != X['Context'].shift()).cumsum()
    X['trial_in_block'] = X.groupby('context_change').cumcount()
    X['block_index'] = X['context_change'] + (X['trial_in_block'] // 8)

    block_means = X.groupby('block_index').agg({
        'y_true': 'mean',
        'y_pred': 'mean',
        'Context': 'first',
    }).reset_index()
    block_means['y_true'] = block_means['y_true'] * 100
    block_means['y_pred'] = block_means['y_pred'] * 100
    block_means['trial'] = block_means['block_index'] * 20 - 10

    custom_palette = {0: 'darkmagenta', 1: 'green'}
    colors_lines = {'true': 'orange', 'pred': 'saddlebrown'}
    line_kwargs = {'marker': 'o', 'linewidth': 2, 'alpha': 1}

    fig, ax = plt.subplots(figsize=(9, 4), dpi=300)
    for cond in ['true', 'pred']:
        sns.lineplot(x='trial', y=f'y_{cond}', data=block_means,
                     color=colors_lines[cond], markeredgecolor=colors_lines[cond],
                     ax=ax, label=cond, **line_kwargs)

    ax.set_xlabel('Trial')
    ax.set_ylabel('Lick probability (%)')

    prev_context = None
    start_idx = 0
    for i, context in enumerate(block_means['Context']):
        if prev_context is None:
            prev_context = context
            start_idx = i
        elif context != prev_context:
            start_x = block_means['trial'].iloc[start_idx] - 10
            end_x = block_means['trial'].iloc[i - 1] + (block_means['trial'].iloc[1] - block_means['trial'].iloc[0]) - 10
            ax.axvspan(start_x, end_x, color=custom_palette[prev_context], alpha=0.5, zorder=1)
            prev_context = context
            start_idx = i
    start_x = block_means['trial'].iloc[start_idx] - 10
    end_x = block_means['trial'].iloc[-1] + (block_means['trial'].iloc[1] - block_means['trial'].iloc[0]) - 10
    ax.axvspan(start_x, end_x, color=custom_palette[prev_context], alpha=0.5, zorder=1)

    ax.legend()
    sns.despine(fig=fig, trim=True, offset=0)
    fig.tight_layout()

    acc_str = str(np.round(bal_acc, decimals=3)).replace('.', 'p')
    for ext in saving_formats:
        fig.savefig(saving_path / f'{name}.{ext}', dpi=300)
    plt.close(fig)
