import os
import warnings
warnings.filterwarnings("ignore")
import pathlib
import platform
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

from codes.utils import figure1_supp3

# Get main data and saving dir
main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
fig_folder = os.path.join(main_dir, 'figures', 'supplementary', 'figure1_supp3')
if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)
load_dir = os.path.join(main_dir, 'data', 'figure1', '1KL')

# A
figure1_supp3.plot_figure1_supp3a(load_dir=load_dir, saving_path=fig_folder, name='Figure1_supp3A', saving_formats=['png', 'svg'])

# B
figure1_supp3.plot_figure1_supp3b(load_dir=load_dir, saving_path=fig_folder, name='Figure1_supp3B', saving_formats=['png', 'svg'])

# C
figure1_supp3.plot_figure1_supp3c(load_dir=load_dir, saving_path=fig_folder, name='Figure1_supp3C', saving_formats=['png', 'svg'])

# D
figure1_supp3.plot_figure1_supp3d(load_dir=load_dir, saving_path=fig_folder, name='Figure1_supp3D', saving_formats=['png', 'svg'])
