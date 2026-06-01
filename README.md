# **Bech & Dard Figure panels reproduction**
This repository allows the reproduction of figure panels in [Bech & Dard, eLife 2026](https://elifesciences.org/reviewed-preprints/109717), starting from intermediate dataset.

Intermediate dataset can be downloaded from [Zenodo](https://zenodo.org/communities/petersen-lab-data) or generated starting from NWB files using [process NWB repo](https://github.com/LSENS-BMI-EPFL/Bech_Dard_process_NWB).

## How to use

**1. Install the conda environment**

```bash
cd path/to/Bech_Dard_plot_figures
conda env create -f bech_dard_plot_environment.yml
conda activate bech_dard_plot
```

**2. Clone the repo or download source code, then download the processed data folder from [Zenodo](https://zenodo.org/communities/petersen-lab-data)**

Place them so the folder structure looks like this:

```
Bech_Dard_plot_figures/   ← your folder name (user choice, adjust cd below accordingly)
├── codes/                ← source code (cloned / downloaded from this repo)
│   ├── figure_files/
│   └── utils/
├── data/                 ← downloaded from Zenodo / generated with Bech_Dard_process_NWB (keep the name 'data')
│   ├── figure1/
│   ├── figure1_supp/
│   ├── figure2/
│   ├── figure2_supp/
│   ├── figure3/
│   ├── figure3_supp/
│   ├── figure4/
│   ├── figure4_supp/
│   └── utils/            ← downloaded from Zenodo only
└── figures/              ← created automatically when running the figure files
```

> **Important:** figure file scripts will fail on different architecture / naming.

## Run

Run the figure files sequentially from the repository root — this produces a `figures/` folder containing the figure panels:

```bash
cd path/to/Bech_Dard_plot_figures
python -m codes.figure_files.figure1
python -m codes.figure_files.figure1_supp1
python -m codes.figure_files.figure1_supp2
python -m codes.figure_files.figure1_supp3
python -m codes.figure_files.figure2
python -m codes.figure_files.figure2_supp1
python -m codes.figure_files.figure2_supp2
python -m codes.figure_files.figure2_supp3
python -m codes.figure_files.figure3
python -m codes.figure_files.figure3_supp1
python -m codes.figure_files.figure3_supp2
python -m codes.figure_files.figure3_supp3
python -m codes.figure_files.figure3_supp4
python -m codes.figure_files.figure3_supp5
python -m codes.figure_files.figure4
```
