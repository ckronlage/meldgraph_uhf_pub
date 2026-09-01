# Evaluation of MELD-graph on paired 7T and 3T MRI data

This is the code for analyses in the manuscript: ____

The code here is meant for documentation. The multi-centre dataset cannot be made publicly available due to privacy restrictions.


### Requirements
- conda
- apptainer/singularity


### Apptainer container images
MELD-graph is run from a fork of the original tool at https://github.com/ckronlage/meld_graph/tree/v2.2.7_uhf. It includes an option to run
the FreeSurfer segmentation at the native ultra-high-field resolution (`--uhf_highres`, used by
`rule meld_graph_segmentation_persubj_uhf`).

Build the docker image from that fork, then convert it:
```
apptainer build meld_graph.sif docker-daemon://meld_graph_fork:latest
```

FreeSurfer 8.1.0 is used for the fsaverage_sym surfaces and labels of the evaluation
(`vol_eval.py`, and the surface plots of Figures 4 and 5) and for the skullstripping/registration
in the slice-plotting notebook:
```
apptainer build freesurfer_8.1.0.sif docker://freesurfer/freesurfer:8.1.0
```

Both licence files are expected in the repository root: the FreeSurfer `license.txt` (bound into
the container as `/license.txt` and exported as `FS_LICENSE`) and `meld_license.txt` (bound as
`/meld_license.txt`, exported as `MELD_LICENSE`).


### Install conda environment
```
conda env create -f environment.yml
```


### Data
- Raw image data is assumed to be saved at `data/raw/<site>/` (BIDS format, ses-3T and ses-7T)
- Manually drawn FCD lesion masks are expected at `data/raw/<site>/bids/derivatives/lesion_masks/`;
  `rule coreg_lesion_masks` coregisters them onto the MELD-graph input images, which is what the
  evaluation scores the predictions against
- Aggregated clinical metadata for all sites is in `data/subjects.csv` (not in individual BIDS
  participants.tsv files, because these are formatted differently); the analysis reads `source`,
  `subject ID`, `group`, `include`, `sex`, `age_years`, `category_FCD` and `category_3T_MR_negative`,
  plus the further `category_*`, surgery, histopathology and outcome columns reported in Table 1
- Data preparation scripts are custom for each site to deal with peculiarities, entry points are
  `prepare_data_meldgraph/prepare_<site>.py`.

The workflow writes, under `data/`:

| folder | contents |
|---|---|
| `meld_graph_scaffold/` | the MELD `meld_params/` and `models/` folder, downloaded once; no subject data |
| `meld_graph/<site>/<B0>/` | inputs, the FreeSurfer segmentations and the ComBat-harmonised predictions |
| `meld_graph_noharmo/<site>/<B0>/` | hardlinked copy of the above plus the unharmonised predictions |
| `results/vol_eval_<timestamp>.csv` | aggregated results, one row per subject, condition and harmonisation |
| `results/vol_eval_clusters_<timestamp>.csv` | one row per predicted cluster of the same run |
| `results/labeled_predictions/` | the predictions relabelled into true and false positives, and the ground-truth labels projected to fsaverage_sym |

The acquisition conditions `<B0>` are the T1w conditions `3T`, `7T`, `7T_CP` and `7T_pTx`, each of
them also with a FLAIR as a second input (`<B0>_FLAIR`) and, for the 7T conditions, once more with
the FreeSurfer segmentation run at 1 mm instead of at the native resolution (`<B0>_nouhf`). Which
of them make up the analysis groups compared in the paper ("3T", "7T default", "7T adapted", and
their FLAIR variants) is set at the top of `evaluation_meldgraph/vol_eval_plots.py`.


### Run MELD-graph
This is automated with Snakemake (https://snakemake.github.io/), which enables parallelisation on different systems, e.g., HPC clusters.
```
conda activate uhf_meld
snakemake -s Snakefile_meldgraph --workflow-profile profiles/default -c 8
# or parallelize with a different workflow profile, e.g. via SLURM on a cluster
```
`meld_graph_segmentation_persubj_*` (FreeSurfer) is by far the most expensive rule; the
unharmonised condition is a hardlinked copy of its output so much faster.

The last rule of the workflow, `vol_eval`, scores every prediction against the coregistered lesion
masks and writes `data/results/vol_eval_<timestamp>.csv` and
`data/results/vol_eval_clusters_<timestamp>.csv`.


### Results and Plots
The notebooks in `evaluation_meldgraph/notebooks/` produce the paper outputs:

| notebook | paper output |
|---|---|
| `01_tables.ipynb` | Table 1 |
| `02_figure1_cohort_overview.ipynb` | Figure 1 |
| `03_figure2_supp_fig2_3_sensitivity_specificity.ipynb` | Figure 2, Supplementary Figures 2 and 3 |
| `04_figure3_plot_slices.ipynb` | Figure 3 |
| `05_figure4_fp_distributions.ipynb` | Figure 4 |
| `06_supp_fig1_cluster_stats.ipynb` | Supplementary Figure 1 |
| `07_supp_fig5_gt_label_features.ipynb` | Supplementary Figure 5 |
| `08_figure5_supp_fig6_feature_maps.ipynb` | Figure 5, Supplementary Figure 6 |

`evaluation_meldgraph/vol_eval_plots.py` holds the helpers shared by more than one notebook,
above all `load_and_prepare_data()`, with which every notebook starts.


## Contact:
If you have any questions, comments or suggestions, get in touch:
<cornelius.kronlage@kcl.ac.uk>
