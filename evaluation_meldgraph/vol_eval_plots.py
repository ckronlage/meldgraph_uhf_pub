"""Helpers shared by more than one of the notebooks in evaluation_meldgraph/notebooks/.

Everything that is used by a single notebook only stays in that notebook, next to its
call site. The exceptions are:

  load_and_prepare_data       -> every notebook
  select_fcd_control_groups   -> 01 (which reports the subject counts before this selection)
  filter_to_common_subjects   -> 03 (the performance table) and 05 (the false-positive maps)
  latest_vol_eval_path        -> the loader, and 06 for the per-cluster file of the same run
  the analysis group names    -> every notebook
  the colour scheme
   (harmo_labels, main_group_hue_family, harmo_saturation, _with_saturation, _with_value)
                              -> 03 (the sensitivity/specificity bars) and, through the
                                 six-condition colours below, 06, 07 and 08
  plot_on_surface             -> 05 (Figure 4) and 08 (Figure 5, Supp. Figure 6)
  the six conditions and their colours, and surf_feature_names
                              -> 06, 07 and 08
  friedman/kruskal/describe_across_conditions
                              -> 06 (cluster metrics) and 07 (features in the lesion label)
  load_surf_feature_maps      -> 07 and 08

The notebooks add this folder to sys.path and import from here by module name, as the
AID-HS notebooks do with evaluation_hs/hs_plots.py.
"""

import colorsys
import glob
import os
import pickle
import subprocess
import tempfile

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import pyvista as pv
from scipy.stats import friedmanchisquare, kruskal
from statsmodels.stats.multitest import multipletests

# the hdf5 feature matrices the meld_graph preprocessing writes, read by the extraction script
from vol_eval import (read_surf_features, copy_fsaverage_sym_labels, NVERT,
                      FSAVERAGE_SYM_CORTEX_LABEL_PATH)


# the B0_conditions making up each analysis group
MAIN_3T_CONDITIONS = ['3T']
MAIN_7T_ADAPTED_CONDITIONS = ['7T', '7T_CP']
MAIN_7T_DEFAULT_CONDITIONS = ['7T_nouhf', '7T_CP_nouhf']
FLAIR_3T_CONDITIONS = ['3T_FLAIR']
FLAIR_7T_ADAPTED_CONDITIONS = ['7T_FLAIR', '7T_CP_FLAIR']

# the analysis group names, used throughout the notebook and as the x-axis groups of the plots
GROUP_3T = '3T'
GROUP_3T_FLAIR = '3T FLAIR'
GROUP_7T_ADAPTED = '7T adapted'
GROUP_7T_ADAPTED_FLAIR = '7T adapted FLAIR'
GROUP_7T_DEFAULT = '7T default'

# the group create_logical_union_condition() builds from the two main groups
GROUP_3T_UNION_7T = f'{GROUP_3T}-union-{GROUP_7T_ADAPTED}'



def latest_vol_eval_path():
    """The most recent data/results/vol_eval_<timestamp>.csv.

    The '2' after 'vol_eval_' is the start of the timestamp; it keeps this glob from
    also matching data/results/vol_eval_clusters_<timestamp>.csv.
    """
    paths_raw = glob.glob('data/results/vol_eval_2*.csv')

    # select file with latest timestamp
    paths_raw.sort()
    print(f'Found {len(paths_raw)} vol_eval CSV files, using latest: {paths_raw[-1]}')
    return paths_raw[-1]


def load_and_prepare_data(select_fcd_groups=True):
    """Load the MELD-graph results and apply the cohort selection shared by all notebooks.

    Reads the most recent data/results/vol_eval_<timestamp>.csv - re-run
    evaluation_meldgraph/vol_eval.py whenever the MELD-graph predictions change, older files
    are kept as a record but have a different set of columns. Keeps the subjects marked
    include == 1 that have data at both 3T and 7T.

    select_fcd_groups=False stops before select_fcd_control_groups(), for the subject counts
    over the whole cohort reported in 01_tables.
    """
    eval_stats_df = pd.read_csv(latest_vol_eval_path())

    # rename site
    eval_stats_df.rename(columns={'site': 'source'}, inplace=True)

    # select only model == 'meld_graph'
    eval_stats_df = eval_stats_df[eval_stats_df['model'] == 'meld_graph']

    # add column "site" with more publication-ready descriptions
    # keep column source to be able to construct paths if necessary
    eval_stats_df['site'] = eval_stats_df['source'].replace({
        'cambridge': 'Cambridge',
        'bonn_7T': 'Bonn',
        'RICE': 'KCL',
        '7T_rory': 'UCL',
    })

    # to get unique IDs, merge columns "site" and "subject ID" into "site_subj_id"
    eval_stats_df['site_subj_id'] = eval_stats_df['site'] + '_' + eval_stats_df['subject ID'].astype(str)

    # split the B0_condition column into two columns: 'B0' and 'condition'
    B0_condition_split = eval_stats_df['B0_condition'].str.split('_', expand=True).iloc[:, :2]
    if B0_condition_split.shape[1] < 2:
        B0_condition_split[1] = ''
    eval_stats_df[['B0', 'condition']] = B0_condition_split

    # add column analysis_group that is one of '7T_adapted', '7T_adapted_FLAIR', '7T_default',
    # '3T', '3T_FLAIR' or 'Other', based on the B0_condition column
    def assign_analysis_group(row):
        if row['B0_condition'] in MAIN_7T_ADAPTED_CONDITIONS:
            return GROUP_7T_ADAPTED
        if row['B0_condition'] in FLAIR_7T_ADAPTED_CONDITIONS:
            return GROUP_7T_ADAPTED_FLAIR
        elif row['B0_condition'] in MAIN_7T_DEFAULT_CONDITIONS:
            return GROUP_7T_DEFAULT
        elif row['B0_condition'] in MAIN_3T_CONDITIONS:
            return GROUP_3T
        elif row['B0_condition'] in FLAIR_3T_CONDITIONS:
            return GROUP_3T_FLAIR
        else:
            return 'Other'
    eval_stats_df['analysis_group'] = eval_stats_df.apply(assign_analysis_group, axis=1)

    # make sure only include == 1 subjects are there
    eval_stats_df = eval_stats_df[eval_stats_df['include'] == 1]

    # after quality control, drop KCL sub-RICE075
    eval_stats_df = eval_stats_df[eval_stats_df['site_subj_id'] != 'KCL_sub-RICE075']

    if '7T_CP' in MAIN_7T_ADAPTED_CONDITIONS:
        eval_stats_df.loc[eval_stats_df['site'] == 'Cambridge', '7T_T1w_pTx'] = 'UP'
        # update the pTx mode to UP for Cambridge subjects if 7T_pTx is in the main analysis

    # select subjects that have data at both 3T and 7T
    subjects_7T = eval_stats_df[eval_stats_df['analysis_group'] == GROUP_7T_ADAPTED]['site_subj_id'].unique()
    subjects_3T = eval_stats_df[eval_stats_df['analysis_group'] == GROUP_3T]['site_subj_id'].unique()
    subjects_both = np.intersect1d(subjects_7T, subjects_3T)

    # and filter to only include those subjects
    eval_stats_df = eval_stats_df[eval_stats_df['site_subj_id'].isin(subjects_both)].reset_index(drop=True)

    if select_fcd_groups:
        eval_stats_df = select_fcd_control_groups(eval_stats_df)

    return eval_stats_df


def select_fcd_control_groups(eval_stats_df):
    """Keep the subjects MELD-graph is evaluated on: the controls and the patients with
    category_FCD == 1."""
    # filter to only include controls or patients with category_FCD == 1 and controls
    eval_stats_df = eval_stats_df[
        (eval_stats_df['group'] == 'control') |
        ((eval_stats_df['group'] == 'patient') & (eval_stats_df['category_FCD'] == 1))
    ].reset_index(drop=True)

    return eval_stats_df


def filter_to_common_subjects(eval_stats_df, analysis_groups, harmo_conditions):
    # some analysis groups (e.g. 3T_FLAIR) may only be available for a reduced subset of subjects;
    # keep only subjects present under every (harmo, analysis_group) combination being compared, so all
    # conditions are evaluated on the same, fixed set of subjects
    subject_sets = [
        set(eval_stats_df.loc[(eval_stats_df['harmo'] == harmo) & (eval_stats_df['analysis_group'] == analysis_group), 'site_subj_id'])
        for harmo, analysis_group in itertools.product(harmo_conditions, analysis_groups)
    ]
    common_subjects = set.intersection(*subject_sets)
    return eval_stats_df[eval_stats_df['site_subj_id'].isin(common_subjects)]


harmo_labels = {'harmo': 'ComBat', 'noharmo': 'No harmo'}

# the colour scheme every comparison plot below shares: one hue family per analysis group (3T blue,
# both 7T conditions orange), the harmonisation as the saturation within that family
main_group_hue_family = {GROUP_3T: '#a9c8f0',
                         GROUP_7T_DEFAULT: '#f4b78a',
                         GROUP_7T_ADAPTED: '#f4b78a'}
harmo_saturation = {'harmo': 1.0, 'noharmo': 0.35}

def _with_saturation(hex_color, saturation_factor):
    # change saturation of a hex color by converting to HSL, multiplying the saturation, and
    # converting back to hex; colors are always passed around as hex, converted to RGB only
    # transiently for the colorsys math
    h, l, s = colorsys.rgb_to_hls(*mcolors.to_rgb(hex_color))
    s = max(0.0, min(1.0, s * saturation_factor))
    return mcolors.to_hex(colorsys.hls_to_rgb(h, l, s))

def _with_value(hex_color, value_factor):
    # darken (lower HSL value) so the errorbar stands out against its own bar
    h, l, s = colorsys.rgb_to_hls(*mcolors.to_rgb(hex_color))
    l = max(0.0, min(1.0, l * value_factor))
    return mcolors.to_hex(colorsys.hls_to_rgb(h, l, s))


def plot_on_surface(data,
                    label,
                    cmap='Oranges',
                    vmin=None,
                    vmax=None,
                    gamma=None,
                    continuous=False,
                    nan_color=None,
                    show_labels=False,
                    roi_border_labels=None, # per-vertex parcellation labels, outlined if given
                    roi_border_color='darkgrey',
                    cbar_label='count per vertex',
                    panel_width=200,
                    spacing=0.1, # around individual panels, as a fraction of panel_width
                    cbar_height=130):

    if gamma is not None:
        data = [np.power(d, gamma) for d in data]
        vmax = np.power(vmax, gamma) if vmax is not None else None

    # use pyvista to plot the summed overlays on the inflated surface
    # data and label are lists, each entry is a COLUMN in the plot,
    # with the two views stacked as the two rows
    os.environ['PYVISTA_OFF_SCREEN'] = 'true'
    os.environ['DISPLAY'] = ''  # Disable X server connection attempts
    os.environ['EGL_PLATFORM'] = 'surfaceless'  # Use surfaceless EGL platform
    pv.start_xvfb()
    pv.set_jupyter_backend('static')

    with tempfile.TemporaryDirectory(dir='tmp/') as temp_dir:
        cmd = f'apptainer run freesurfer_8.1.0.sif /bin/bash -c "cp \$FREESURFER_HOME/subjects/fsaverage_sym/surf/lh.inflated {temp_dir}"'
        subprocess.run(cmd, shell=True, check=True)
        fsaverage_surf = nib.freesurfer.io.read_geometry(f'{temp_dir}/lh.inflated')
        vertices = fsaverage_surf[0]
        faces = fsaverage_surf[1]

    n_cols = len(data)

    # world-space extents of the mesh, used to size each row so that the
    # viewport aspect matches the aspect of the view it contains
    ext = vertices.max(axis=0) - vertices.min(axis=0)   # [x, y, z]

    # row 0 (lateral): horizontal = y (ant-post), vertical = z (sup-inf)
    # row 1 (dorsal):  horizontal = y (ant-post), vertical = x (med-lat)
    vert_extent = [ext[2], ext[0]]
    aspect = [ext[1] / ext[2], ext[1] / ext[0]]   # ~1.43 and ~2.48

    # brain size in px, then the viewport it sits in (brain + whitespace)
    brain_px = [panel_width / a for a in aspect]
    row_px = [h * (1 + spacing) for h in brain_px]
    col_px = panel_width * (1 + spacing)
    window_size = (int(round(col_px * n_cols)),
                   int(round(row_px[0] + row_px[1] + cbar_height)))

    # single shared colour scale across all columns
    plotting_min = vmin if vmin is not None else 0
    plotting_max = vmax if vmax is not None else np.nanmax([np.nanmax(d) for d in data])
    print(f'Shared colour scale: {plotting_min} to {plotting_max}')
    print(f'View aspects: {aspect[0]:.2f}:1, {aspect[1]:.2f}:1 '
          f'-> brain heights {brain_px[0]:.0f}px, {brain_px[1]:.0f}px, '
          f'gap {spacing * panel_width:.0f}px between columns')

    # 2 view rows sized to their content + a thin strip for the colorbar
    plotter = pv.Plotter(shape=(3, n_cols),
                         row_weights=[row_px[0], row_px[1], cbar_height],
                         col_weights=[1] * n_cols,
                         groups=[(2, slice(None))],   # bottom row spans all columns
                         window_size=window_size,
                         border=False,
                         notebook=True)

    faces_pv = np.column_stack([[3] * len(faces), faces]).ravel()

    # the outlines of a parcellation, so that a map of per-ROI values reads as an ROI analysis and
    # not as a vertexwise one: every triangle edge whose two vertices carry different labels,
    # lifted off the surface along the vertex normal so that the surface does not hide it
    border_mesh = None
    if roi_border_labels is not None:
        border_edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        border_edges = border_edges[roi_border_labels[border_edges[:, 0]] !=
                                    roi_border_labels[border_edges[:, 1]]]
        border_edges = np.unique(np.sort(border_edges, axis=1), axis=0)
        border_points = (vertices +
                         0.003 * ext.max() * pv.PolyData(vertices, faces_pv).point_normals)
        border_mesh = pv.PolyData(border_points,
                                  lines=np.column_stack([np.full(len(border_edges), 2),
                                                         border_edges]).ravel())
        print(f'{len(border_edges)} parcellation border edges outlined in {roi_border_color}')

    for i in range(n_cols):
        mesh_lh = pv.PolyData(vertices, faces_pv)
        mesh_lh['overlay'] = data[i]

        for row in range(2):
            plotter.subplot(row, i)
            plotter.add_mesh(mesh_lh, scalars='overlay', cmap=cmap,
                             clim=[plotting_min, plotting_max], show_scalar_bar=False,
                             nan_color=nan_color)
            if border_mesh is not None:
                plotter.add_mesh(border_mesh, color=roi_border_color, line_width=1)
            if show_labels and row == 0:
                plotter.add_text(label[i], position='upper_edge', font_size=8, color='black')
            plotter.camera_position = 'yz'
            if row == 0:
                plotter.camera.azimuth = 180
            else:
                plotter.camera.elevation = -90

            # parallel projection + explicit scale: the mesh fills the viewport
            # exactly, with `spacing` as the only margin. Because the viewport
            # was grown by the same (1 + spacing) factor, the margin is
            # spacing/2 on every side and the brain keeps its intended size.
            plotter.enable_parallel_projection()
            plotter.camera.parallel_scale = 0.5 * vert_extent[row] * (1 + spacing)
            plotter.reset_camera_clipping_range()

    # single horizontal colorbar spanning the bottom row
    fig, ax = plt.subplots(figsize=(3 * n_cols, 0.5))
    fig.patch.set_alpha(0.0)
    fig.patch.set_edgecolor('none')
    fig.patch.set_linewidth(0)
    ax.axis('off')

    if gamma is not None:
        vmax_linear = np.power(plotting_max, 1 / gamma)
        norm = matplotlib.colors.PowerNorm(gamma=gamma, vmin=0, vmax=vmax_linear)
        # steps evenly spaced in the gamma/visual space, so ticks are denser near 0
        # matching the power-law colour mapping
        ticks = [t for t in [1, 5, 10, 20, 30, 50] if t <= vmax_linear]
    elif continuous:
        # the values plotted are not counts, so map them continuously
        norm = matplotlib.colors.Normalize(vmin=plotting_min, vmax=plotting_max)
        ticks = np.linspace(plotting_min, plotting_max, 5)
    else:
        # one colour band per integer count
        ncolors = 256 if plotting_max > plotting_min else 1  # to show the 0 colour when all values are 0
        norm = matplotlib.colors.BoundaryNorm(
            boundaries=np.arange(plotting_min, plotting_max + 2, 1) - 0.5, ncolors=ncolors)
        ticks = np.arange(plotting_min, plotting_max + 1,
                          np.ceil((plotting_max - plotting_min + 1) / 3.).astype(int))

    cbar = plt.colorbar(
        plt.cm.ScalarMappable(
            cmap=cmap,
            norm=norm,
        ),
        ax=ax,
        orientation='horizontal',
        fraction=1.0,
        shrink=2 * 0.8 / n_cols if n_cols > 1 else 0.8,
    )

    cbar.set_ticks(ticks)
    cbar.ax.tick_params(width=1.5)
    cbar.ax.tick_params(labelsize=12)
    if cbar_label is not None:
        cbar.set_label(cbar_label, size=12)
    if gamma is not None:
        minor = np.concatenate([np.arange(0, 10, 1), np.arange(10, 100, 5)])
        cbar.ax.xaxis.set_minor_locator(
            matplotlib.ticker.FixedLocator(minor[minor <= vmax_linear]))
        cbar.ax.set_xticklabels([f'{t:.0f}' for t in ticks])
    elif continuous:
        cbar.ax.minorticks_off()
        cbar.ax.set_xticklabels([f'{t:.3g}' for t in ticks])
    else:
        cbar.ax.minorticks_off()

    # remove border from colorbar
    cbar.outline.set_visible(False)
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)

    plotter.subplot(2, 0)  # the grouped bottom row
    chart = pv.ChartMPL(fig)
    chart.size = (1.0, 1.0)
    chart.loc = (0.0, 0.00)
    chart.border_width = 0
    chart.border_style = None
    chart.border_color = (0, 0, 0, 0)
    chart.background_color = (0, 0, 0, 0)

    plt.close(fig)
    plotter.add_chart(chart)

    return plotter


# the analysis groups compared from here on, in both harmonisation conditions - six conditions in
# total. they are also held as a single label per (analysis group, harmonisation) so that the
# unpaired and paired tests below can treat them as one grouping
analysis_groups_clusters = [GROUP_3T, GROUP_7T_DEFAULT, GROUP_7T_ADAPTED]
harmo_conditions_clusters = ['noharmo', 'harmo']
harmonisation_order = [harmo_labels[harmo] for harmo in harmo_conditions_clusters]
analysis_conditions = [f'{analysis_group} {harmonisation}'
                       for analysis_group in analysis_groups_clusters
                       for harmonisation in harmonisation_order]

# similar colours as the sensitivity / specificity bars above, slightly darker
# also desaturated for the "no Harmo" condition
scatterplot_hue_family = {key: _with_value(value, 0.7) for key, value in main_group_hue_family.items()}
analysis_condition_colors = {
    f'{analysis_group} {harmo_labels[harmo]}':
        _with_saturation(scatterplot_hue_family[analysis_group], harmo_saturation[harmo])
    for analysis_group in analysis_groups_clusters
    for harmo in harmo_conditions_clusters}

def color_strips_by_condition(grid):
    # seaborn collection and analysis_conditionts follow the same order, so zip()
    # them together to assign the correct colours
    for ax in grid.axes.flat:
        for collection, condition in zip(ax.collections, analysis_conditions):
            collection.set_facecolor(analysis_condition_colors[condition])

def label_strips_by_condition(grid):
    # two rows of x-axis labels instead of a colour legend, as the sensitivity / specificity bars
    # have: the harmonisation under each dodged strip, the analysis group under the pair. seaborn
    # dodges the hue levels evenly across a category width of 0.8
    dodge_width = 0.8
    n_harmonisations = len(harmonisation_order)
    dodge_offsets = (np.linspace(0, dodge_width * (n_harmonisations - 1) / n_harmonisations,
                                 n_harmonisations)
                     + dodge_width / (2 * n_harmonisations) - dodge_width / 2)

    for ax in grid.axes.flat:
        # the inner axes of a shared x-axis keep their tick labels hidden, so leave them alone
        if not (ax.get_xticklabels() and ax.get_xticklabels()[0].get_visible()):
            ax.tick_params(axis='x', which='major', pad=0, length=0, labelsize=0)
            continue


        ax.set_xticks(range(len(analysis_groups_clusters)))
        ax.set_xticklabels(analysis_groups_clusters)
        ax.set_xticks([group_index + dodge_offset
                       for group_index in range(len(analysis_groups_clusters))
                       for dodge_offset in dodge_offsets], minor=True)
        ax.set_xticklabels(harmonisation_order * len(analysis_groups_clusters), minor=True,
                           fontsize=8, rotation=70, ha='right', rotation_mode='anchor')
        ax.tick_params(axis='x', which='minor', pad=2, length=0)
        ax.tick_params(axis='x', which='major', pad=50, length=0, labelsize=10)

    # add vertical lines between the analysis groups, at the midpoints of the gaps between them
    for ax in grid.axes.flat:
        for group_index in range(len(analysis_groups_clusters) - 1):
            x = (group_index + 0.5)
            ax.axvline(x=x, color='black', alpha=0.2, linewidth=1, zorder=0)
            # and extend below the x-axis if this is the last row or sharex is False
            if ax.get_subplotspec().is_last_row() or not ax.get_shared_x_axes().joined(ax, grid.axes.flat[-1]):
                line = matplotlib.lines.Line2D([x, x], [ax.get_ylim()[0], ax.get_ylim()[0] - 0.3 * (ax.get_ylim()[1] - ax.get_ylim()[0])],
                                            color='black', alpha=0.2, linewidth=1, zorder=0)
                line.set_clip_on(False)
                ax.add_line(line)

# the FreeSurfer surface features meld_graph extracts, in the naming vol_eval.py writes them under
surf_feature_names = {'curv': 'Mean curvature',
                      'pial.K_filtered.sm20': 'Intrinsic curvature',
                      'sulc': 'Sulcal depth',
                      'thickness': 'Cortical thickness',
                      'w-g.pct': 'White-grey contrast'}


def friedman_across_conditions(long_df, conditions, key_col='metric', value_col='value',
                               group_col='analysis_condition', correction_method='fdr_bh'):
    # compare the conditions per key (a cluster metric or a surface feature) with a Friedman test
    # over subjects, and correct the p-values across keys with correction_method (any method
    # statsmodels' multipletests accepts, e.g. 'fdr_bh' or 'bonferroni')
    long_df = long_df[long_df[group_col].isin(conditions)]

    # pair the observations by subject: one value per (subject, key, condition), averaged over the
    # clusters a subject has in that condition, so every subject contributes exactly one
    # observation per condition
    paired = pd.pivot_table(long_df, index=['site_subj_id', key_col],
                            columns=group_col, values=value_col,
                            aggfunc='mean').reset_index()
    # the Friedman test needs complete cases across all conditions
    paired = paired.dropna(subset=conditions)

    p_values = {}
    for key in paired[key_col].unique():
        data_groups = paired[paired[key_col] == key]
        _, p_values[key] = friedmanchisquare(*[data_groups[condition] for condition in conditions])

    p_values_df = pd.DataFrame.from_dict(p_values, orient='index', columns=['p_value']).reset_index()
    p_values_df = p_values_df.rename(columns={'index': key_col}).dropna()
    p_values_df['p_value_corrected'] = multipletests(p_values_df['p_value'], method=correction_method, alpha=0.05)[1]

    return paired, p_values_df

def kruskal_across_conditions(long_df, conditions, key_col='metric', value_col='value',
                              group_col='analysis_condition', correction_method='fdr_bh'):
    # compare the conditions per key with an unpaired Kruskal-Wallis test, and correct the p-values
    # across keys with correction_method. every cluster is its own observation, so subjects that
    # have a cluster under only some of the conditions still contribute
    long_df = long_df[long_df[group_col].isin(conditions)]

    p_values = {}
    for key in long_df[key_col].unique():
        data_groups = [long_df.loc[(long_df[key_col] == key) &
                                   (long_df[group_col] == condition), value_col].dropna()
                       for condition in conditions]
        if min(len(data) for data in data_groups) == 0:
            continue
        _, p_values[key] = kruskal(*data_groups)

    p_values_df = pd.DataFrame.from_dict(p_values, orient='index', columns=['p_value']).reset_index()
    p_values_df = p_values_df.rename(columns={'index': key_col}).dropna()
    p_values_df['p_value_corrected'] = multipletests(p_values_df['p_value'], method=correction_method, alpha=0.05)[1]

    return p_values_df

def describe_across_conditions(long_df, conditions, key_col='metric', value_col='value',
                               group_col='analysis_condition'):
    # n, mean and median per key and condition, as one wide row per key, so that the p-value tables
    # can show the descriptive statistics next to the test result
    long_df = long_df[long_df[group_col].isin(conditions)]
    # melting the mixed-dtype report columns gives an object column, so make it numeric here
    long_df = long_df.assign(**{value_col: pd.to_numeric(long_df[value_col], errors='coerce')})

    described = long_df.groupby([key_col, group_col])[value_col].agg(['count', 'mean', 'median'])
    described = described.unstack(group_col).swaplevel(axis='columns')
    # keep the three statistics of a condition together, in the order they are compared in
    described = described.reindex(columns=pd.MultiIndex.from_product([conditions,
                                                                     ['count', 'mean', 'median']]))
    described.columns = [f'{statistic} {condition}'
                         for condition, statistic in described.columns]

    return described.reset_index()



def load_surf_feature_maps(eval_stats_df):
    """The normalised per-vertex surface features of every subject and condition.

    Returns ({(harmo, analysis_group, site_subj_id, feature): (2, NVERT) array}, cortex_mask),
    left hemisphere first, right hemisphere second, cached as
    data/results/vol_eval_surf_features_<timestamp>.pkl next to the results the rows come from.
    """
    results_timestamp = latest_vol_eval_path().split('vol_eval_')[-1].removesuffix('.csv')

    # the fsaverage_sym labels, copied out of the freesurfer container on first use
    copy_fsaverage_sym_labels()

    # the features are zero outside the cortex label they were written on, so every mask below is
    # intersected with it
    cortex_mask = np.zeros(NVERT, dtype=bool)
    cortex_mask[nib.freesurfer.io.read_label(FSAVERAGE_SYM_CORTEX_LABEL_PATH)] = True

    print(f'{cortex_mask.sum()} cortical vertices')

    # MELD-graph input features ("normalised") per vertex, gathered and written to a cache pickle dict:
    # {(harmo, analysis_group, site_subj_id, feature): (2, NVERT) array}, left hemisphere first, right hemisphere
    # second
    surf_features_path = f'data/results/vol_eval_surf_features_{results_timestamp}.pkl'

    # one row per subject, acquisition condition and harmonisation
    feature_rows = eval_stats_df[eval_stats_df['analysis_group'].isin(analysis_groups_clusters)]
    # (assumes that there is only one B0_condition per subject per analysis_group)
    feature_rows = feature_rows.drop_duplicates(subset=['harmo', 'analysis_group', 'site_subj_id'])

    if os.path.exists(surf_features_path):
        with open(surf_features_path, 'rb') as features_file:
            surf_feature_maps = pickle.load(features_file)
        print(f'{len(surf_feature_maps)} vertexwise maps read for {results_timestamp}.')
    else:
        surf_feature_maps = {}
        for _, row in feature_rows.iterrows():
            print(f'Extracting surface features: {row["site_subj_id"]}, {row["B0_condition"]}, '
                  f'harmo {row["harmo"]}')
            if pd.isna(row['prediction_path']):
                print(f'No prediction path for {row["site_subj_id"]} {row["B0_condition"]}, skipped.')
                continue

            # only the normalised features, the intra- and then inter-subject z-scored ones the
            # classifier itself sees; the raw variant, the harmonised one meld_graph calls 'combat'
            # (the smoothed raw feature under noharmo, the ComBat-harmonised one under harmo) and the
            # asymmetry variant are not interpreted here and are not read at all
            features = read_surf_features(row['prediction_path'].split('/output/')[0],
                                          row['subject ID'], variants=['norm'])
            if features is None:
                continue

            # outside the cortex label the features are zero and comparing them is meaningless, so it
            # is dropped here rather than in every cell that reads the maps back
            for feature_name in surf_feature_names:
                maps = []
                for hemi in ['lh', 'rh']:
                    if ('norm', hemi, feature_name) not in features:
                        print(f'{row["site_subj_id"]} has no {feature_name} {hemi} in '
                              f'{row["analysis_group"]}, no map for it.')
                        maps.append(np.full(NVERT, np.nan, dtype=np.float32))
                        continue
                    maps.append(np.where(cortex_mask, features[('norm', hemi, feature_name)],
                                         np.nan).astype(np.float32))

                surf_feature_maps[(row['harmo'], row['analysis_group'], row['site_subj_id'],
                                   feature_name)] = np.stack(maps)

        with open(surf_features_path, 'wb') as features_file:
            pickle.dump(surf_feature_maps, features_file)
        print(f'{len(surf_feature_maps)} vertexwise maps written for {results_timestamp}.')

    return surf_feature_maps, cortex_mask
