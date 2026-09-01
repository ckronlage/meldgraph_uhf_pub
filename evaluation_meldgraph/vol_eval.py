import os
import re
import glob
import datetime
import difflib
import multiprocessing
import subprocess

import h5py
import numpy as np
import nibabel as nib
import pandas as pd
import scipy


# columns identifying a set of predictions, and equally a set of surface features: all but the raw
# features depend on the harmonisation, just like the predictions do
PREDICTION_LABEL_COLUMNS = ['model', 'harmo', 'B0_condition', 'site', 'subject ID']

# the surface features the meld_graph pipeline extracts, in the naming of the raw feature matrix;
# all are resampled to the fsaverage_sym left hemisphere (rh through xhemi), so lh and rh values can
# be pooled directly
SURF_FEATURE_NAMES = ['curv', 'pial.K_filtered.sm20', 'sulc', 'thickness', 'w-g.pct']

# the same features after the smoothing the meld_graph preprocessing applies before harmonising and
# normalising them; pial.K_filtered.sm20 comes smoothed already and is left alone
SURF_FEATURE_SMOOTHED_NAMES = {'curv': 'curv.sm3',
                               'pial.K_filtered.sm20': 'pial.K_filtered.sm20',
                               'sulc': 'sulc.sm3',
                               'thickness': 'thickness.sm3',
                               'w-g.pct': 'w-g.pct.sm3'}

# the feature variants to read, as (feature matrix file, dataset name) per variant. the raw features
# live in <harmo_code>_<group>_featurematrix.hdf5; the smoothed and ComBat-harmonised ('combat'),
# the intra- and then inter-subject z-scored ('norm', the latter against the MELD control cohort)
# and the asymmetry-scored ('asym') ones in <harmo_code>_<group>_featurematrix_combat.hdf5.
SURF_FEATURE_VARIANTS = {
    'raw':    ('featurematrix',        '.on_lh.{feature}.mgh'),
    'combat': ('featurematrix_combat', '.combat.on_lh.{smoothed_feature}.mgh'),
    'norm':   ('featurematrix_combat', '.inter_z.intra_z.combat.on_lh.{smoothed_feature}.mgh'),
    'asym':   ('featurematrix_combat', '.inter_z.asym.intra_z.combat.on_lh.{smoothed_feature}.mgh'),
}

# the meld_graph base dirs to evaluate, as globs over data/meld_graph<_noharmo>/<site>/<B0>/
# (listed explicitly, so that data/meld_graph_scaffold/ - which holds no subject data - is not
# picked up). Every base dir is labelled by its path: the harmonisation by whether 'noharmo' is
# part of it, the site and the B0 condition by its last two components
MELD_GRAPH_DIRS = ['data/meld_graph/*/*/', 'data/meld_graph_noharmo/*/*/']

# vertices per fsaverage_sym hemisphere
NVERT = 163842

# the Desikan-Killiany parcellation and the cortex label of fsaverage_sym, copied out of the
# freesurfer container once and then read by every worker process
FSAVERAGE_SYM_ANNOT_PATH = 'data/results/fsaverage_sym/lh.aparc.annot'
FSAVERAGE_SYM_CORTEX_LABEL_PATH = 'data/results/fsaverage_sym/lh.cortex.label'


def extract_subject_session_from_path(path):
    """Extract subject ID and session from a file path."""
    subj_id = re.sub(r'.*(sub-.+?)[/_-].*', r'\1', path)
    ses_id = re.sub(r'.*(ses-.+?)[/_-].*', r'\1', path)
    if ses_id == path:
        ses_id = None
    if subj_id == path:
        raise ValueError(f"Could not extract subject ID from path: {path}")

    return subj_id, ses_id


def discover_input_paths_meld_graph(meld_graph_base_dir):
    """Discover T1w and UNIT1 input paths."""
    paths = glob.glob(f'{meld_graph_base_dir}/input/sub-*/**/*T1w.nii*', recursive=True)
    paths += glob.glob(f'{meld_graph_base_dir}/input/sub-*/**/*UNIT1.nii*', recursive=True)
    
    records = []
    for path in paths:
        subj_id, ses_id = extract_subject_session_from_path(path)
        records.append({'subject ID': subj_id, 'session': ses_id, 'input_path': path})
    return pd.DataFrame(records)

def discover_recon_ids_meld_graph(meld_graph_base_dir):
    """Discover T1w and UNIT1 input paths."""
    paths = glob.glob(f'{meld_graph_base_dir}/output/fs_outputs/sub-*/', recursive=True)    
    records = []
    for path in paths:
        subj_id, _ = extract_subject_session_from_path(path)
        records.append({'subject ID': subj_id, 'input_path': path})
    return pd.DataFrame(records)

def discover_prediction_paths_meld_graph(meld_graph_base_dir):
    """Discover prediction paths from meld_graph output."""
    paths = glob.glob(f'{meld_graph_base_dir}/output/predictions_reports/sub-*/predictions/prediction.nii.gz')
    records = []
    for path in paths:
        subj_id, _ = extract_subject_session_from_path(path)
        records.append({'subject ID': subj_id, 'prediction_path': path})
    return pd.DataFrame(records)

def discover_ground_truth_paths(site, include_HS=False):
    """Discover ground truth lesion mask paths for a site."""
    paths = glob.glob(f'data/raw/{site}/bids/derivatives/lesion_masks_coreg/*.nii.gz', recursive=True)
    if include_HS:
        paths += glob.glob(f'data/raw/{site}/bids/derivatives/lesion_masks_hs/*.nii.gz', recursive=True)
    records = []
    for path in paths:
        subj_id, ses_id = extract_subject_session_from_path(path)
        records.append({'subject ID': subj_id, 'session': ses_id, 'ground_truth_path': path})
    return pd.DataFrame(records)

def match_ground_truth_to_input(paths_df, ground_truth_df):
    """Match best ground truth file to each input based on filename similarity."""
    matched = []
    
    for _, row in paths_df.iterrows():
        subject_gt = ground_truth_df[
            (ground_truth_df['subject ID'] == row['subject ID']) & 
            (ground_truth_df['session'] == row['session'])
        ]
        
        if len(subject_gt) == 0:
            continue
        
        gts = subject_gt['ground_truth_path'].tolist()

        valid_gts = []
        for gt in gts:
            try:
                assert_matching_image_geometry(nib.squeeze_image(nib.load(row['input_path'])),
                                              nib.squeeze_image(nib.load(gt)))
                valid_gts.append(gt)
            except ValueError:
                continue
 
        best_gt = max(
            valid_gts,
            key=lambda gt: difflib.SequenceMatcher(
                None, os.path.basename(gt), os.path.basename(row['input_path'])
            ).ratio()
        )
        matched.append({'subject ID': row['subject ID'], 'ground_truth_path': best_gt})
    
    return pd.DataFrame(matched)


def assert_matching_image_geometry(image_a, image_b):
    """Validate that image dimensions and affines match."""
    img_a = nib.squeeze_image(image_a)
    img_b = nib.squeeze_image(image_b)

    if not (img_a.shape == img_b.shape):
        raise ValueError(
            f"Image shapes do not match for images:"
            f"shape {img_a.shape} for {image_a.get_filename()}"
            f"shape {img_b.shape} for {image_b.get_filename()}"
        )

    if not np.allclose(img_a.affine, img_b.affine, rtol=1e-3, atol=1e-5):
        raise ValueError(
            f"Image affines do not match for images:"
            f" {img_a.affine} for {image_a.get_filename()}"
            f" {img_b.affine} for {image_b.get_filename()}"
        )

def remove_clusters_smaller_than(threshold, data, keep_largest=False):
    # estimate connected components
    clusters, num_clusters = scipy.ndimage.label(data,
                                            structure=np.ones((3, 3, 3)))

    if num_clusters == 0:
        return data

    cluster_sizes = np.bincount(clusters.ravel())
    keep_mask = cluster_sizes >= threshold
    keep_mask[0] = False # background cluster should not be kept

    if keep_largest:
        largest_cluster = np.argmax(cluster_sizes[1:]) + 1
        keep_mask[largest_cluster] = True

    data_filtered = keep_mask[clusters]

    return data_filtered

def read_surf_features(meld_graph_base_dir, subject_id, variants=None):
    """Returns {(variant, hemi, feature): vertexwise array}, or None

    The surface features of one subject, read out of the hdf5 feature matrices the meld_graph
    preprocessing writes to <base>/output/preprocessed_surf_data/MELD_<harmo_code>/. Every variant
    but the raw one depends on the harmonisation, so the tree the base dir points at decides whether
    those are ComBat-harmonised ('UHF') or not ('noHarmo'). All of them are zero outside the
    fsaverage_sym cortex label.

    variants selects the feature variants to read, all of SURF_FEATURE_VARIANTS by default.
    """
    harmo_code = 'noHarmo' if 'meld_graph_noharmo' in meld_graph_base_dir else 'UHF'
    surf_data_dir = f'{meld_graph_base_dir}/output/preprocessed_surf_data/MELD_{harmo_code}'

    features = {}
    for variant, (file_name, dataset_template) in SURF_FEATURE_VARIANTS.items():
        if variants is not None and variant not in variants:
            continue

        # the noharmo run gets a synthesised demographics file that calls every subject a patient,
        # the harmonised run takes the group from the site's demographics, so both files are tried
        for group in ['patient', 'control']:
            hdf5_path = f'{surf_data_dir}/{harmo_code}_{group}_{file_name}.hdf5'
            if not os.path.exists(hdf5_path):
                continue

            with h5py.File(hdf5_path, 'r') as hdf5_file:
                # the layout is <harmo_code>/<scanner>/<group>/<subject>/<hemi>/<dataset>; the
                # subject is looked up by exact key, so that 'sub-1' does not match 'sub-13'
                surf_group = None
                for scanner in hdf5_file.get(harmo_code, {}):
                    surf_group = hdf5_file[harmo_code][scanner].get(group, {}).get(subject_id)
                    if surf_group is not None:
                        break
                if surf_group is None:
                    continue

                for feature_name in SURF_FEATURE_NAMES:
                    dataset_name = dataset_template.format(
                        feature=feature_name,
                        smoothed_feature=SURF_FEATURE_SMOOTHED_NAMES[feature_name])
                    for hemi in ['lh', 'rh']:
                        if dataset_name not in surf_group.get(hemi, {}):
                            print(f'{hdf5_path} has no {dataset_name} for {subject_id} {hemi}, skipping.')
                            continue
                        features[(variant, hemi, feature_name)] = surf_group[hemi][dataset_name][:]

            break

    if not features:
        print(f'No surface features for {subject_id} in {surf_data_dir}, skipping.')
        return None

    return features

def input_image_geometry(input_path):
    """Voxel sizes and field of view of one input image, as in evaluation_hs/hs_eval.py.

    The FLAIR conditions have a second input image; if the input image's own directory holds a file
    with 'FLAIR' in its name, the same statistics are reported for the first such file, prefixed
    'flair_'. Conditions without a FLAIR simply carry no 'flair_' columns.
    """
    if pd.isna(input_path) or not os.path.exists(input_path):
        print(f'input file path {input_path} does not exist, no image geometry reported.')
        return {'voxel_sizes': np.nan,
                'image_shape': np.nan,
                'image_dimensions': np.nan,
                'min_voxel_size': np.nan,
                'min_dimension': np.nan,
                'isotropic': np.nan}

    paths = {'T1w_': input_path}
    flair_paths = sorted(glob.glob(f'{os.path.dirname(input_path)}/*FLAIR*.nii*'))
    if flair_paths:
        paths['FLAIR_'] = flair_paths[0]

    geometry = {}
    for prefix, path in paths.items():
        img = nib.squeeze_image(nib.load(path))
        # rounded to 2 decimals, so that near-identical voxel dimensions still count as isotropic
        voxel_sizes = tuple(np.round(img.header.get_zooms(), 2))
        image_dimensions = np.array(img.header.get_zooms()) * np.array(img.shape)

        geometry.update({f'{prefix}voxel_sizes': voxel_sizes,
                         f'{prefix}image_shape': img.shape,
                         f'{prefix}image_dimensions': image_dimensions,
                         f'{prefix}min_voxel_size': min(voxel_sizes),
                         f'{prefix}min_dimension': min(image_dimensions),
                         f'{prefix}isotropic': len(set(voxel_sizes)) == 1})

    return geometry

def evaluate_prediction(subject_id,
                        group,
                        prediction_path,
                        ground_truth_path,
                        anat_path,
                        criterium,
                        criterium_threshold=None,
                        min_cluster_size_pred=0,
                        min_cluster_size_gt=20,
                        mode='meld_graph',
                        labeled_pred_surfs_output_folder=None,
                        gt_surf_fsav_folder=None,
                        load_meldgraph_cluster_stats=False):

    if pd.isna(prediction_path) or not os.path.exists(prediction_path):
        print(f'prediction file path {prediction_path} does not exist for subject {subject_id}. Returning recon_successful=False and np.nan for all metrics.')
        return {'recon_successful': False,
            'pred_volume': np.nan,
            'gt_volume': np.nan,
            'intersection_volume': np.nan,
            'dice': np.nan,
            'num_gt_clusters': np.nan,
            'num_pred_clusters': np.nan,
            'tp_clusters': np.nan,
            'fp_clusters': np.nan,
            'max_cluster_dice': np.nan,
            'min_cluster_distance': np.nan,
            'tp_patient': np.nan,
            'tn_control': np.nan,
            }, None


    pred_img = nib.squeeze_image(nib.load(prediction_path))

    if pd.notna(ground_truth_path) and os.path.exists(ground_truth_path):
        gt_img = nib.squeeze_image(nib.load(ground_truth_path))
        
    elif group == 'control':
        # create empty image with same shape as prediction
        gt_img = nib.Nifti1Image(np.zeros(pred_img.shape), pred_img.affine)
    else:
        raise ValueError(f"Ground truth path/file is missing for patient {subject_id}: {ground_truth_path}")

    assert_matching_image_geometry(pred_img, gt_img)

    pred_data_orig = pred_img.get_fdata()
    pred_data = pred_img.get_fdata() > 0 # to boolean

    # cluster prediction (use melg_graph indices or connected components in volume)
    if mode == 'meld_graph':
        # here, cluster indices already correspond to surface-based clusters
        # that don't necessarily need to be spatially connected in volume
        # cluster indices 1 and 100, 2 and 200 etc. correspond to same cluster
        #
        # min_cluster_size_pred is ignored
        pred_clusters = pred_data_orig.copy()

        # nearest neighbor resampling of volume indices can in rare cases lead to spurious
        # indices, e.g. [  0. ,   1. ,   1.5,   2. ,  51. , 100. , 100.5, 150. , 200. ]
        # we avoid this by a) removing all non-integer values, b) assuming a maximum cluster number of 40 
        # and c) only allowing multiples of 100 for indices >= 100

        def filter_meldgraph_cluster_id(cluster_id):
            if (
                (cluster_id <= 0)
                or (cluster_id % 1 != 0)
                or (40 < cluster_id < 100)
                or ((cluster_id >= 100) and (cluster_id % 100 != 0))
            ):
                return None
            if cluster_id < 100:
                return int(cluster_id)
            return int(cluster_id // 100)

        cluster_ids = []
        for raw_cluster_id in np.unique(pred_data_orig):
            mapped_id = filter_meldgraph_cluster_id(raw_cluster_id)
            if mapped_id is not None:
                cluster_ids.append(mapped_id)

        meldgraph_cluster_ids = np.array(sorted(set(cluster_ids)), dtype=int)

        for i in meldgraph_cluster_ids:
            pred_clusters[pred_data_orig == (100*i)] = i
        pred_cluster_ids = meldgraph_cluster_ids
        num_pred_clusters = len(pred_cluster_ids)
        if (num_pred_clusters > 0) and (len(pred_cluster_ids) != max(meldgraph_cluster_ids)):
            raise ValueError(f"For prediction path {prediction_path}, meldgraph cluster ids are not consecutive. Found {len(pred_cluster_ids)} clusters, but max cluster id is {max(meldgraph_cluster_ids)}. This may indicate missing clusters.")
    else:
        pred_data = remove_clusters_smaller_than(min_cluster_size_pred, pred_data)
        pred_clusters, num_pred_clusters = scipy.ndimage.label(pred_data,
                                                 structure=np.ones((3, 3, 3)))
        pred_cluster_ids = np.arange(1, num_pred_clusters + 1) # cluster ids are 1-indexed, background is 0



    gt_data = gt_img.get_fdata() > 0
    gt_data = remove_clusters_smaller_than(min_cluster_size_gt, gt_data)
    gt_clusters, num_gt_clusters = scipy.ndimage.label(gt_data,
                                             structure=np.ones((3, 3, 3)))      
     
    # convert gt back to nifti image for plotting
    gt_img = nib.Nifti1Image(gt_data.astype(np.float32), gt_img.affine) 

    # Calculate ground truth volume
    pred_vol = np.sum(pred_data) * np.prod(pred_img.header.get_zooms())
    gt_vol = np.sum(gt_data) * np.prod(gt_img.header.get_zooms())
    intersection_vol = np.sum(pred_data & gt_data) * np.prod(pred_img.header.get_zooms())

    # Calculate voxel-wise dice
    numerator = 2 * intersection_vol
    denominator = pred_vol + gt_vol
    dice = numerator / denominator if ((denominator > 0) and (gt_vol > 0)) else np.nan

    # loop over clusters and evaluate whether positive or negative for each
    tp_clusters = 0
    fp_clusters = 0

    max_dice = 0.0
    min_distance = np.inf

    pred_indices_tp = []
    pred_indices_fp = []

    for pred_cluster_id in pred_cluster_ids: # cluster ids are 1-indexed, background is 0
        pred_cluster_mask = (pred_clusters == pred_cluster_id)

        if num_gt_clusters == 0:
            # no ground truth clusters, all predicted clusters are false positives
            fp_clusters += 1
            pred_indices_fp = np.concatenate([pred_indices_fp, np.unique(pred_data_orig[pred_cluster_mask])])
            continue

        is_true_positive = False
        for gt_cluster_id in range(1, num_gt_clusters + 1):
            gt_cluster_mask = (gt_clusters == gt_cluster_id)

            cluster_metrics = evaluate_cluster(pred_cluster_mask, 
                                               gt_cluster_mask, 
                                               voxel_sizes=pred_img.header.get_zooms())
            
            # here we don't want to break the loop after a positive detection
            # because other clusters might have better dice or distance metrics
            # that we want to report later on
            if criterium == 'overlap_mm3':
                threshold = criterium_threshold if criterium_threshold is not None else 0.0
                if cluster_metrics['overlap_mm3'] > threshold:
                    is_true_positive = True
            elif criterium == 'dice':
                threshold = criterium_threshold if criterium_threshold is not None else 0.2
                if cluster_metrics['dice'] > threshold:
                    is_true_positive = True
            elif criterium == 'centroid_in_gt':
                if cluster_metrics['centroid_in_gt']:
                    is_true_positive = True
            elif criterium == 'centroid_distance':
                threshold = criterium_threshold if criterium_threshold is not None else 15.0
                if cluster_metrics['centroid_distance'] < threshold:
                    is_true_positive = True
            else:
                raise ValueError(f"Unknown criterium: {criterium}")

            max_dice = max(max_dice, cluster_metrics['dice'])
            min_distance = min(min_distance, cluster_metrics['centroid_distance'])

        if is_true_positive:
            tp_clusters += 1
            pred_indices_tp = np.concatenate([pred_indices_tp, np.unique(pred_data_orig[pred_cluster_mask])])
        else:
            fp_clusters += 1
            pred_indices_fp = np.concatenate([pred_indices_fp, np.unique(pred_data_orig[pred_cluster_mask])])


    tp_patient = (tp_clusters > 0) if group == 'patient' else np.nan
    tn_control = (fp_clusters == 0) if group == 'control' else np.nan

    # the surface predictions hold the cluster indices in fsaverage_sym space; they are used both
    # to re-label the clusters and to average the raw surface features inside them below
    surf_pred_imgs = {}
    if (mode == 'meld_graph') and ((labeled_pred_surfs_output_folder is not None) or
                                   load_meldgraph_cluster_stats):
        for hemi in ['lh', 'rh']:
            surf_pred_imgs[hemi] = nib.load(f'{os.path.dirname(prediction_path)}'
                                            f'/fsaverage_sym/{hemi}.prediction.mgh')

    if ((labeled_pred_surfs_output_folder is not None) and 
        (gt_surf_fsav_folder is not None) and 
        (mode == 'meld_graph')):
        os.makedirs(f'{labeled_pred_surfs_output_folder}/{subject_id}/', exist_ok=True)
        for hemi in ['lh', 'rh']:
            # re-label the surface predictions after the evaluation:
            # assign index 2 to vertices that belong to true positive clusters, 
            # index 1 to vertices that belong to false positive clusters
            surf_pred = surf_pred_imgs[hemi]
            surf_pred_data = surf_pred.get_fdata()

            # in surf_pred_lh_data, replace values in pred_indices_tp with 2, values in pred_indices_fp with 1
            labeled_surf_pred_data = np.zeros_like(surf_pred_data)
            for idx in pred_indices_tp:
                labeled_surf_pred_data[surf_pred_data == idx] = 2
            for idx in pred_indices_fp:
                labeled_surf_pred_data[surf_pred_data == idx] = 1

            labeled_surf = nib.Nifti1Image(labeled_surf_pred_data, surf_pred.affine)
            out_path = f'{labeled_pred_surfs_output_folder}/{subject_id}/{hemi}.prediction.mgh'
            nib.save(labeled_surf, out_path) 

            gt_surf_fsav_path = f'{gt_surf_fsav_folder}/{hemi}.gt.fsaverage_sym.mgh'

            if (group == 'patient' and os.path.exists(gt_surf_fsav_path)) or group == 'control':
                # label gt_surf_fsav
                # if patient is a true positive, set foreground vertices to 2
                # otherwise (false negative) set foreground vertices to 1
                if os.path.exists(gt_surf_fsav_path):
                    gt_surf_fsav = nib.load(gt_surf_fsav_path)
                elif group == 'control':
                    gt_surf_fsav = nib.Nifti1Image(np.zeros(surf_pred_data.shape), surf_pred.affine)
                gt_surf_fsav_data = gt_surf_fsav.get_fdata()
                labeled_gt_surf_fsav_data = np.zeros_like(gt_surf_fsav_data)
                if tp_patient:
                    labeled_gt_surf_fsav_data[gt_surf_fsav_data > 0] = 2
                else:
                    labeled_gt_surf_fsav_data[gt_surf_fsav_data > 0] = 1
                labeled_gt_surf_fsav = nib.Nifti1Image(labeled_gt_surf_fsav_data, gt_surf_fsav.affine)
                out_gt_fsav_path = f'{labeled_pred_surfs_output_folder}/{subject_id}/{hemi}.gt_labeled.fsaverage_sym.mgh'
                nib.save(labeled_gt_surf_fsav, out_gt_fsav_path)
        
    meldgraph_stats_df = None
    if (mode == 'meld_graph') and (load_meldgraph_cluster_stats) and (num_pred_clusters > 0):
        # one row per predicted cluster, with the mean of every feature variant over the whole
        # cluster, read from this tree's hdf5 feature matrices
        surf_features = read_surf_features(prediction_path.split('/output/')[0], subject_id)

        # squeezed to match the vertexwise surface features
        surf_pred_data = {hemi: img.get_fdata().squeeze()
                          for hemi, img in surf_pred_imgs.items()}

        # the features are zero outside the cortex label they were written on
        cortex_mask = np.zeros(NVERT, dtype=bool)
        cortex_mask[nib.freesurfer.io.read_label(FSAVERAGE_SYM_CORTEX_LABEL_PATH)] = True

        cluster_records = []
        for cluster_id in pred_cluster_ids:
            # a cluster lives in one hemisphere only, so pooling both is safe; the surface carries
            # the same i / 100*i duality as the volume (see filter_meldgraph_cluster_id above), and
            # the features are zero outside the cortex label they were written on
            cluster_masks = {hemi: (((data == cluster_id) | (data == 100 * cluster_id)) &
                                    cortex_mask)
                             for hemi, data in surf_pred_data.items()}
            record = {'cluster': cluster_id,
                      'cluster_type': ('tp' if cluster_id in pred_indices_tp else
                                       'fp' if cluster_id in pred_indices_fp else
                                       'unknown'),
                      'hemi': ','.join(hemi for hemi, mask in cluster_masks.items() if mask.any()),
                      'n_vertices': sum(mask.sum() for mask in cluster_masks.values())}

            for variant in SURF_FEATURE_VARIANTS:
                for feature_name in SURF_FEATURE_NAMES:
                    values_in_cluster = []
                    if surf_features is not None:
                        values_in_cluster = [surf_features[(variant, hemi, feature_name)][mask]
                                             for hemi, mask in cluster_masks.items()
                                             if (variant, hemi, feature_name) in surf_features]
                    values_in_cluster = (np.concatenate(values_in_cluster) if values_in_cluster
                                         else np.array([]))
                    if values_in_cluster.size == 0:
                        print(f'No surface vertices for cluster {cluster_id} of {subject_id} '
                              f'in {prediction_path}, no {variant} {feature_name} reported.')
                    record[f'{variant} {feature_name} mean'] = (values_in_cluster.mean()
                                                                if values_in_cluster.size
                                                                else np.nan)

            cluster_records.append(record)

        meldgraph_stats_df = pd.DataFrame(cluster_records)

        # the confidence is the one report column that cannot be recomputed from the surfaces
        path_meldgraph_stats = glob.glob(f'{os.path.dirname(os.path.dirname(prediction_path))}'
                                         f'/reports/info_clusters_sub*.csv')
        if len(path_meldgraph_stats) != 1:
            raise ValueError("Expected exactly one meldgraph cluster stats file")
        meldgraph_stats_df = pd.merge(meldgraph_stats_df,
                                      pd.read_csv(path_meldgraph_stats[0])[['cluster', 'confidence']],
                                      on='cluster', how='left', validate='one_to_one')

    return {'recon_successful': True,
            'pred_volume': pred_vol,
            'gt_volume': gt_vol,
            'intersection_volume': intersection_vol,
            'dice': dice,
            'num_gt_clusters': num_gt_clusters,
            'num_pred_clusters': num_pred_clusters,
            'tp_clusters': tp_clusters,
            'fp_clusters': fp_clusters,
            'max_cluster_dice': max_dice,
            'min_cluster_distance': min_distance,
            'tp_patient': tp_patient,
            'tn_control': tn_control,
            }, meldgraph_stats_df

def evaluate_cluster(pred,
                     gt, 
                     voxel_sizes = (1.0, 1.0, 1.0)):
    if gt.sum() == 0:
        # if no ground truth, false positive cluster
        return {'dice': 0.0,
                'overlap_mm3': 0.0,
                'centroid_in_gt': False,
                'centroid_distance': np.inf}
    

    if not pred.dtype == bool:
        raise ValueError("Predicted cluster mask must be boolean.")
    if not gt.dtype == bool:
        raise ValueError("Ground truth cluster mask must be boolean.")

    # dice-score
    numerator = 2 * np.sum(pred & gt)
    denominator = np.sum(pred) + np.sum(gt)
    dice = numerator / denominator if denominator != 0 else 1.0

    # overlap (in mm^3)
    overlap = np.sum(pred & gt)
    overlap_mm3 = overlap * np.prod(voxel_sizes)

    # centroid contained in gt
    pred_centroid = np.array(np.nonzero(pred)).mean(axis=1).astype(int)
    centroid_in_gt = gt[tuple(pred_centroid)] > 0

    # centroid distance (in mm)
    gt_centroid = np.array(np.nonzero(gt)).mean(axis=1).astype(int)
    centroid_distance = np.linalg.norm((pred_centroid - gt_centroid) * voxel_sizes)

    return {'dice': dice,
            'overlap_mm3': overlap_mm3,
            'centroid_in_gt': centroid_in_gt,
            'centroid_distance': centroid_distance}

def gt_to_fsav_surf(gt_paths_3T_df, output_folder, min_cluster_size_gt=20):
    def freesurfer_cmd(cmd):
        # escape cmd so that expansion happens in the inner shell
        cmd = cmd.replace('"', '\"').replace('$', '\\$').replace('`', r'\`')
        cmd = f'FS_LICENSE={os.getcwd()}/license.txt ' + cmd
        cmd_out = f'apptainer run freesurfer_8.1.0.sif /bin/bash -c "{cmd}"'
        subprocess.run(cmd_out, shell=True, check=True)

    for _, row in gt_paths_3T_df.iterrows():
        gt_vol_filtered_path = f'{output_folder}/{row["site"]}/{row["subject ID"]}/gt_volume.nii.gz'
        if os.path.exists(gt_vol_filtered_path):
            continue
        os.makedirs(os.path.dirname(gt_vol_filtered_path), exist_ok=True)

        gt_vol_path = row['ground_truth_path']
        # load gt volume, remove small clusters, and save back to disk for surface projection
        gt_img = nib.squeeze_image(nib.load(gt_vol_path))
        gt_data = gt_img.get_fdata() > 0
        gt_data = remove_clusters_smaller_than(min_cluster_size_gt, gt_data)
        gt_img = nib.Nifti1Image(gt_data.astype(np.float32), gt_img.affine)

        nib.save(gt_img, gt_vol_filtered_path)

        # use the 3T reconstruction to coregister lesion labels to fsaverage_sym surface space so that these are comparable
        fs_subjs_dir_3T = os.path.dirname(row['input_path']).split('/input/')[0] + '/output/fs_outputs/'

        for hemi in ['lh', 'rh']:
            gt_surf_path = f'{output_folder}/{row["site"]}/{row["subject ID"]}/{hemi}.gt.mgh'
            cmd = f'SUBJECTS_DIR={fs_subjs_dir_3T} mri_vol2surf --mov {gt_vol_filtered_path} --regheader {row["subject ID"]} --hemi {hemi} --o {gt_surf_path} --projfrac-max -1 1 0.2'
            cmd = freesurfer_cmd(cmd)

            # then coregister to fsaverage_sym
            gt_surf_raw_fsav_path = f'{output_folder}/{row["site"]}/{row["subject ID"]}/{hemi}.gt_raw.fsaverage_sym.mgh'
            if hemi == 'lh':
                cmd = f'SUBJECTS_DIR={fs_subjs_dir_3T} mris_apply_reg --src {gt_surf_path} --trg {gt_surf_raw_fsav_path} --streg {fs_subjs_dir_3T}/{row["subject ID"]}/surf/lh.fsaverage_sym.sphere.reg {fs_subjs_dir_3T}/fsaverage_sym/surf/lh.sphere.reg'
            else:
                cmd = f'SUBJECTS_DIR={fs_subjs_dir_3T} mris_apply_reg --src {gt_surf_path} --trg {gt_surf_raw_fsav_path} --streg {fs_subjs_dir_3T}/{row["subject ID"]}/xhemi/surf/lh.fsaverage_sym.sphere.reg {fs_subjs_dir_3T}/fsaverage_sym/surf/lh.sphere.reg'
            cmd = freesurfer_cmd(cmd)

            # only keep largest cluster / connected component 
            # this works because the mri_surfcluster outputs are ordered by size,
            # so the largest cluster is always assigned index 1
            gt_surf_fsav_path = f'{output_folder}/{row["site"]}/{row["subject ID"]}/{hemi}.gt.fsaverage_sym.mgh'
            cmd = f'SUBJECTS_DIR={fs_subjs_dir_3T} mri_surfcluster --in {gt_surf_raw_fsav_path} --subject fsaverage_sym --hemi {hemi} --thmin 0.5 --ocn {gt_surf_fsav_path}'
            cmd = freesurfer_cmd(cmd)
            gt_surf_fsav = nib.load(gt_surf_fsav_path)
            gt_surf_fsav_data = gt_surf_fsav.get_fdata()
            gt_surf_fsav_data[gt_surf_fsav_data != 1] = 0
            gt_surf_fsav = nib.Nifti1Image(gt_surf_fsav_data, gt_surf_fsav.affine)
            nib.save(gt_surf_fsav, gt_surf_fsav_path)


def copy_fsaverage_sym_labels():
    """Copy the Desikan-Killiany annotation and the cortex label of fsaverage_sym out of the
    freesurfer container."""
    for label_path in [FSAVERAGE_SYM_ANNOT_PATH, FSAVERAGE_SYM_CORTEX_LABEL_PATH]:
        if os.path.exists(label_path):
            continue

        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        cmd = (f'apptainer run freesurfer_8.1.0.sif /bin/bash -c '
               f'"cp \\$FREESURFER_HOME/subjects/fsaverage_sym/label/'
               f'{os.path.basename(label_path)} {label_path}"')
        subprocess.run(cmd, shell=True, check=True)


def discover_match_paths_meld_graph_dir(meld_graph_base_dir):
    print(f"Discovering paths in {meld_graph_base_dir}...")

    harmo = 'noharmo' if 'noharmo' in meld_graph_base_dir else 'harmo'
    site = meld_graph_base_dir.split('/')[-3]
    B0_condition = meld_graph_base_dir.split('/')[-2]

    input_df = discover_input_paths_meld_graph(meld_graph_base_dir)
    prediction_df = discover_prediction_paths_meld_graph(meld_graph_base_dir)

    if len(input_df) != len(prediction_df):
        raise ValueError(
            f"Number of input paths ({len(input_df)}) does not match "
            f"number of prediction paths ({len(prediction_df)}) for site {site} and condition {B0_condition}"
        )

    paths_df = pd.merge(input_df, prediction_df, on='subject ID', how='inner', validate='one_to_one')
    
    # Match and validate ground truths
    ground_truth_df = discover_ground_truth_paths(site)
    matched_gt = match_ground_truth_to_input(paths_df, ground_truth_df)
    
    paths_df = pd.merge(paths_df, matched_gt, on='subject ID', how='left')
    
    # Validate image geometry for each matched set of paths
    for _, row in paths_df.iterrows():
        if pd.isna(row['ground_truth_path']):
            continue
        assert_matching_image_geometry(nib.squeeze_image(nib.load(row['input_path'])),
                                       nib.squeeze_image(nib.load(row['prediction_path'])))
        assert_matching_image_geometry(nib.squeeze_image(nib.load(row['input_path'])),
                                       nib.squeeze_image(nib.load(row['ground_truth_path'])))

    # Add ids of subjects with attempted freesurfer reconstructions (created folders)
    # even if the reconstructions failed and thus no meld_graph outputs exist, to these
    # are visible in the final evaluation table with NaN for all metrics
    recon_ids = discover_recon_ids_meld_graph(meld_graph_base_dir)['subject ID']
    paths_df = pd.merge(recon_ids, paths_df, on='subject ID', how='left')

    # Add metadata columns
    paths_df['model'] = 'meld_graph'
    paths_df['harmo'] = harmo
    paths_df['site'] = site
    paths_df['B0_condition'] = B0_condition
    
    return paths_df[['model', 'harmo', 'site', 'B0_condition', 'subject ID', 'session', 'input_path', 'prediction_path', 'ground_truth_path']]

def evaluate_row(row):
    print(f'Evaluating: model {row["model"]}, site {row["site"]}, subject {row["subject ID"]}, B0_condition {row["B0_condition"]}, harmo {row["harmo"]}')
    labeled_pred_surfs_output_folder=f'data/results/labeled_predictions/{row["model"]}_{row["harmo"]}_{row["site"]}_{row["B0_condition"]}'
    gt_surf_fsav_folder=f'data/results/labeled_predictions/gt_fsaverage_sym/{row["site"]}/{row["subject ID"]}'
    eval_stats, cluster_stats = evaluate_prediction(subject_id=row['subject ID'], 
                                group=row['group'], 
                                prediction_path=row['prediction_path'],
                                ground_truth_path=row['ground_truth_path'],
                                anat_path=row['input_path'],
                                criterium='overlap_mm3',
                                criterium_threshold=0.0,
                                min_cluster_size_pred=0,
                                min_cluster_size_gt=20, 
                                mode=row['model'],
                                labeled_pred_surfs_output_folder=labeled_pred_surfs_output_folder,
                                gt_surf_fsav_folder=gt_surf_fsav_folder,
                                load_meldgraph_cluster_stats=True)
    eval_stats.update({col: row[col] for col in PREDICTION_LABEL_COLUMNS})
    # the geometry of the input image, reported here rather than in evaluate_prediction() so that
    # rows whose reconstruction failed still carry it
    eval_stats.update(input_image_geometry(row['input_path']))
    eval_stats.update({'path_labeled_pred_surfs': labeled_pred_surfs_output_folder,
                       'path_groundtruth_surf_fsaverage': gt_surf_fsav_folder})
    if cluster_stats is not None:
        for col in PREDICTION_LABEL_COLUMNS:
            cluster_stats[col] = row[col]
    return eval_stats, cluster_stats

def vol_eval_all(threads=1):
    # the fsaverage_sym annotation and cortex label the worker processes read below
    copy_fsaverage_sym_labels()

    clinical_df = pd.read_csv('data/subjects.csv')
    # rename column 'source' to 'site' for better merging later
    clinical_df = clinical_df.rename(columns={'source': 'site'})

    input_pred_gt_paths = []
    for dir_glob in MELD_GRAPH_DIRS:
        for dir_path in glob.glob(dir_glob):
            input_pred_gt_paths.append(discover_match_paths_meld_graph_dir(dir_path))

    ## for debugging only one site/B0_condition
    #input_pred_gt_paths = [discover_match_paths_meld_graph_dir('data/meld_graph/RICE/3T/')]
    
    input_pred_gt_paths_df = pd.concat(input_pred_gt_paths, ignore_index=True)

    # merge with clinical_df to get group labels
    input_pred_gt_paths_df = pd.merge(clinical_df, input_pred_gt_paths_df, on=['site', 'subject ID'], how='right')

    # filter for 'include' tag in clinical_df
    # needs to be done after merging with paths
    input_pred_gt_paths_df = input_pred_gt_paths_df[input_pred_gt_paths_df['include'] == 1]

    # filter to only include only rows where either group is 'control' or a
    # ground truth path exists (i.e. exclude patients without ground truth)
    input_pred_gt_paths_df = input_pred_gt_paths_df[((input_pred_gt_paths_df['group'] == 'control') | 
                                                     (pd.notna(input_pred_gt_paths_df['ground_truth_path'])))]


    # use 'B0_condition' == '3T' to project volume ground truths to fsaverage_sym surface space
    gt_paths_3T_df = input_pred_gt_paths_df[((input_pred_gt_paths_df['B0_condition'] == '3T') & 
                                             (input_pred_gt_paths_df['harmo'] == 'harmo') &
                                             (input_pred_gt_paths_df['model'] == 'meld_graph') &
                                             (pd.notna(input_pred_gt_paths_df['ground_truth_path'])))]
    
    gt_to_fsav_surf(gt_paths_3T_df, output_folder=f'data/results/labeled_predictions/gt_fsaverage_sym')


    rows = [row.to_dict() for _, row in input_pred_gt_paths_df.iterrows()]

    # non-parallel version for debugging
    #results = []
    #for row in rows:
    #    results.append(evaluate_row(row))

    with multiprocessing.Pool(processes=threads) as pool:
        results = pool.map(evaluate_row, rows)

    eval_stats = [d for d, _ in results]
    # cluster stats only exist for meld_graph rows with at least one predicted cluster
    cluster_stats = [df for _, df in results if df is not None]

    eval_stats_df = pd.DataFrame(eval_stats)

    eval_stats_df = pd.merge(input_pred_gt_paths_df, eval_stats_df, on=PREDICTION_LABEL_COLUMNS, how='right')
    # label columns first
    eval_stats_df = eval_stats_df[PREDICTION_LABEL_COLUMNS +
                                  [col for col in eval_stats_df.columns
                                   if col not in PREDICTION_LABEL_COLUMNS]]

    #save to data/results/vol_eval_<timestamp>.csv, with the per-cluster meld_graph stats in
    # data/results/vol_eval_clusters_<timestamp>.csv (both with the same timestamp).
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    eval_stats_df.to_csv(f'data/results/vol_eval_{timestamp}.csv', index=False)

    if cluster_stats:
        cluster_stats_df = pd.concat(cluster_stats, ignore_index=True)
        # label columns first
        cluster_stats_df = cluster_stats_df[PREDICTION_LABEL_COLUMNS +
                                            [col for col in cluster_stats_df.columns
                                             if col not in PREDICTION_LABEL_COLUMNS]]
        cluster_stats_df.to_csv(f'data/results/vol_eval_clusters_{timestamp}.csv', index=False)
    else:
        print('No meld_graph cluster stats found, not writing vol_eval_clusters csv.')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate lesion segmentation predictions against ground truth.')
    parser.add_argument('--threads', type=int, default=4, help='Number of parallel threads to use for evaluation')
    args = parser.parse_args()

    vol_eval_all(threads=args.threads)

