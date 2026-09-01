
import os
import glob
import re
import shutil
import json

import numpy as np
import nibabel as nib
import pandas as pd

from prepare_data.util import copy_json_sidecar

SELECTED_PATHS_CSV = 'data/raw/bonn_7T/bids/derivatives/3T_7T_MELD_paths.csv'

def select_sessions_files(base_dir='data/raw/bonn_7T/bids'):
    all_niftis = glob.glob(f'{base_dir}/sub-*/**/*.nii.gz', recursive=True)
    subj_ids = [re.sub(r'.*(sub-.*?)_.*', r'\1', f) for f in all_niftis]
    ses = [re.sub(r'.*ses-(.*?)_.*', r'\1', f) for f in all_niftis]
    modalities = [re.sub(r'.*_(.*?)\.nii\.gz', r'\1', f) for f in all_niftis]
    ce = [True if 'ce-Gd' in f else False for f in all_niftis]

    max_voxel_sizes = []
    for f in all_niftis:
        #print('reading header of', f)
        img = nib.load(f)
        max_voxel_size = np.array(img.header.get_zooms()).max()
        max_voxel_sizes.append(max_voxel_size)


    all_niftis_df = pd.DataFrame({
        'file_path': all_niftis,
        'subj_id': subj_ids,
        'session': ses,
        'modality': modalities,
        'ce': ce,
        'max_voxel_size': max_voxel_sizes
    })

    session_data = []
    all_session_tsvs = glob.glob(os.path.join(base_dir, '**', '*sessions.tsv'), recursive=True)
    for tsv_file in all_session_tsvs:
        subj_id = re.sub(r'.*(sub-.*?)_.*', r'\1', tsv_file)
        session_df = pd.read_csv(tsv_file, sep='\t')
        session_df['session_id'] = session_df['session_id'].str.replace('ses-', '')
        # rename session_id to session to match with all_niftis_df
        session_df = session_df.rename(columns={'session_id': 'session'})
        session_df['subj_id'] = subj_id

        session_df['n_files'] = [len(all_niftis_df[((all_niftis_df['session'] == ses) &
                                                    (all_niftis_df['subj_id'] == subj_id))]) for ses in session_df['session']]
        session_df['n_FLAIR'] = [len(all_niftis_df[((all_niftis_df['modality'] == 'FLAIR') &
                                                    (all_niftis_df['session'] == ses) &
                                                    (all_niftis_df['subj_id'] == subj_id))]) for ses in session_df['session']]
        session_df['has_FLAIR'] = session_df['n_FLAIR'] > 0

        for ses in session_df['session']:
            t1w_voxel_sizes = all_niftis_df[((all_niftis_df['modality'] == 'T1w') &
                                                            (all_niftis_df['session'] == ses) &
                                                            (all_niftis_df['subj_id'] == subj_id))]['max_voxel_size'].values
            if len(t1w_voxel_sizes) == 0:
                continue
            t1w_voxel_sizes = np.stack(t1w_voxel_sizes)
            t1w_voxel_sizes = t1w_voxel_sizes.min()
            session_df.loc[session_df['session'] == ses, 'T1w_voxel_size'] = t1w_voxel_sizes

            flair_voxel_sizes = all_niftis_df[((all_niftis_df['modality'] == 'FLAIR') &
                                                            (all_niftis_df['session'] == ses) &
                                                            (all_niftis_df['subj_id'] == subj_id))]['max_voxel_size'].values
            if len(flair_voxel_sizes) == 0:
                continue
            flair_voxel_sizes = np.stack(flair_voxel_sizes)
            flair_voxel_sizes = flair_voxel_sizes.min() 
            session_df.loc[session_df['session'] == ses, 'FLAIR_voxel_size'] = flair_voxel_sizes


        # select 7T session, criteria:
        # 1. has FLAIR
        # 2. maximum number of files
        session_df.sort_values(by=['n_FLAIR', 'n_files'], ascending=[False, False], inplace=True)
        # select the first 7t session
        session_df['selected'] = False
        for i, row in session_df.iterrows():
            if '7t' in row['session'].lower():
                session_df.at[i, 'selected'] = True
                break

        if session_df['selected'].sum() == 0:
            print(f'Warning: No 7T session found for subject {subj_id} in {tsv_file}, skipping subject.')
            continue
        
        # select 3T session
        # criteria:
        # 1. has FLAIR
        # 2. lowest FLAIR voxel size
        # 3. minimum time difference to the selected 7T session
        # 4. maximum number of files

        time_at_7t_ses = session_df[session_df['selected'] == True]['rel_acq_time'].values[0]
        session_df['time_diff_7t_ses'] = np.abs(session_df['rel_acq_time'] - time_at_7t_ses)

        session_df.sort_values(by=['has_FLAIR', 'FLAIR_voxel_size', 'time_diff_7t_ses', 'n_files'], 
                            ascending=[False, True, True, False], 
                            inplace=True)
        for i, row in session_df.iterrows():
            if '3t' in row['session'].lower():
                session_df.at[i, 'selected'] = True
                break

        #print(session_df)
        session_data.append(session_df)

    all_session_data_df = pd.concat(session_data, ignore_index=True)

    all_data_df = pd.merge(all_niftis_df, all_session_data_df, on=['subj_id', 'session'], how='left')
    all_data_df = all_data_df.sort_values(by=['subj_id', 'session', 'modality']).reset_index(drop=True)

    # keep only selected sessions
    all_data_df = all_data_df[all_data_df['selected'] == True]

    # sort by max_voxel_size ascending
    all_data_df.sort_values(by=['subj_id', 'max_voxel_size'], ascending=True, inplace=True)
    
    # remove ce-Gd scans
    all_data_df = all_data_df[~all_data_df['ce']]
    # remove scans with voxel size > 2.05 mm
    all_data_df = all_data_df[all_data_df['max_voxel_size'] <= 2.05]

    output_paths = []
    for subj_id in all_data_df['subj_id'].unique():
        T1w_7T = all_data_df[((all_data_df['subj_id'] == subj_id) &
                            (all_data_df['modality'] == 'T1w') &
                            (all_data_df['session'].str.contains('7t')))]
        if len(T1w_7T) == 0:
            print(f'No T1w found for subject {subj_id} at 7T')
            continue
        T1w_7T = T1w_7T.iloc[0]['file_path']

        FLAIR = all_data_df[((all_data_df['subj_id'] == subj_id) &
                            (all_data_df['modality'] == 'FLAIR') &
                            (all_data_df['session'].str.contains('7t')))]
        
        FLAIR_7T = FLAIR.iloc[0]['file_path'] if len(FLAIR) > 0 else np.nan

        T1w_3T = all_data_df[((all_data_df['subj_id'] == subj_id) &
                            (all_data_df['modality'] == 'T1w') &
                            (all_data_df['session'].str.contains('3t')))]
        T1w_3T = T1w_3T.iloc[0]['file_path'] if len(T1w_3T) > 0 else np.nan

        FLAIR_3T = all_data_df[((all_data_df['subj_id'] == subj_id) &
                            (all_data_df['modality'] == 'FLAIR') &
                            (all_data_df['session'].str.contains('3t')))]
        FLAIR_3T = FLAIR_3T.iloc[0]['file_path'] if len(FLAIR_3T) > 0 else np.nan

        age_7T = all_data_df[all_data_df['file_path'] == T1w_7T]['age_at_scan'].values[0]
        age_7T_years = age_7T*5.0 -2.0 # convert to mean years from bonn age code

        output_paths.append({
            'subj_id': subj_id,
            'T1w_7T': T1w_7T,
            'FLAIR_7T': FLAIR_7T,
            'T1w_3T': T1w_3T,
            'FLAIR_3T': FLAIR_3T
        })

    paths_df = pd.DataFrame(output_paths)
    os.makedirs('tmp', exist_ok=True)
    paths_df.to_csv(SELECTED_PATHS_CSV, index=False)


def copy_bonn_7T(dest_dir,
                 source_dir='data/raw/bonn_7T/bids/',
                 bids_config_filename='bids_config.json',
                 FLAIR=False,
                 B0='7T'):
    if B0 not in ['3T', '7T']:
        raise ValueError("B0 must be '3T' or '7T'")

    os.makedirs(dest_dir, exist_ok=True)

    raw_data = pd.read_csv('data/subjects.csv')
    raw_data = raw_data[raw_data['source'] == 'bonn_7T']
    raw_data = raw_data[raw_data['include'] == 1]
    subjects_df = raw_data[['subject ID', 'group', 'age_years', 'sex']]
    subjects_df = subjects_df.rename(columns={
        'subject ID': 'ID',
        'group': 'Group',
        'age_years': 'Age at preoperative',
        'sex': 'Sex'
    })
    subjects_df['Harmo code'] = 'UHF'

    if not os.path.exists(SELECTED_PATHS_CSV):
        raise FileNotFoundError(f"Selected paths CSV file not found: {SELECTED_PATHS_CSV}. Please generate and edit first.")
    paths_df = pd.read_csv(SELECTED_PATHS_CSV)
    subjects_df = pd.merge(subjects_df, paths_df, left_on='ID', right_on='subj_id', how='inner')

    # save demographics
    demographics = subjects_df[['ID', 'Harmo code', 'Group', 'Age at preoperative', 'Sex']]
    demographics.to_csv(os.path.join(dest_dir, 'demographics_file.csv'), index=False)
    # copy demographics_file.csv to parent directory of dest_dir
    shutil.copy2(f'{dest_dir}/demographics_file.csv', f'{os.path.dirname(dest_dir)}/demographics_file.csv')

    T1w_column = 'T1w_3T' if B0 == '3T' else 'T1w_7T'
    FLAIR_column = 'FLAIR_3T' if B0 == '3T' else 'FLAIR_7T'

    subjects_df = subjects_df.dropna(subset=[T1w_column])
    if FLAIR:
        subjects_df = subjects_df.dropna(subset=[FLAIR_column])

    # copy T1w files to dest_dir, preserving directory structure
    for f in subjects_df[T1w_column]:
        dest_path = os.path.join(dest_dir, os.path.relpath(f, source_dir))
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(f, dest_path)
        copy_json_sidecar(f, dest_path)
    if FLAIR:
        # copy FLAIR files to dest_dir, preserving directory structure
        for f in subjects_df[FLAIR_column].dropna():
            dest_path = os.path.join(dest_dir, os.path.relpath(f, source_dir))
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(f, dest_path)
            copy_json_sidecar(f, dest_path)

    # write dest_dir/bids_config.json
    config = {"T1": {"datatype": "anat",
                    "suffix": "T1w"}   
            }
    if FLAIR:
        config["FLAIR"] = {"datatype": "anat",
                           "suffix": "FLAIR"}
    else:
        config["FLAIR"] = {"datatype": "none"}

    with open(os.path.join(dest_dir, bids_config_filename), 'w') as f:
        json.dump(config, f)

    # write dataset_description.json
    description = {
        "Name": "7T_rory",
        "BIDSVersion": "1.10.1",
    }
    with open(os.path.join(dest_dir, 'dataset_description.json'), 'w') as f:
        json.dump(description, f)

    # write subjects_list.txt
    with open(os.path.join(dest_dir, 'subjects_list.txt'), 'w') as f:
        for subject in subjects_df['ID']:
            f.write(f"{subject}\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Select the bonn_7T sessions and files to use.")
    parser.add_argument("--select_sessions_and_files", default=False, action='store_true',
                        help=f"If set, select sessions and files and save to CSV at {SELECTED_PATHS_CSV}")
    args = parser.parse_args()

    if not args.select_sessions_and_files:
        parser.print_help()
        exit(1)

    select_sessions_files()
