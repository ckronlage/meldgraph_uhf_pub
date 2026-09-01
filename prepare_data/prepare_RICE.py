
import glob
import os
import shutil
import json
import pandas as pd
import re

from prepare_data.util import copy_json_sidecar

def copy_RICE(dest_dir,
                 source_dir='data/raw/RICE/bids/',
                 bids_config_filename='bids_config.json',
                 FLAIR=False,
                 B0='7T'):
    if B0 not in ['3T', '7T']:
        raise ValueError("B0 must be '3T' or '7T'")

    os.makedirs(dest_dir, exist_ok=True)

    subjects_df = gather_T1w_flair_paths(source_dir, B0)

    if FLAIR:
        subjects_df = subjects_df.dropna(subset=['FLAIR_file'])

    # save demographics 
    demographics = subjects_df.drop(columns=['T1w_file', 'FLAIR_file'])
    demographics.to_csv(os.path.join(dest_dir, 'demographics_file.csv'), index=False)
    # copy demographics_file.csv to parent directory of dest_dir
    shutil.copy2(f'{dest_dir}/demographics_file.csv', f'{os.path.dirname(dest_dir)}/demographics_file.csv')

    # copy T1w/UNI files to dest_dir
    for i, row in subjects_df.iterrows():
        out_path = f'{dest_dir}/{os.path.relpath(row["T1w_file"], source_dir)}'
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        shutil.copy2(row["T1w_file"], out_path)    
        copy_json_sidecar(row["T1w_file"], out_path)
        if FLAIR and row["FLAIR_file"] is not None:
            out_path = f'{dest_dir}/{os.path.relpath(row["FLAIR_file"], source_dir)}'
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            shutil.copy2(row["FLAIR_file"], out_path)
            copy_json_sidecar(row["FLAIR_file"], out_path)

    # write dest_dir/bids_config.json
    config = {"T1": {"datatype": "anat",
                    "session": B0,
                    "suffix": "T1w" if B0 == '3T' else "UNIT1"}   
            }
    if FLAIR:
        config["FLAIR"] = {"datatype": "anat",
                           "session": B0,
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

def gather_T1w_flair_paths(source_dir, B0):
    raw_data = pd.read_csv('data/subjects.csv')
    raw_data = raw_data[raw_data['source'] == 'RICE']
    raw_data = raw_data[raw_data['include'] == 1]
    subjects_df = raw_data[['subject ID', 'group', 'age_years', 'sex']]
    subjects_df = subjects_df.rename(columns={
        'subject ID': 'ID',
        'group': 'Group',
        'age_years': 'Age at preoperative',
        'sex': 'Sex'
    })
    subjects_df['Harmo code'] = 'UHF'

    paths = []
    for subj in subjects_df['ID']:
        UNI_files =     glob.glob(f"{source_dir}/{subj}/ses-{B0}/anat/{subj}_ses-{B0}_*rec-scanner_UNIT1.nii.gz", recursive=True)
        UNIden_files =  glob.glob(f"{source_dir}/{subj}/ses-{B0}/anat/{subj}_ses-{B0}_*rec-offline_UNIT1.nii.gz", recursive=True)
        T1w_files =     glob.glob(f"{source_dir}/{subj}/ses-{B0}/anat/{subj}_ses-{B0}_*T1w.nii.gz", recursive=True)
        
        # if a T1w is found, use it
        if len(T1w_files) == 1:
            T1w = T1w_files[0]
        elif len(UNIden_files) == 1:
            T1w = UNIden_files[0]
        elif len(UNI_files) == 1:
            print("Warning: Using (probably non-denoised) UNIT1 file as T1-weighted input for subject", subj)
            T1w = UNI_files[0]
        else:
            print(f"Warning: No T1w file found for subject {subj} in session {B0}. Skipping subject.")
            continue
        
        FLAIR_files =   glob.glob(f"{source_dir}/{subj}/ses-{B0}/anat/{subj}_ses-{B0}_*FLAIR.nii.gz", recursive=True)
        if len(FLAIR_files) > 1:
            raise ValueError(f"Multiple FLAIR files found for subject {subj} in session {B0}.")
        if len(FLAIR_files) == 1:
            FLAIR = FLAIR_files[0]
        else:
            FLAIR = None

        paths.append({
            'ID': subj,
            'T1w_file': T1w,
            'FLAIR_file': FLAIR,
        })
        
    paths_df = pd.DataFrame(paths)

    # merge subjects_df and paths_df on 'ID' to keep only subjects with T1w volumes
    # and drop excluded subjects
    subjects_df = pd.merge(subjects_df, paths_df, on='ID', how='inner')
    return subjects_df
