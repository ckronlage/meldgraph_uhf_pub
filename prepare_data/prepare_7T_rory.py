
import glob
import os
import shutil
import json
import pandas as pd
import re

from prepare_data.util import copy_json_sidecar

def copy_7T_rory(dest_dir,
                 source_dir='data/raw/7T_rory/bids/',
                 bids_config_filename='bids_config.json',
                 FLAIR=False,
                 B0='7T'):
    if B0 not in ['3T', '7T']:
        raise ValueError("B0 must be '3T' or '7T'")

    os.makedirs(dest_dir, exist_ok=True)

    # write participants.tsv
    raw_data = pd.read_csv(os.path.join(source_dir, '7-TLE_crf.csv'))
    #raw_data = raw_data[raw_data['Include'] == 1]
    raw_data = raw_data[raw_data['Study origin:'] != 'RICE'] # exclude RICE subjects
    subjects_df = pd.DataFrame()
    subjects_df['ID'] = raw_data['Record ID']
    subjects_df['Harmo code'] = 'UHF'
    subjects_df['Group'] = ['control' if x == 'Healthy volunteer' else 'patient' for x in raw_data.iloc[:,1]]
    # compute difference in years between 'Date of 7T MRI' and 'Date of birth'
    raw_data['Date of 7T MRI'] = pd.to_datetime(raw_data['Date of 7T MRI'], format='%Y-%m-%d')
    raw_data['Date of birth'] = pd.to_datetime(raw_data['Date of birth'], format='%Y-%m-%d')
    # age at preoperative in years
    subjects_df['Age at preoperative'] = (raw_data['Date of 7T MRI'] - raw_data['Date of birth']).dt.days // 365
    # sex converted to lowercase
    subjects_df['Sex'] = raw_data['Sex'].str.lower()

    T1w_files = glob.glob(os.path.join(source_dir, f'sub-*/ses-{B0}/**/*T1w*.nii*'), recursive=True)
    T1w_files = {re.search(r'(sub-\d+)', f).group(1): f for f in T1w_files}
    # to pandas dataframe
    T1w_df = pd.DataFrame(list(T1w_files.items()), columns=['ID', 'T1w_file'])

    # merge subjects_df and T1w_df on 'ID' 
    # to keep only subjects with T1w files and drop excluded subjects
    subjects_df = pd.merge(subjects_df, T1w_df, on='ID', how='inner')
    if FLAIR:
        # also merge FLAIR files, if available
        FLAIR_files = glob.glob(os.path.join(source_dir, f'sub-*/ses-{B0}/**/*FLAIR*.nii*'), recursive=True)
        FLAIR_files = {re.search(r'(sub-\d+)', f).group(1): f for f in FLAIR_files}
        # to pandas dataframe
        FLAIR_df = pd.DataFrame(list(FLAIR_files.items()), columns=['ID', 'FLAIR_file'])
        subjects_df = pd.merge(subjects_df, FLAIR_df, on='ID', how='inner')

    # save demographics 
    demographics = subjects_df.drop(columns=['T1w_file'])
    demographics.to_csv(os.path.join(dest_dir, 'demographics_file.csv'), index=False)
    # copy demographics_file.csv to parent directory of dest_dir
    shutil.copy2(f'{dest_dir}/demographics_file.csv', f'{os.path.dirname(dest_dir)}/demographics_file.csv')

    # copy T1w files to dest_dir, preserving directory structure
    for f in subjects_df['T1w_file']:
        dest_path = os.path.join(dest_dir, os.path.relpath(f, source_dir))
        # change suffix from _T1w_raw.nii.gz to _T1w.nii.gz
        dest_path = re.sub(r'_T1w_raw(\.nii(\.gz)?)$', r'_T1w\1', dest_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(f, dest_path)
        copy_json_sidecar(f, dest_path)

    if FLAIR:
        # copy FLAIR files to dest_dir, preserving directory structure
        for f in subjects_df['FLAIR_file'].dropna():
            dest_path = os.path.join(dest_dir, os.path.relpath(f, source_dir))
            # change suffix from _FLAIR_raw.nii.gz to _FLAIR.nii.gz
            dest_path = re.sub(r'_FLAIR_raw(\.nii(\.gz)?)$', r'_FLAIR\1', dest_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(f, dest_path)
            copy_json_sidecar(f, dest_path)

    # write dest_dir/bids_config.json
    config = {"T1": {"datatype": "anat",
                    "session": B0,
                    "suffix": "T1w"}   
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
