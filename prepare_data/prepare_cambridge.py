

import os
import json
import shutil

import pandas as pd


def write_additional_files(subj_ids, source_dir, dest_dir):
    # create demographics_file.csv
    demographics = pd.read_csv(f'{source_dir}/participants.tsv', sep='\t', dtype={'participant_id': str})
    demographics.rename(columns={'participant_id': 'ID', 'age':'Age at preoperative', 'sex': 'Sex'}, inplace=True)
    demographics = demographics[demographics['ID'].str.contains('|'.join(subj_ids))] # keep only rows where one of subj_ids is in 'ID' column
    #prepend 'sub-' to ID values
    demographics['ID'] = demographics['ID'].apply(lambda x: f'sub-{x}')
    demographics['Group'] = 'patient'
    demographics['Harmo code'] = f'UHF'
    demographics = demographics[['ID', 'Harmo code', 'Group', 'Age at preoperative', 'Sex']]
    demographics.to_csv(f'{dest_dir}/demographics_file.csv', index=False)
    # copy demographics_file.csv to parent directory of dest_dir
    shutil.copy2(f'{dest_dir}/demographics_file.csv', f'{os.path.dirname(dest_dir)}/demographics_file.csv')

    # create subjects_list.txt
    with open(f'{dest_dir}/subjects_list.txt', 'w') as f:
        for subj_id in subj_ids:
            f.write(f'sub-{subj_id}\n')

    # create dataset_description.json
    dataset_description = {"Name": "uhf_meld_partial_dataset", "BIDSVersion": "1.10.1"}
    with open(f'{dest_dir}/dataset_description.json', 'w') as json_file:
        json.dump(dataset_description, json_file)
