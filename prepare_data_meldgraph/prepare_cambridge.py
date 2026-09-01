
"""Copy the cambridge data into a MELD-graph condition folder.

Usage:
    python -m prepare_data_meldgraph.prepare_cambridge --meld_graph --B0_condition 7T_CP
"""

import os
import json
import shutil

import bids

from prepare_data.prepare_cambridge import write_additional_files
from prepare_data_meldgraph import mp2rage_denoise


MP2RAGE_DENOISE_BETA = 1.0

def prepare_meld_graph(source_dir = 'data/raw/cambridge/bids',
                       dest_dir = 'data/meld_graph/cambridge/7T_pTx/input',
                       bids_entities = {'session': '7T', 'acquisition': 'pTx'},
                       FLAIR = False):
    os.makedirs(dest_dir, exist_ok=True)

    layout = bids.BIDSLayout(source_dir, validate=False)
    subj_ids = layout.get_subjects()

    subj_ids_existing = []
    for s in subj_ids:
        print(f'Processing subject {s} ...')
        uni_path = layout.get(subject=s, suffix='UNIT1', extension=['.nii', '.nii.gz'], **bids_entities, return_type='file')
        inv1_path = layout.get(subject=s, inv='1', extension=['.nii', '.nii.gz'], **bids_entities, return_type='file')
        inv2_path = layout.get(subject=s, inv='2', extension=['.nii', '.nii.gz'], **bids_entities, return_type='file')

        if len(uni_path) != 1 or len(inv1_path) != 1 or len(inv2_path) != 1:
            print(f'  Missing MP2RAGE files for subject {s}, skipping ...')
            continue

        flair_path = layout.get(subject=s, suffix='FLAIR', extension=['.nii', '.nii.gz'], **bids_entities, return_type='file')
        if FLAIR and len(flair_path) != 1:
            print(f'  Missing FLAIR file for subject {s}, skipping ...')
            continue

        out_path = f'{dest_dir}/sub-{s}/anat/{os.path.basename(uni_path[0])}'
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        mp2rage_denoise.mp2rage_robust_combination(
            uni_path = uni_path[0],
            inv1_path = inv1_path[0],
            inv2_path = inv2_path[0],
            output_path = out_path,
            beta = MP2RAGE_DENOISE_BETA,
            overwrite = False
        )
        subj_ids_existing.append(s)

        if FLAIR:
            out_path = f'{dest_dir}/sub-{s}/anat/sub-{s}_ses-{bids_entities["session"]}_FLAIR.nii.gz'
            if len(flair_path) != 1:
                print(f'  Missing FLAIR file for subject {s}, skipping FLAIR ...')
                continue
            shutil.copy2(flair_path[0], out_path)

    # create bids_config.json different versions
    config = {"T1": {"datatype": "anat",
                     "session": bids_entities['session'],
                     "suffix": 'UNIT1'},
              "FLAIR": {"datatype": "none"}
              }
    if FLAIR:
        config["FLAIR"] = {"datatype": "anat",
                           "session": bids_entities['session'],
                           "suffix": "FLAIR"}
    with open(f'{dest_dir}/meld_bids_config.json', 'w') as json_file:
        json.dump(config, json_file)

    write_additional_files(subj_ids_existing, source_dir, dest_dir)


def prepare_meld_graph_nouhf(source_dir = 'data/raw/cambridge/bids',
                             dest_dir = 'data/meld_graph/cambridge/7T_pTx_nouhf/input',
                             bids_entities = {'suffix': 'UNIT1', 'session': '7T', 'acquisition': 'pTx'}):
    os.makedirs(dest_dir, exist_ok=True)

    layout = bids.BIDSLayout(source_dir, validate=False)
    subj_ids = layout.get_subjects()

    subj_ids_existing = []
    for s in subj_ids:
        print(f'Processing subject {s} ...')
        T1w_path = layout.get(subject=s, extension=['.nii', '.nii.gz'], **bids_entities, return_type='file')
        if len(T1w_path) != 1:
            print(f'  Missing file for subject {s}, skipping ...')
            continue
        out_path = f'{dest_dir}/sub-{s}/anat/{os.path.basename(T1w_path[0])}'
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        shutil.copy2(T1w_path[0], out_path)
        subj_ids_existing.append(s)

    # create bids_config.json different versions
    config = {"T1": {"datatype": "anat",
                     "session": bids_entities['session'],
                     "suffix": bids_entities['suffix']},
              "FLAIR": {"datatype": "none"}
              }
    with open(f'{dest_dir}/meld_bids_config.json', 'w') as json_file:
        json.dump(config, json_file)

    write_additional_files(subj_ids_existing, source_dir, dest_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Prepare cambridge data for MELD-graph.")
    parser.add_argument('--meld_graph', action='store_true', help='Prepare data for MELD-graph pipeline')
    parser.add_argument('--B0_condition', type=str, required=True, help='B0 condition to prepare (e.g., 7T_CP, 7T_pTx, 7T_pTx_nouhf, 7T_pTx_FLAIR)')
    args = parser.parse_args()

    if not args.meld_graph:
        parser.print_help()
        exit(1)

    if args.meld_graph:
        match args.B0_condition:
            case '7T_CP':
                prepare_meld_graph(dest_dir='data/meld_graph/cambridge/7T_CP/input',
                                   bids_entities={'session': '7T', 'acquisition': 'CP'})

            case '7T_CP_nouhf':
                prepare_meld_graph_nouhf(dest_dir='data/meld_graph/cambridge/7T_CP_nouhf/input',
                                         bids_entities={'suffix': 'UNIT1', 'session': '7T', 'acquisition': 'CP'})

            case '7T_CP_FLAIR':
                prepare_meld_graph(dest_dir='data/meld_graph/cambridge/7T_CP_FLAIR/input',
                                   bids_entities={'session': '7T', 'acquisition': 'CP'},
                                   FLAIR=True)
            case '7T_pTx':
                prepare_meld_graph(dest_dir='data/meld_graph/cambridge/7T_pTx/input',
                                   bids_entities={'session': '7T', 'acquisition': 'pTx'})

            case '7T_pTx_nouhf':
                prepare_meld_graph_nouhf(dest_dir='data/meld_graph/cambridge/7T_pTx_nouhf/input',
                                         bids_entities={'suffix': 'UNIT1', 'session': '7T', 'acquisition': 'pTx'})

            case '7T_pTx_FLAIR':
                prepare_meld_graph(dest_dir='data/meld_graph/cambridge/7T_pTx_FLAIR/input',
                                   bids_entities={'session': '7T', 'acquisition': 'pTx'},
                                   FLAIR=True)

            case '3T':
                prepare_meld_graph_nouhf(dest_dir='data/meld_graph/cambridge/3T/input',
                                   bids_entities={'suffix': 'T1w', 'session': '3T', 'acquisition': None})
