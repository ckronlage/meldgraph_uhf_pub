
"""Copy the bonn_7T data into a MELD-graph condition folder.

Needs the session/file selection at data/raw/bonn_7T/bids/derivatives/3T_7T_MELD_paths.csv,
written by `python -m prepare_data.prepare_bonn_7T --select_sessions_and_files`.

Usage:
    python -m prepare_data_meldgraph.prepare_bonn_7T --meld_graph --B0_condition 3T
"""

from prepare_data.prepare_bonn_7T import copy_bonn_7T


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Prepare bonn_7T data for MELD-graph.")
    parser.add_argument("--meld_graph", default=False, action='store_true', help="If set, prepare data for meld_graph pipeline.")
    parser.add_argument("--B0_condition", type=str, required=False, help="If set, only prepare data for the specified B0 condition (3T or 7T) for meld_graph pipeline.")
    args = parser.parse_args()

    # if no arguments are provided, display help and exit
    if not args.meld_graph:
        parser.print_help()
        exit(1)

    if args.meld_graph:
        match args.B0_condition:
            case '7T':
                copy_bonn_7T(dest_dir='data/meld_graph/bonn_7T/7T/input',
                             bids_config_filename='meld_bids_config.json',
                             FLAIR=False,
                             B0='7T')

            case '7T_fastsurfer':
                copy_bonn_7T(dest_dir='data/meld_graph/bonn_7T/7T_fastsurfer/input',
                            bids_config_filename='meld_bids_config.json',
                            FLAIR=False,
                            B0='7T')

            case '7T_nouhf':
                copy_bonn_7T(dest_dir='data/meld_graph/bonn_7T/7T_nouhf/input',
                            bids_config_filename='meld_bids_config.json',
                            FLAIR=False,
                            B0='7T')

            case '7T_FLAIR':
                copy_bonn_7T(dest_dir='data/meld_graph/bonn_7T/7T_FLAIR/input',
                            bids_config_filename='meld_bids_config.json',
                            FLAIR=True,
                            B0='7T')

            case '3T':
                copy_bonn_7T(dest_dir='data/meld_graph/bonn_7T/3T/input',
                             bids_config_filename='meld_bids_config.json',
                            FLAIR=False,
                            B0='3T')

            case '3T_fastsurfer':
                copy_bonn_7T(dest_dir='data/meld_graph/bonn_7T/3T_fastsurfer/input',
                            bids_config_filename='meld_bids_config.json',
                            FLAIR=False,
                            B0='3T')

            case '3T_FLAIR':
                copy_bonn_7T(dest_dir='data/meld_graph/bonn_7T/3T_FLAIR/input',
                            bids_config_filename='meld_bids_config.json',
                            FLAIR=True,
                            B0='3T')
