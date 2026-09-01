
"""Copy the 7T_rory data into a MELD-graph condition folder.

Usage:
    python -m prepare_data_meldgraph.prepare_7T_rory --meld_graph --B0_condition 3T
"""

from prepare_data.prepare_7T_rory import copy_7T_rory


def prepare_meld_graph(B0_condition):
    match B0_condition:
        case '7T':
            copy_7T_rory(dest_dir='data/meld_graph/7T_rory/7T/input',
                        bids_config_filename='meld_bids_config.json',
                        FLAIR=False,
                        B0='7T')

        case '7T_fastsurfer':
            copy_7T_rory(dest_dir='data/meld_graph/7T_rory/7T_fastsurfer/input',
                        bids_config_filename='meld_bids_config.json',
                        FLAIR=False,
                        B0='7T')

        case '7T_FLAIR':
            copy_7T_rory(dest_dir='data/meld_graph/7T_rory/7T_FLAIR/input',
                        bids_config_filename='meld_bids_config.json',
                        FLAIR=True,
                        B0='7T')

        case '7T_nouhf':
            copy_7T_rory(dest_dir='data/meld_graph/7T_rory/7T_nouhf/input',
                        bids_config_filename='meld_bids_config.json',
                        FLAIR=False,
                        B0='7T')

        case '3T':
            copy_7T_rory(dest_dir='data/meld_graph/7T_rory/3T/input',
                        bids_config_filename='meld_bids_config.json',
                        FLAIR=False,
                        B0='3T')

        case '3T_fastsurfer':
            copy_7T_rory(dest_dir='data/meld_graph/7T_rory/3T_fastsurfer/input',
                        bids_config_filename='meld_bids_config.json',
                        FLAIR=False,
                        B0='3T')

        case '3T_FLAIR':
            copy_7T_rory(dest_dir='data/meld_graph/7T_rory/3T_FLAIR/input',
                        bids_config_filename='meld_bids_config.json',
                        FLAIR=True,
                        B0='3T')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Prepare 7T Rory data for MELD-graph.")
    parser.add_argument("--meld_graph", default=False, action='store_true', help="If set, prepare data for meld_graph pipeline.")
    parser.add_argument("--B0_condition", type=str, required=True, help="B0 and condition, e.g.: 3T, 3T_FLAIR, 7T, 7T_nouhf")
    args = parser.parse_args()

    # if no arguments are provided, display help and exit
    if not args.meld_graph:
        parser.print_help()
        exit(1)

    if args.meld_graph:
        prepare_meld_graph(B0_condition=args.B0_condition)
