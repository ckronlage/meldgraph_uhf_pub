
import re
import os
import glob
import difflib
import shutil
import subprocess
import tempfile
import multiprocessing

import numpy as np
import pandas as pd
import nibabel as nib
import nilearn
import nilearn.plotting


def coreg_per_subj(bids_dir, path_dict, threads=1):
    # target filename is the same as the meld input nifti, but with the _<suffix> 
    # appended as _space-suffix> and _lesionmask appended as suffix
    target_filename = os.path.basename(path_dict['meld_input_nifti'])
    target_filename = re.sub(r'_([^_]+?)\.nii.*', r'_space-\1_lesionmask.nii.gz', target_filename)
    target_dir = f'{bids_dir}/derivatives/lesion_masks_coreg/'
    target_path = os.path.join(target_dir, target_filename)
    print(f"Coregistering lesion mask {path_dict['lesion_mask_nifti']} to {path_dict['meld_input_nifti']} and saving to {target_path}")
    os.makedirs(target_dir, exist_ok=True)

    # dir for QC plots
    qc_dir = f'{bids_dir}/derivatives/lesion_masks_coreg_qc/'
    os.makedirs(qc_dir, exist_ok=True)

    # if meld_input_nifti is the same as anat_nifti, we just just be able to copy
    # but to be safe, we will still check the voxel dimensions and affine are the same
    # if not - coregister
    if path_dict['meld_input_nifti'] == path_dict['anat_nifti']:
        lesion_mask_img = nib.load(path_dict['lesion_mask_nifti'])
        meld_input_img = nib.load(path_dict['meld_input_nifti'])
        # remove possible extra dimensions (e.g. from 4D nifti with single timepoint)
        lesion_mask_img = nib.squeeze_image(lesion_mask_img)
        meld_input_img = nib.squeeze_image(meld_input_img)
        if (lesion_mask_img.shape == meld_input_img.shape) and np.allclose(lesion_mask_img.affine, meld_input_img.affine):
            print(f"Lesion mask {path_dict['lesion_mask_nifti']} is already in the same space as {path_dict['meld_input_nifti']}. Copying file.")
            nib.save(lesion_mask_img, target_path)
            return
    
    if not os.path.exists(target_path): 
        plot_qc_path = f'{qc_dir}/{os.path.basename(path_dict["meld_input_nifti"]).replace(".nii.gz", "_coreg_qc.png")}'
        coreg_and_align_easyreg(moving_path=path_dict['anat_nifti'],
                                target_path=path_dict['meld_input_nifti'],
                                mask_input_path=path_dict['lesion_mask_nifti'],
                                mask_output_path=target_path,
                                threads=threads,
                                plot_qc_path=plot_qc_path)
        #coreg_and_align_mri_coreg(moving_path=path_dict['anat_nifti'], 
        #                target_path=path_dict['meld_input_nifti'],
        #                mask_input_path=path_dict['lesion_mask_nifti'],
        #                mask_output_path=target_path,
        #                threads=threads,
        #                plot_qc_path=plot_qc_path)
    
    # make plots for QC - overlay the original lesion mask on the original nifti,
    # and then the coregistered lesion mask on the target nifti, and save these 
    # as pngs in the derivatives/lesion_masks_coreg_qc/ directory with the same filename as the target nifti but with .png extension
    qc_orig_path = f'{qc_dir}/{os.path.basename(path_dict["anat_nifti"]).replace(".nii.gz", "_orig_lesionoverlay.png")}'
    if not os.path.isfile(qc_orig_path): # might already exist
        anat_nifti_img = nib.load(path_dict['anat_nifti'])
        lesion_mask_img = nib.load(path_dict['lesion_mask_nifti'])  
        cut_coords = nilearn.plotting.find_xyz_cut_coords(lesion_mask_img)
        plot_orig = nilearn.plotting.plot_anat(anat_img=anat_nifti_img, 
                                            title='Original Lesion Mask',
                                            radiological=True,
                                            cut_coords=cut_coords,
                                            vmin=np.percentile(anat_nifti_img.get_fdata(),1),
                                            vmax=np.percentile(anat_nifti_img.get_fdata(),99))
        plot_orig.add_overlay(lesion_mask_img, threshold=0.5)
        plot_orig.savefig(qc_orig_path)
        plot_orig.close()

    qc_coreg_path = f'{qc_dir}/{os.path.basename(path_dict["meld_input_nifti"]).replace(".nii.gz", "_lesionoverlay.png")}'
    if not os.path.isfile(qc_coreg_path):
        meld_input_img = nib.load(path_dict['meld_input_nifti'])
        target_lesionmask_img = nib.load(target_path)
        cut_coords = nilearn.plotting.find_xyz_cut_coords(target_lesionmask_img)
        plot_coreg = nilearn.plotting.plot_anat(anat_img=meld_input_img,
                                            title='Coregistered Lesion Mask',
                                            radiological=True,
                                            cut_coords=cut_coords,
                                            vmin=np.percentile(meld_input_img.get_fdata(),1),
                                            vmax=np.percentile(meld_input_img.get_fdata(),99))
        plot_coreg.add_overlay(target_lesionmask_img, threshold=0.5)
        plot_coreg.savefig(qc_coreg_path)
        plot_coreg.close()

def coreg_and_align_easyreg(moving_path, target_path, mask_input_path, mask_output_path, threads=1, plot_qc_path=None):
    with tempfile.TemporaryDirectory() as workdir:
        cmd_fs_setup = f'export APPTAINERENV_FS_LICENSE=/license.txt && export APPTAINER_BINDPATH=license.txt:/license.txt:ro && '
        cmd_fs_setup += f'export APPTAINER_BINDPATH={moving_path}:/moving_img.nii.gz:ro,$APPTAINER_BINDPATH && '
        cmd_fs_setup += f'export APPTAINER_BINDPATH={target_path}:/target_img.nii.gz:ro,$APPTAINER_BINDPATH && '
        cmd_fs_setup += f'export APPTAINER_BINDPATH={mask_input_path}:/mask_input.nii.gz:ro,$APPTAINER_BINDPATH && '
        cmd_fs_setup += f'export APPTAINER_BINDPATH={workdir}:/workdir,$APPTAINER_BINDPATH && '

        # skullstrip moving
        cmd = f'apptainer exec --containall freesurfer_8.1.0.sif /bin/bash -c "mri_synthstrip -i /moving_img.nii.gz -o /workdir/moving_img_strip.nii.gz"'
        result = subprocess.run(cmd_fs_setup + cmd, shell=True, executable='/bin/bash')

        # skullstrip target
        cmd = f'apptainer exec --containall freesurfer_8.1.0.sif /bin/bash -c "mri_synthstrip -i /target_img.nii.gz -o /workdir/target_img_strip.nii.gz"'
        result = subprocess.run(cmd_fs_setup + cmd, shell=True, executable='/bin/bash')

        # easyreg on skullstripped images, affine only
        cmd = f'apptainer exec --containall freesurfer_8.1.0.sif /bin/bash -c "mri_easyreg --flo /workdir/moving_img_strip.nii.gz --ref /workdir/target_img_strip.nii.gz --fwd_field /workdir/fwd_field.nii.gz --flo_seg /workdir/flo_seg.nii.gz --ref_seg /workdir/ref_seg.nii.gz --threads {threads} --affine_only"'
        result = subprocess.run(cmd_fs_setup + cmd, shell=True, executable='/bin/bash')

        if plot_qc_path is not None:
            # apply to non-skullstripped moving image (only for QC plots)
            cmd = f'apptainer exec --containall freesurfer_8.1.0.sif /bin/bash -c "mri_easywarp --i /moving_img.nii.gz --o /workdir/moving_reg.nii.gz --field /workdir/fwd_field.nii.gz --threads {threads} --nearest"'
            result = subprocess.run(cmd_fs_setup + cmd, shell=True, executable='/bin/bash')

        # apply to lesion mask
        cmd = f'apptainer exec --containall freesurfer_8.1.0.sif /bin/bash -c "mri_easywarp --i /mask_input.nii.gz --o /workdir/mask_output.nii.gz --field /workdir/fwd_field.nii.gz --threads {threads} --nearest"'
        result = subprocess.run(cmd_fs_setup + cmd, shell=True, executable='/bin/bash')

        # copy to target path
        shutil.copy2(os.path.join(workdir, 'mask_output.nii.gz'), mask_output_path)

        # QC plot if specified - overlay edge filtered moving image on target 
        if plot_qc_path is not None:
            os.makedirs(os.path.dirname(plot_qc_path), exist_ok=True)
            target_img = nib.squeeze_image(nib.load(target_path))
            display = nilearn.plotting.plot_anat(target_img,
                                                 title='Target Image',
                                                 display_mode='ortho',
                                                 draw_cross=False,
                                                 radiological=True,
                                                 vmin=np.percentile(target_img.get_fdata(),1),
                                                 vmax=np.percentile(target_img.get_fdata(),99))
            moving_img_reg = nib.squeeze_image(nib.load(f'{workdir}/moving_reg.nii.gz'))
            display.add_edges(moving_img_reg)
            display.savefig(plot_qc_path)
            display.close()


def get_meld_input_niftis(bids_dir):
    anat_niftis = glob.glob(f"{bids_dir}/sub-*/**/anat/*.nii*", recursive=True)
    if os.path.isfile(f"{bids_dir}/derivatives/3T_7T_MELD_paths.csv"):
        meld_paths_df = pd.read_csv(f"{bids_dir}/derivatives/3T_7T_MELD_paths.csv")
        meld_input_niftis = meld_paths_df['T1w_7T'].dropna().tolist() + meld_paths_df['T1w_3T'].dropna().tolist()
    else:
        meld_input_niftis = [path for path in anat_niftis if 'T1w' in path or 'UNIT1' in path]

    subj_ids = [re.sub(r'.*(sub-.+?)_.*', r'\1', path) for path in meld_input_niftis]
    meld_input_niftis_df = pd.DataFrame({'subj_id': subj_ids,
                                    'meld_input_nifti': meld_input_niftis})
    return meld_input_niftis_df

def coreg_lesion_masks(bids_dir, threads=1):
    # 1) Pair each lesion mask with the anatomical nifti it is in register with
    # first get paths of all anatomical niftis
    anat_niftis = glob.glob(f"{bids_dir}/sub-*/**/anat/*.nii*", recursive=True)
    subj_ids = [re.sub(r'.*(sub-.+?)_.*', r'\1', path) for path in anat_niftis]
    anat_niftis_df = pd.DataFrame({'subj_id': subj_ids,
                                'anat_nifti': anat_niftis})


    lesion_mask_niftis = glob.glob(f"{bids_dir}/derivatives/lesion_masks/**/*.nii*", recursive=True)
    subj_ids = [re.sub(r'.*(sub-.+?)_.*', r'\1', path) for path in lesion_mask_niftis]
    lesion_mask_niftis_df = pd.DataFrame({'subj_id': subj_ids,
                                        'lesion_mask_nifti': lesion_mask_niftis})

    # Now for each lesion mask, find the corresponding one among the subject's 
    # nifti files, searching for the maximum string match between the filenames. 
    # Typically, this might be: 
    #   lesionmask: sub-005_ses-7T_acq-pTx_space-UNIT1_lesionmask.nii.gz
    #   anat-nifti: sub-005_ses-7T_acq-pTx_UNIT1.nii.gz
    # A little hacky without proper bids metadata, but works given proper 
    # naming conventions.

    lesion_mask_pairs = []
    for subj_id, lesion_mask_nifti in lesion_mask_niftis_df.itertuples(index=False):
        l_basename = os.path.basename(lesion_mask_nifti)
        best_match = None
        best_ratio = 0.0
        for a in anat_niftis_df[anat_niftis_df['subj_id'] == subj_id]['anat_nifti']:
            a_basename =  os.path.basename(a)
            ratio = difflib.SequenceMatcher(None, l_basename, a_basename).ratio()
            #print(f"Comparing lesion mask {l_basename} with anat nifti {a_basename}: ratio {ratio}")
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = a
        print(f"Lesion mask: {lesion_mask_nifti}")
        print(f"Best match:  {best_match} (ratio: {best_ratio})")
        print("")
        lesion_mask_pairs.append({'subj_id': subj_id,
                                'lesion_mask_nifti': lesion_mask_nifti,
                                'anat_nifti': best_match})

    lesion_mask_pairs_df = pd.DataFrame(lesion_mask_pairs)


    # 2) Now select all the T1w and UNIT1 anat files that were used as MELD inputs
    # (we need to coregister the masks to these)
    # if there is a 'derivatives/3T_7T_MELD_paths.csv', use that
    # otherwise use all _T1w and _UNIT1 files
    meld_input_niftis_df = get_meld_input_niftis(bids_dir)

    # 3) Match the MELD input files to lesion mask pairs by subject id
    coreg_pairs_df = pd.merge(lesion_mask_pairs_df, meld_input_niftis_df, on='subj_id', how='inner')
    # ... and then coregister the lesion masks to the MELD input niftis
    #rows = coreg_pairs_df.to_dict(orient='records')
    #with multiprocessing.Pool(processes=threads) as pool:
    #    pool.starmap(coreg_per_subj, [(bids_dir, row, 1) for row in rows])

    # non-parallel version for debugging
    for i, row in coreg_pairs_df.iterrows():
       coreg_per_subj(bids_dir, row, threads=threads)
        

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Coregister lesion masks to MELD input niftis')
    parser.add_argument('--bids_dir', type=str, required=True, help='Path to the BIDS directory containing the data')
    parser.add_argument('--coreg', action='store_true', help='Coregister lesion masks in /derivatives/lesion_masks/ to the MELD input niftis and save in /derivatives/lesion_masks_coreg/')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads to use for coregistration')
    args = parser.parse_args()  

    if args.coreg:
        coreg_lesion_masks(bids_dir=args.bids_dir,
                        threads=args.threads)


if __name__ == "__main__":
    main()
    #coreg_lesion_masks(bids_dir="data/raw/bonn_7T/bids", threads=4)