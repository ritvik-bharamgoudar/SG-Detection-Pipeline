import nibabel as nib
import pandas as pd

def load_nifti(file_path, return_affine=False):
    """
    Load a nifti file using nibabel.

    Parameters:
        file_path (str): Path to the .nii file.
        return_affine (bool): If True, return the affine matrix as well.

    Returns:
        np.ndarray or (np.ndarray, np.ndarray): Data array, optionally with affine matrix.
    """
    nii = nib.load(file_path)
    data = nii.get_fdata()
    if return_affine:
        return data, nii.affine
    return data


def load_all_data(mri_path, column_path, layer_path):
    """
    Load all nifti files needed for the SG detection pipeline.

    Parameters:
        mri_path (str): Path to the raw intensity MRI data
        column_path (str): Path to the column segmentation from LAYNII: LN2_COLUMNS.
        layer_path (str): Path to the cortical depths segmentation from LAYNII: LN2_LAYERS.

    Returns:
        tuple of np.ndarrays: (mri_data, column_data, cortical_depth)
    """
    mri_data = load_nifti(mri_path)
    column_data = load_nifti(column_path)
    cortical_depth = load_nifti(layer_path)
    
    return mri_data, column_data, cortical_depth

def load_manual_labels(label_file):
    """
    Load column labels and classification from a CSV file.

    Parameters:
        label_file (str): Path to the label CSV. Should contain 'col_id' and 'label' columns.

    Returns:
        dict: {col_id: label}

    Raises:
        ValueError: If duplicate column IDs found.
    """
    df = pd.read_csv(label_file)

    if df['col_id'].duplicated().any():
        duplicates = df[df['col_id'].duplicated(keep=False)]
        raise ValueError(f"Duplicate column IDs found in {label_file}:\n{duplicates}")

    return dict(zip(df['col_id'], df['label']))
