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

'''
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
'''

def load_manual_labels(label_file, clean_duplicates=False, sort_by=None, save_cleaned=False):
    """
    Load column labels and classification from a CSV file.
    Optionally clean duplicates, sort, and save back to file.

    Parameters:
        label_file (str): Path to the manual labels CSV. Should contain 'col_id' and 'label' columns.
        clean_duplicates (bool): If True, remove duplicates. Default False.
        sort_by (str): Sort by 'col_id', 'label', or None. Default None.
        save_cleaned (bool): If True, save cleaned/sorted data back to file. Default False.

    Returns:
        dict: {col_id: label}

    Raises:
        ValueError: If duplicate column IDs found and clean_duplicates=False.
    """
    df = pd.read_csv(label_file)
    original_length = len(df)
    changes_made = False
    
    # Handle duplicates
    if df['col_id'].duplicated().any():
        duplicates = df[df['col_id'].duplicated(keep=False)]
        
        if clean_duplicates:
            print(f"Found {len(duplicates)} duplicate entries:")
            print(duplicates[['col_id', 'label']].to_string(index=False))
            
            # Keep first occurrence of each col_id
            df = df.drop_duplicates(subset=['col_id'], keep='first')
            removed_count = original_length - len(df)
            print(f"Removed {removed_count} duplicate entries, keeping first occurrence")
            changes_made = True
        else:
            raise ValueError(f"Duplicate column IDs found in {label_file}:\n{duplicates}")
    
    # Handle sorting
    if sort_by:
        if sort_by == 'col_id':
            df = df.sort_values('col_id')
            print(f"Sorted by column ID")
            changes_made = True
        elif sort_by == 'label':
            # Sort by label first (0s then 1s), then by col_id within each group
            df = df.sort_values(['label', 'col_id'])
            print(f"Sorted by label (0s first, then 1s), then by column ID")
            changes_made = True
        else:
            print(f"Unknown sort option '{sort_by}'. Valid options: 'col_id', 'label'")
    
    # Save back to file if changes were made and requested
    if changes_made and save_cleaned:
        df.to_csv(label_file, index=False)
        print(f"Saved cleaned data back to {label_file}")
        print(f"Final dataset: {len(df)} entries")
    
    return dict(zip(df['col_id'], df['label']))
