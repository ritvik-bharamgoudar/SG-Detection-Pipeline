import nibabel as nib
import numpy as np

def save_column_map_to_nifti(feature_df, column_data, value_col, affine, output_path):
    """
    Save any per-column scalar value from feature_df to a NIfTI file.

    Parameters:
        feature_df (DataFrame): Must contain 'col_id' and value_col
        column_data (np.ndarray): 3D volume with column IDs
        value_col (str): Column in feature_df to save as map
        affine (np.ndarray): Affine matrix from reference NIfTI
        output_path (str): File path to save output
    """
    out_map = np.zeros_like(column_data)

    for _, row in feature_df.iterrows():
        col_id = row['col_id']
        val = row[value_col]
        out_map[column_data == col_id] = val

    nib.save(nib.Nifti1Image(out_map, affine), output_path)
    print(f"Saved column map '{value_col}' to: {output_path}")