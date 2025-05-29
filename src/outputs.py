import nibabel as nib
import numpy as np

def save_probability_map_to_nifti(feature_df, column_data, prob_col, affine, output_path):
    """
    Save a probability column from feature_df to a nifit file.
    
    Parameters:
        feature_df: DataFrame with 'col_id' and one probability column
        column_data: 3D array with column IDs
        prob_col: name of the probability column (ie. 'rf_sg_probability')
        affine: affine from a reference nifit (used for spatial orientation)
        output_path: path to save the output nifti file
    """
    prob_map = np.zeros_like(column_data)
    # Store each columns predicted probability of SG from feature_df
    for _, row in feature_df.iterrows():
        col_id = row['col_id']
        prob = row[prob_col]
        prob_map[column_data == col_id] = prob

    # Save nifit
    nifti_img = nib.Nifti1Image(prob_map, affine)
    nib.save(nifti_img, output_path)
    print(f"Saved probability map to: {output_path}")