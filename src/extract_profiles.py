import numpy as np

def get_column_ids(column_data):
    """
    Extract all unique non-zero column IDs from the column segmentation.
    Should be number of columns from LAYNII LN2_COLUMNS

    Parameters:
        column_data (np.ndarray): 3D array of column labels.

    Returns:
        np.ndarray: Array of unique column IDs.
    """
    return np.unique(column_data[column_data > 0])

def get_max_depth(depths):
    """
    Get the maximum cortical depth value from the layers file.

    Parameters:
        depth_map (np.ndarray): 3D array of cortical depth values.

    Returns:
        int: Maximum depth value found in the volume.
    """
    return int(np.nanmax(depths))

def extract_median_profile(col_id, column_data, cortical_depth, mri_data):
    """
    Extract the median intensity profile across cortical depth for a given column.

    Parameters:
        col_id (int): Column ID to extract.
        column_data (np.ndarray): 3D array with column labels.
        cortical_depth (np.ndarray): 3D array with layer depth values.
        mri_data (np.ndarray): 3D array with intensity values.

    Returns:
        tuple: (depths, median_intensities)
            - depths (np.ndarray): Unique depth values normally number of layers from LAYNII LN2_LAYERS
            - median_intensities (np.ndarray): Median intensity per depth
    """
    column_mask = (column_data == col_id)

    depth_values = cortical_depth[column_mask]
    intensity_values = mri_data[column_mask]

    valid_mask = ~np.isnan(depth_values)
    depth_values = depth_values[valid_mask]
    intensity_values = intensity_values[valid_mask]

    unique_depths = np.unique(depth_values)
    median_profile = [np.median(intensity_values[depth_values == d]) for d in unique_depths]

    return unique_depths, np.array(median_profile)

