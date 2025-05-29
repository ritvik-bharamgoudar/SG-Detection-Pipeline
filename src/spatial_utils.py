import numpy as np
import nibabel as nib
import pandas as pd

def get_column_adjacency(column_file):
    """
    Identify adjacency pairs between column labels in a 3D nifti volume.

    Parameters:
        column_file (str): Path to columns file.

    Returns:
        set: Set of (col_A, col_B) tuples, where each pair indicates adjacency.
    """
    data = nib.load(column_file).get_fdata().astype(np.int32)
    adjacency = set()

    # Check only positive directions to avoid duplicate pairs
    directions = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

    for dx, dy, dz in directions:
        a = data[:-dx or None, :-dy or None, :-dz or None]
        b = data[dx:, dy:, dz:]

        # Find boundary voxels with differing, non-zero labels
        mask = (a != b) & (a != 0) & (b != 0)
        a_vals = a[mask]
        b_vals = b[mask]

        for val1, val2 in zip(a_vals, b_vals):
            adjacency.add(tuple(sorted((val1, val2))))

    return adjacency

def adjacency_to_dict(adjacency_pairs):
    """
    Convert adjacency pairs into a neighbor dictionary.
    
    Parameters:
        adjacency_pairs (set): Set of (a, b) column ID tuples.

    Returns:
        dict: {col_id: sorted list of neighbor col_ids}
    """
    from collections import defaultdict

    neighbors = defaultdict(set)

    for a, b in adjacency_pairs:
        neighbors[a].add(b)
        neighbors[b].add(a)

    # Convert sets to sorted lists
    return {col: sorted(list(neighbs)) for col, neighbs in neighbors.items()}

def smooth_confidence_scores(feature_df, adjacency_dict, conf_column, alpha=0.6):
    """
    Apply smoothing to SG confidence scores using neighbouring columns.

    Parameters:
        feature_df (pd.DataFrame): Must contain 'col_id' and confidence column from classifier ('rf_sg_probability')
        adjacency_dict (dict): {col_id: [neighbor_ids]}
        conf_column (str): Name of the confidence column to smooth
        alpha (float): Weight for the column’s own confidence (default: 0.6)

    Returns:
        pd.Series: Smoothed confidence values (same length as feature_df)
    """
    feature_df = feature_df.copy()
    col_conf_map = dict(zip(feature_df['col_id'], feature_df[conf_column]))

    smoothed_scores = []

    for col_id in feature_df['col_id']:
        own_conf = col_conf_map[col_id]
        neighbors = adjacency_dict.get(col_id, [])

        # Get valid neighbor confidences
        neighbor_confs = [col_conf_map[n] for n in neighbors if n in col_conf_map]

        if neighbor_confs:
            neighbor_avg = np.mean(neighbor_confs)
        else:
            neighbor_avg = own_conf  # fallback if no neighbors

        smoothed = alpha * own_conf + (1 - alpha) * neighbor_avg
        smoothed_scores.append(smoothed)

    return pd.Series(smoothed_scores, index=feature_df.index)

if __name__ == "__main__":
    
        column_file = "/Users/Ritvik/Desktop/EN3100/200um_01/Columns/20_05_columns300.nii"  # Update with actual path
        adjacency_pairs = get_column_adjacency(column_file)
        adjacency_dict = adjacency_to_dict(adjacency_pairs)
        print(adjacency_dict)
    
