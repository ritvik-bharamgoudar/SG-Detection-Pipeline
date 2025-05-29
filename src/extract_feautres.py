import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

def extract_regression_points(depths, intensities, lower_pct=0.15, upper_pct=0.85, edge_pct=0.1):
    """
    Fit a linear regression trendline to the ends of the central part of the profile.

    Parameters:
        depths (np.ndarray): Cortical depth values for the profile.
        intensities (np.ndarray): Intensity values along the profile.
        lower_pct (float): Start percentile of the depth range to include (default 0.15).
        upper_pct (float): End percentile of the depth range to include (default 0.85).
        edge_pct (float): Proportion of points from both ends to use in regression (default 0.1).

    Returns:
        tuple: (
            depths_filtered (np.ndarray),     # depths in central region
            intensities_filtered (np.ndarray),# corresponding intensities
            regression_depths (np.ndarray),   # x-values used to fit trendline
            regression_intensities (np.ndarray), # y-values used to fit trendline
            predicted_intensities (np.ndarray),  # trendline across filtered depths
            profile_slope (float)                 # slope of trendline
        )
    """
    total_points = len(depths)
    
    start_depth = depths[int(lower_pct * total_points)]
    end_depth = depths[int(upper_pct * total_points)]
    
    mid_mask = (depths >= start_depth) & (depths <= end_depth)
    depths_filtered = depths[mid_mask]
    intensities_filtered = intensities[mid_mask]

    sample_size = max(1, int(np.ceil(edge_pct * len(depths_filtered))))
    
    regression_depths = np.concatenate((depths_filtered[:sample_size], depths_filtered[-sample_size:]))
    regression_intensities = np.concatenate((intensities_filtered[:sample_size], intensities_filtered[-sample_size:]))

    model = LinearRegression()
    model.fit(regression_depths.reshape(-1, 1), regression_intensities)
    predicted_intensities = model.predict(depths_filtered.reshape(-1, 1))

    profile_slope = model.coef_[0]

    return (
        depths_filtered,
        intensities_filtered,
        regression_depths,
        regression_intensities,
        predicted_intensities,
        profile_slope
    )

def compute_dip_width(depths, intensities, trendline, threshold_factor=0.02):
    """
    Compute the width of the largest consecutive dip region below the trendline.

    A dip is defined as a stretch of points where the intensity is
    below (trendline * threshold). Only the longest such region is returned.

    Parameters:
        depths (np.ndarray): Depth values (must match intensity order).
        intensities (np.ndarray): Intensity values at each depth.
        trendline (np.ndarray): Trendline values (same length as depths).
        threshold_factor (float): Proportional threshold below trendline to 
        define a dip (default: 0.02) needs to be empricially tuned for the data.

    Returns:
        tuple: (
            max_dip_width (int): Length of the longest dip,
            best_dip_depths (np.ndarray): Depth values in longest dip region,
            dip_intensities (np.ndarray): Intensities in that region
        )
    """
    threshold_values = threshold_factor * trendline
    dip_mask = intensities < (trendline - threshold_values)

    dip_width = 0
    max_dip_width = 0
    temp_depths = []
    best_dip_depths = []

    for i, below in enumerate(dip_mask):
        if below:
            dip_width += 1
            temp_depths.append(depths[i])
        else:
            if dip_width > max_dip_width:
                max_dip_width = dip_width
                best_dip_depths = temp_depths.copy()
            dip_width = 0
            temp_depths = []

    if dip_width > max_dip_width:
        max_dip_width = dip_width
        best_dip_depths = temp_depths.copy()

    dip_intensities = [intensities[np.where(depths == d)[0][0]] for d in best_dip_depths] if best_dip_depths else []

    return max_dip_width, np.array(best_dip_depths), np.array(dip_intensities)


def compute_dip_depth(depths, intensities, trendline, threshold_factor=0.02):
    """
    Identify the depth of maximum deviation below the trendline within the dip region.

    Parameters:
        depths (np.ndarray): Depth values (filtered).
        intensities (np.ndarray): Intensity values.
        trendline (np.ndarray): Trendline over the same depths.
        threshold_factor (float): Same factor used to define dip region.

    Returns:
        tuple: (
            dip_depth (float): Depth of maximum dip,
            dip_depth_diff (float): Magnitude of deviation at that point
        )
    """
    # Find dip region
    dip_width, dip_depths, _ = compute_dip_width(depths, intensities, trendline, threshold_factor)

    if dip_width < 2 or len(dip_depths) == 0:
        return None, None

    # Calculate deviations from trendline
    deviations = np.abs(intensities - trendline)

    # Mask to keep only the dip region
    dip_region_mask = np.isin(depths, dip_depths)
    deviations_in_dip = deviations[dip_region_mask]

    if len(deviations_in_dip) == 0:
        return None, None

    # Find deepest point in the dip
    max_idx = np.argmax(deviations_in_dip)
    dip_depth = dip_depths[max_idx]
    dip_depth_diff = deviations_in_dip[max_idx]

    return dip_depth, dip_depth_diff

def compute_post_dip_regression_slope(depths, intensities, dip_depth, num_depths=4):
    """
    Compute slope of intensity values just after the dip.

    Parameters:
        depths (np.ndarray): Depth values in filtered profile.
        intensities (np.ndarray): Corresponding intensities.
        dip_depth (float or int): Depth at which the dip occurs.
        num_depths (int): Number of depths to include after the dip (default: 4)
        Should be empirically tuned for the data.

    Returns:
        float or None: Slope of post-dip segment, or None if invalid
    """
    if dip_depth is None:
        return None

    # Find index of dip_depth
    dip_idx = np.where(depths == dip_depth)[0]

    if len(dip_idx) == 0:
        return None  # dip depth not found

    dip_idx = dip_idx[0]

    # Make sure there are enough points after the dip
    if dip_idx + num_depths >= len(depths):
        return None

    local_depths = depths[dip_idx : dip_idx + num_depths + 1].reshape(-1, 1)
    local_intensities = intensities[dip_idx : dip_idx + num_depths + 1]

    model = LinearRegression()
    model.fit(local_depths, local_intensities)

    return model.coef_[0]  # Slope

def compute_second_derivative(depths, intensities, dip_width, dip_depths, smoothing_window=3, buffer=0.1):
    """
    Compute maximum second derivative within +/- buffer region around the dip. Buffer
    to ensure full dip region curvature is captured.

    Parameters:
        depths (np.ndarray): Depth values.
        intensities (np.ndarray): Intensity values.
        dip_width (int): Width of the dip.
        dip_depths (np.ndarray): Depths of the dip region.
        smoothing_window (int): Size of moving average window (default: 3).
        buffer (float): Fractional margin around dip region (default: 0.1).

    Returns:
        tuple:
            max_second_derivative (float or None): max curvature in dip region
            second_derivative (np.ndarray): Full profile's second derivative (smoothed)
    """
    # Compute first and second derivative
    first_derivative = np.gradient(intensities, depths)
    second_derivative = np.gradient(first_derivative, depths)

    # Smooth to mitigate noise amplification
    if smoothing_window > 1:
        second_derivative = np.convolve(second_derivative, np.ones(smoothing_window)/smoothing_window, mode='same')

    if dip_width < 2 or len(dip_depths) == 0:
        return None, second_derivative

    dip_start = np.min(dip_depths)
    dip_end = np.max(dip_depths)
    range_start = dip_start - (buffer * dip_width)
    range_end = dip_end + (buffer * dip_width)

    dip_mask = (depths >= range_start) & (depths <= range_end)
    second_deriv_in_dip = second_derivative[dip_mask]

    if len(second_deriv_in_dip) == 0:
        return None, second_derivative

    max_curvature = np.max(second_deriv_in_dip)

    return max_curvature, second_derivative

def compute_features(profiles, threshold_factor=0.02):
    """
    Compute features for each column based on its intensity profile.

    Parameters:
        profiles (dict): {col_id: (depths, intensities)}
        threshold_factor (float): Proportional threshold below trendline to 
        define a dip (default: 0.02), needs to be empricially tuned for the data.

    Returns:
        pd.DataFrame: Table with features for each column
    """
    feature_list = []

    for col_id, (depths, intensities) in profiles.items():
        if len(depths) == 0 or len(intensities) == 0:
            continue  # Skip empty profiles

        # Trendline and profile slope
        try:
            d_filt, i_filt, _, _, trendline, profile_slope = extract_regression_points(depths, intensities)
        except Exception as e:
            print(f"Skipping column {col_id} due to regression error:", e)
            continue

        # Dip region and dip deviation metrics
        dip_width, dip_depths, _ = compute_dip_width(d_filt, i_filt, trendline, threshold_factor)
        dip_depth, dip_depth_diff = compute_dip_depth(d_filt, i_filt, trendline, threshold_factor)

        # Post-dip regression slope
        post_dip_slope = compute_post_dip_regression_slope(d_filt, i_filt, dip_depth)

        # Ratio of post-dip slope to trendline slope
        post_dip_ratio = post_dip_slope / profile_slope if (profile_slope != 0 and post_dip_slope is not None) else 0

        # Second derivative across dip region
        max_second_derivative, _ = compute_second_derivative(d_filt, i_filt, dip_width, dip_depths)

        # Store features
        feature_list.append({
            "col_id": col_id,
            "dip_width": dip_width,
            "dip_depth_diff": dip_depth_diff,
            "post_dip_ratio": post_dip_ratio,
            "second_derivative_max": max_second_derivative
        })

    return pd.DataFrame(feature_list)
