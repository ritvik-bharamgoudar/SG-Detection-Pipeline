import os
import numpy as np
import pandas as pd
from src.utils import load_all_data, get_column_ids, get_max_depth, load_manual_labels
from src.extract_profiles import extract_median_profile
from src.extract_feautres import compute_features
from src.classifier_training import (
    prepare_training_data,
    train_rf_model,
    train_xgb_model,
    predict_sg_probabilities,
)
from src.spatial_utils import get_column_adjacency, adjacency_to_dict, smooth_confidence_scores
from src.outputs import save_probability_map_to_nifti
from sklearn.preprocessing import StandardScaler

# Step 1: Define input file paths
# Change these paths to your data locations
# columns from LAYNII LN2_COLUMNS and layers from LAYNII LN2_LAYERS
mri_path = "../200um_01/200um_scoop_V1_02.nii.gz"
column_path = "../200um_01/Columns/20_05_columns300.nii"
layer_path = "../200um_01/Layers/50_05_layers_equivol.nii"
label_path = "manual_labels.csv"

# Step 2: Load data
mri_data, column_data, cortical_depth = load_all_data(mri_path, column_path, layer_path)
affine = load_all_data(column_path, return_affine=True)[1]  # Needed for saving

# Step 3: Extract intensity profiles and compute features
col_IDs = get_column_ids(column_data)
profiles = {
    col_id: extract_median_profile(col_id, column_data, cortical_depth, mri_data)
    for col_id in col_IDs
}
feature_df = compute_features(profiles)

# Step 4: Load labelled data and prepare training data
label_dict = load_manual_labels(label_path)
feature_columns = ['dip_width', 'dip_depth_diff', 'post_dip_ratio', 'second_derivative_max']
X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_training_data(
    feature_df, label_dict, features_to_use=feature_columns
)

# Step 5: Train classifier models
rf_model = train_rf_model(X_train_scaled, y_train)
xgb_model = train_xgb_model(X_train_scaled, y_train)

# Step 6: Predict SG probability on all columns (Random Forest specified here)
feature_df = predict_sg_probabilities(
    feature_df, rf_model, scaler, model_name="rf"
)

# Step 7: Save raw RF (or XGB if specified) probability map
save_probability_map_to_nifti(
    feature_df=feature_df,
    column_data=column_data,
    prob_col="rf_sg_probability",
    affine=affine,
    output_path="SG_RF_probability_map_raw.nii"
)

# Step 8: Apply smoothing using 3D column adjacency
adjacency_pairs = get_column_adjacency(column_path)
adj_dict = adjacency_to_dict(adjacency_pairs)
feature_df["rf_sg_probability_smoothed"] = smooth_confidence_scores(
    feature_df, adj_dict, "rf_sg_probability", alpha=0.6
)

# Step 9: Save smoothed RF probability map
save_probability_map_to_nifti(
    feature_df=feature_df,
    column_data=column_data,
    prob_col="rf_sg_probability_smoothed",
    affine=affine,
    output_path="SG_RF_smoothed_probability_map.nii"
)

print("Pipeline complete.")