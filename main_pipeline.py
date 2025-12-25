
from src.utils import load_all_data, load_manual_labels, load_nifti
from src.extract_profiles import extract_median_profile, get_column_ids
from src.extract_features import compute_features
from src.classifier_training import (
    prepare_training_data,
    train_rf_model,
    train_xgb_model,
    predict_sg_probabilities,
)
from src.spatial_utils import get_column_adjacency, adjacency_to_dict, smooth_confidence_scores
from src.outputs import save_column_map_to_nifti

# Step 1: Define input file paths
# Change paths to data locations
# columns from LAYNII LN2_COLUMNS and layers from LAYNII LN2_LAYERS
mri_path = "sample_data/Edlow_2019_200um_scoop_V1.nii.gz"
column_path = "sample_data/scoop_V1_columns_300.nii.gz"
layer_path = "sample_data/scoop_V1_layers_50_equivol.nii.gz"
label_path = "sample_data/manual_labels.csv"

# Step 2: Load data
mri_data, column_data, cortical_depth = load_all_data(mri_path, column_path, layer_path)
_, affine = load_nifti(column_path, return_affine=True) # match input output affine

# Step 3: Extract intensity profiles and compute features
col_IDs = get_column_ids(column_data)
profiles = {
    col_id: extract_median_profile(col_id, column_data, cortical_depth, mri_data)
    for col_id in col_IDs
}
feature_df = compute_features(profiles)

# Step 4: Load labelled data and prepare training data
label_dict = load_manual_labels(label_path, clean_duplicates=True, sort_by='col_id', save_cleaned=True)
feature_columns = ['dip_width', 'dip_depth_diff', 'post_dip_ratio', 'second_derivative_max']
X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_training_data(
    feature_df, label_dict, features_to_use=feature_columns
)

# Step 5: Train classifier models
rf_model = train_rf_model(X_train_scaled, y_train, save_model=True, model_path="outputs/test_rf_model.joblib", feature_names=feature_columns, scaler=scaler)
xgb_model = train_xgb_model(X_train_scaled, y_train, save_model=True, model_path="outputs/test_xgb_model.joblib", feature_names=feature_columns, scaler=scaler)

# Step 6: Predict SG probability on all columns (Random Forest specified here)
feature_df = predict_sg_probabilities(
    feature_df, rf_model, scaler, model_name="rf"
)

# Step 7: Save raw RF (or XGB if specified) probability map
save_column_map_to_nifti(
    feature_df=feature_df,
    column_data=column_data,
    value_col="rf_sg_probability",
    affine=affine,
    output_path="SG_RF_probability_map_raw_100.nii"
)

# Step 8: Apply smoothing using 3D column adjacency
adjacency_pairs = get_column_adjacency(column_path)
adj_dict = adjacency_to_dict(adjacency_pairs)
feature_df["rf_sg_probability_smoothed"] = smooth_confidence_scores(
    feature_df, adj_dict, "rf_sg_probability", alpha=0.6
)

# Step 9: Save smoothed RF probability map
save_column_map_to_nifti(
    feature_df=feature_df,
    column_data=column_data,
    value_col="rf_sg_probability_smoothed",
    affine=affine,
    output_path="SG_RF_smoothed_probability_map_100.nii"
)

# Save final feature_df after all processing steps
output_csv_path = "outputs/morelab_feature_df_full.csv"
feature_df.to_csv(output_csv_path, index=False)
print(f"Saved final feature_df to {output_csv_path}")


print("Pipeline complete.")