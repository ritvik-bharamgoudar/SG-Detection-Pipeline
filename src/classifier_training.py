import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

def prepare_training_data(feature_df, label_dict, features_to_use=None, test_size=0.4, random_state=42):
    """
    Prepare training and test data with standard scaling.

    Parameters:
        feature_df (pd.DataFrame): DataFrame with column features
        label_dict (dict): {col_id: label}
        features_to_use (list or None): Which columns to use as features. Default None, use all except col_id/label.
        test_size (float): Proportion of data to reserve for test
        random_state (int): Seed for reproducibility

    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """
    # Add label column
    df = feature_df.copy()
    df['label'] = df['col_id'].map(label_dict).fillna(-1)
    labeled_data = df[df['label'] != -1]

    if features_to_use is None:
        features_to_use = [col for col in df.columns if col not in ['col_id', 'label']]

    X = labeled_data[features_to_use]
    y = labeled_data['label']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_rf_model(X_train, y_train, n_estimators=500, max_depth=3, min_samples_split=3, class_weight="balanced", 
                   random_state=42, save_model=False, model_path=None, feature_names=None, scaler=None):
    """
    Train a Random Forest classifier with specified hyperparameters.

    Parameters:
        X_train (np.ndarray): Scaled feature matrix for training
        y_train (array-like): Binary labels for training
        n_estimators (int, optional): Number of trees in the forest (default: 500)
        max_depth (int, optional): Maximum depth of each tree (default: 3)
        min_samples_split (int, optional): Minimum samples to split a node (default: 3)
        class_weight (str or dict, optional): Class balancing method (default: 'balanced')
        random_state (int, optional): Random seed for reproducibility (default: 42)
        save_model (bool, optional): Whether to save the trained model (default: False)
        model_path (str, optional): Path to save the model if save_model is True
        feature_names (list, optional): Names of the features used for training
        scaler (StandardScaler, optional): Fitted scaler used to scale features

    Returns:
        rf_model: Trained model
    """
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        class_weight=class_weight,
        random_state=random_state
    )
    rf_model.fit(X_train, y_train)

    if save_model:
        # save model and scaler
        model_package = {
            'model': rf_model,
            'scaler': scaler,
            'feature_names': feature_names
        }
        joblib.dump(model_package, model_path)
        print(f"RF model saved to {model_path}")

    return rf_model

def train_xgb_model(X_train, y_train, n_estimators=200, learning_rate=0.1, max_depth=3, subsample=0.8, colsample_bytree=0.8, 
                    scale_pos_weight=1.5, eval_metric="logloss", random_state=42, save_model=False, model_path=None, feature_names=None, scaler=None):
    """
    Train a XGBoost classifier with specified hyperparameters.

    Parameters:
        X_train (np.ndarray): Scaled feature matrix for training
        y_train (array-like): Binary labels for training
        n_estimators (int, optional): Number of boosting rounds/trees (default: 200)
        learning_rate (float, optional): Step size shrinkage to reduce overfitting (default: 0.1)
        max_depth (int, optional): Maximum depth of a tree (default: 3)
        subsample (float, optional): Fraction of training instances used per tree (default: 0.8)
        colsample_bytree (float, optional): Fraction of features used per tree (default: 0.8)
        scale_pos_weight (float, optional): Weighting factor to balance positive/negative classes (default: 1.5)
        eval_metric (str, optional): Evaluation metric used during training (default: 'logloss')
        random_state (int, optional): Random seed for reproducibility (default: 42)
        save_model (bool, optional): Whether to save the trained model (default: False)
        model_path (str, optional): Path to save the model if save_model is True
        feature_names (list, optional): Names of the features used for training
        scaler (StandardScaler, optional): Fitted scaler used to scale features

    Returns:
        xgb_model: Trained model
    """
    xgb_model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        eval_metric=eval_metric,
        random_state=random_state
    )
    xgb_model.fit(X_train, y_train)


    if save_model:
        # save model and scaler
        model_package = {
            'model': xgb_model,
            'scaler': scaler,
            'feature_names': feature_names
        }
        joblib.dump(model_package, model_path)
        print(f"XGB model saved to {model_path}")

    return xgb_model

def evaluate_model(model, X_test, y_test, col_ids=None, model_name="Model"):
    """
    Evaluate classifier performance on test data and print results.

    Parameters:
        model: trained classifier
        X_test, y_test: test data
        col_ids (optional): column IDs for identifying misclassified samples
        model_name: for display

    Returns:
        preds, probs, misclassified_col_ids (if col_ids given)
    """
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print(f"\nEvaluation for {model_name}")
    print(classification_report(y_test, preds))
    print(f"{model_name} ROC-AUC: {roc_auc_score(y_test, probs):.3f}")

    if col_ids is not None:
        misclassified = col_ids[(preds != y_test)]
        print(f"Misclassified {model_name} column IDs: {misclassified.values}")
        return preds, probs, misclassified.values

    return preds, probs


def cross_validate_auc(model, X_train, y_train, cv=5):
    """
    Perform k-fold cross-validation and print ROC-AUC scores.

    Parameters:
        model: trained classifer to evaluate
        X_train (np.ndarray or pd.DataFrame): Feature matrix for training
        y_train (array): Corresponding binary class labels
        cv (int, optional): Number of folds for cross-validation (default: 5)

    Returns:
        auc_scores (np.ndarray): Array of ROC-AUC scores from each fold

    Notes:
        This function is useful for checking the model's stability and generalisation.
        A large spread between folds may suggest overfitting.
    """
    auc_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"Cross-Validation AUC Scores ({cv}-fold):", auc_scores)
    print(f"Mean AUC: {auc_scores.mean():.3f}")
    return auc_scores

def test_with_reduced_training(model_func, X_train, y_train, X_test, y_test, scale=True):
    """
    Evaluate model performance when trained on half training set.
    Parameters:
        model_func (callable): Function that returns a trained model (ie. train_rf_model)
        X_train (np.ndarray): Full training feature set
        y_train (array-like): Corresponding training labels
        X_test (np.ndarray): Full test feature set
        y_test (array-like): Corresponding test labels
        scale (bool, optional): Whether to apply standard scaling to features (default: True)
    Returns:
       auc (float): ROC-AUC score on the test set after training on half of the training data
    Notes:
        This test helps evaluate overfitting. A steep drop in performance
        with reduced training data may indicate that the model is too dependent on specific samples.
    """

    X_train_small, _, y_train_small, _ = train_test_split(X_train, y_train, test_size=0.5, random_state=42)

    if scale:
        scaler = StandardScaler()
        X_train_small = scaler.fit_transform(X_train_small)
        X_test = scaler.transform(X_test)

    model_small = model_func(X_train_small, y_train_small)
    auc = roc_auc_score(y_test, model_small.predict_proba(X_test)[:, 1])
    print(f"AUC with reduced training data: {auc:.3f}")
    return auc

def predict_sg_probabilities(feature_df, model, scaler, model_name="model"):
    """
    Predict SG probabilities for each column using a trained model and scaler.

    Parameters:
        feature_df: DataFrame with features and 'col_id'
        model: trained model 
        scaler: fitted sklearn scaler used during training
        model_name: used for naming the output column
        
    Returns:
        feature_df with a new column: '{model_name}_sg_probability'
    """
    feature_df = feature_df.copy()

    # Drop non-feature columns
    features = feature_df.drop(columns=['col_id', 'label'], errors='ignore')
    features = features.loc[:, features.columns != f"{model_name}_sg_probability"]

    # Apply scaling
    features_scaled = scaler.transform(features)

    # Predict SG probability
    probs = model.predict_proba(features_scaled)[:, 1]

    # Store in new column
    feature_df[f"{model_name}_sg_probability"] = probs

    return feature_df


def predict_sg_probabilities_from_saved_model(feature_df, model_path, model_name="model"):
    """
    Predict SG probabilities using a saved model package.
    
    Parameters:
        feature_df: DataFrame with features and 'col_id'
        model_path: Path to saved .joblib model
        model_name: used for naming the output column
        
    Returns:
        feature_df with a new column: '{model_name}_sg_probability'
    """
    # Load model package
    model_package = joblib.load(model_path)
    model = model_package['model']
    scaler = model_package['scaler']
    expected_features = model_package['feature_names']
    
    feature_df = feature_df.copy()
    
    # Ensure feature consistency
    available_features = [col for col in feature_df.columns if col in expected_features]
    if len(available_features) != len(expected_features):
        missing = set(expected_features) - set(available_features)
        raise ValueError(f"Missing required features: {missing}")
    
    # Extract and order features correctly (important!)
    features = feature_df[expected_features]
    
    # Apply scaling
    features_scaled = scaler.transform(features)
    
    # Predict SG probability
    probs = model.predict_proba(features_scaled)[:, 1]
    
    # Store in new column
    feature_df[f"{model_name}_sg_probability"] = probs
    
    return feature_df