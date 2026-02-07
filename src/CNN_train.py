import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

def prepare_cnn_data(X_train, X_test, y_train, y_test):
    """
    Reshape data for CNN and convert labels from [1,2] to [0,1]
    
    Parameters:
        X_train, X_test: scaled feature arrays (n_samples, 29)
        y_train, y_test: labels as [1, 2]
        
    Returns:
        Reshaped X arrays (n_samples, 29, 1) and relabeled y arrays [0, 1]
    """
    # Reshape for CNN: (samples, sequence_length, channels)
    X_train_cnn = X_train.reshape(-1, 29, 1)
    X_test_cnn = X_test.reshape(-1, 29, 1)
    
    # Convert labels from [1,2] to [0,1]
    # Class 1 (SG) -> 0, Class 2 (non-SG) -> 1
    y_train_cnn = (y_train == 2).astype(int)
    y_test_cnn = (y_test == 2).astype(int)
    
    return X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn


def build_cnn_model(input_shape=(29, 1), learning_rate=0.001):
    """
    Build 1D CNN for intensity profile classification
    
    Architecture:
        - Conv1D layers learn local patterns in depth profiles
        - MaxPooling reduces dimensionality
        - Dropout prevents overfitting
        - Dense layers combine features for final classification
    """
    model = keras.Sequential([
        # First convolutional block
        # 32 filters of size 5 learn different 5-depth patterns
        layers.Conv1D(filters=32, kernel_size=5, activation='relu', 
                     input_shape=input_shape, padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # Second convolutional block
        # 64 filters of size 3 learn higher-level patterns
        layers.Conv1D(filters=64, kernel_size=3, activation='relu', 
                     padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # Third convolutional block
        layers.Conv1D(filters=128, kernel_size=3, activation='relu', 
                     padding='same'),
        layers.GlobalMaxPooling1D(),  # Pool across entire sequence
        
        # Dense classification layers
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # Output: probability
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    return model


def train_cnn_model(X_train, y_train, X_val, y_val, epochs=100, batch_size=32, 
                   patience=15, verbose=1):
    """
    Train CNN with early stopping
    
    Parameters:
        X_train, y_train: training data (already reshaped for CNN)
        X_val, y_val: validation data
        epochs: maximum training iterations
        batch_size: samples per gradient update
        patience: stop if validation doesn't improve for this many epochs
        verbose: 0=silent, 1=progress bar, 2=one line per epoch
        
    Returns:
        trained model, training history
    """
    # Build model
    model = build_cnn_model(input_shape=(X_train.shape[1], 1))
    
    # Print model architecture
    if verbose:
        print("\nModel Architecture:")
        model.summary()
        print()
    
    # Early stopping: stop training if validation AUC doesn't improve
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_auc',
        patience=patience,
        restore_best_weights=True,
        mode='max',
        verbose=1
    )
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=verbose
    )
    
    return model, history


def evaluate_cnn_model(model, X_test, y_test, model_name='cnn'):
    """
    Evaluate CNN performance
    
    Returns:
        predictions (0 or 1), probabilities (0 to 1)
    """
    # Get probability predictions
    probs = model.predict(X_test, verbose=0).flatten()
    
    # Convert to binary predictions
    preds = (probs > 0.5).astype(int)
    
    # Calculate metrics
    auc = roc_auc_score(y_test, probs)
    
    print(f"\n{'='*50}")
    print(f"{model_name.upper()} Model Evaluation")
    print(f"{'='*50}")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, preds, 
                                target_names=['SG (1)', 'non-SG (2)']))
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))
    
    return preds, probs


def plot_training_history(history):
    """
    Visualize training progress
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Training')
    axes[0].plot(history.history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Model Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Training')
    axes[1].plot(history.history['val_accuracy'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # AUC
    axes[2].plot(history.history['auc'], label='Training')
    axes[2].plot(history.history['val_auc'], label='Validation')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('AUC')
    axes[2].set_title('Model AUC')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def predict_cnn_probabilities(feature_df, model, scaler, middle_depths, 
                              model_name='cnn'):
    """
    Predict SG probabilities for full dataset using trained CNN
    
    Note: Returns probability of class 1 (SG), matching your RF convention
    after flipping from CNN's native output
    
    Parameters:
        feature_df: DataFrame with intensity profiles and 'col_id'
        model: trained CNN model
        scaler: fitted scaler from training
        middle_depths: list of depth columns to use
        model_name: prefix for output column
        
    Returns:
        feature_df with new column '{model_name}_sg_probability'
    """
    feature_df = feature_df.copy()
    
    # Extract features
    features = feature_df[middle_depths].values
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Reshape for CNN
    features_cnn = features_scaled.reshape(-1, len(middle_depths), 1)
    
    # Predict
    probs = model.predict(features_cnn, verbose=0).flatten()
    
    # CNN outputs probability of class 0 in our encoding (which is label=2, non-SG)
    # Flip to match RF convention: probability of SG (label=1)
    probs_sg = 1 - probs
    
    # Store in DataFrame
    feature_df[f'{model_name}_sg_probability'] = probs_sg
    
    return feature_df