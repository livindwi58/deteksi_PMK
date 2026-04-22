import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from utils.preprocessing import preprocess_image, preprocess_pipeline
from utils.feature_extraction import FeatureExtractor

def prepare_dataset(data_dir='dataset'):
    """
    Prepare dataset from directory structure using enhanced preprocessing pipeline
    """
    features = []
    labels = []
    image_paths = []
    
    extractor = FeatureExtractor()
    
    # Define class directories
    class_dirs = {
        'healthy': 'sehat',
        'sick': 'sakit'
    }
    
    for class_name, label in class_dirs.items():
        class_dir = os.path.join(data_dir, class_name)
        
        if os.path.exists(class_dir):
            print(f"Memproses gambar dari: {class_dir}")
            
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(class_dir, img_file)
                    try:
                        # Use threshold-based preprocessing pipeline
                        # Returns: img_rgb (for RGB features), gray_processed (for GLCM)
                        img_rgb, gray_eq = preprocess_pipeline(img_path, target_size=(128, 128))
                        
                        # Extract features (RGB averages + GLCM dari preprocessing pipeline)
                        img_features = extractor.extract_all_features(img_rgb, gray_eq)
                        
                        features.append(img_features)
                        labels.append(label)
                        image_paths.append(img_path)
                        
                        print(f"  ✓ {img_file}")
                        
                    except Exception as e:
                        print(f"  ✗ Error processing {img_file}: {e}")
    
    if len(features) == 0:
        raise ValueError("Tidak ada gambar yang ditemukan untuk training!")
    
    return np.array(features), np.array(labels), image_paths

def save_model(model, scaler, label_encoder):
    """Save trained model and preprocessing objects"""
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(model, 'models/knn_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(label_encoder, 'models/label_encoder.pkl')
    
    print("Model disimpan di folder 'models/'")

def load_model():
    """Load trained model and preprocessing objects"""
    model = joblib.load('models/knn_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    label_encoder = joblib.load('models/label_encoder.pkl')
    
    return model, scaler, label_encoder


def estimate_prediction_confidence(model, features_scaled):
    """Estimate a conservative prediction confidence for the current model."""
    try:
        features_array = np.asarray(features_scaled)
        if features_array.ndim == 1:
            features_array = features_array.reshape(1, -1)

        if hasattr(model, 'kneighbors') and hasattr(model, 'classes_') and hasattr(model, '_y'):
            n_neighbors = int(getattr(model, 'n_neighbors', 5))
            distances, indices = model.kneighbors(features_array, n_neighbors=n_neighbors)
            neighbor_labels = np.asarray(model._y)[indices[0]]
            weights = 1.0 / (distances[0] + 1e-6)

            class_weights = {}
            for label, weight in zip(neighbor_labels, weights):
                class_weights[label] = class_weights.get(label, 0.0) + float(weight)

            sorted_weights = sorted(class_weights.values(), reverse=True)
            top_weight = sorted_weights[0]
            second_weight = sorted_weights[1] if len(sorted_weights) > 1 else 0.0
            total_weight = sum(sorted_weights)
            n_classes = max(len(getattr(model, 'classes_', [])), 2)

            support = (top_weight + 1.0) / (total_weight + n_classes)
            margin = (top_weight - second_weight) / total_weight if total_weight > 0 else 0.0
            confidence = (0.85 * support + 0.15 * margin) * 100.0
            return float(np.clip(confidence, 50.0, 98.5))

        probabilities = model.predict_proba(features_array)[0]
        probabilities = np.asarray(probabilities, dtype=float)
        top = float(np.max(probabilities))
        if probabilities.size >= 2:
            second = float(np.partition(probabilities, -2)[-2])
        else:
            second = 0.0

        confidence = (0.9 * top + 0.1 * max(top - second, 0.0)) * 100.0
        return float(np.clip(confidence, 50.0, 98.5))
    except Exception:
        return None

def get_scaler_params(scaler):
    """
    Extract and return scaler parameters (mean, scale, variance)
    
    Parameters:
    - scaler: StandardScaler object from sklearn
    
    Return: dictionary dengan scaler parameters
    """
    if scaler is None:
        return None
    
    try:
        params = {
            'type': scaler.__class__.__name__,
            'mean': scaler.mean_.tolist() if hasattr(scaler, 'mean_') else None,
            'scale': scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None,
            'variance': scaler.var_.tolist() if hasattr(scaler, 'var_') else None,
            'n_features': scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else None,
        }
        return params
    except Exception as e:
        print(f"Error extracting scaler params: {e}")
        return None

def analyze_features():
    """Analyze and visualize feature distributions"""
    os.makedirs('results', exist_ok=True)

    # Load dataset features (prefer the new consolidated export)
    dataset_path = 'features/dataset.csv'
    legacy_path = 'features/all_features.csv'

    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
    elif os.path.exists(legacy_path):
        df = pd.read_csv(legacy_path)
    else:
        return

    extractor = FeatureExtractor()
    feature_columns = [feature for feature in extractor.feature_names if feature in df.columns]
    if not feature_columns:
        return

    label_column = 'label_name' if 'label_name' in df.columns else 'label'

    # Create feature comparison plot
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.ravel()

    for idx, feature in enumerate(feature_columns[:15]):
        if label_column == 'label_name':
            healthy_vals = df[df[label_column] == 'normal'][feature]
            sick_vals = df[df[label_column] == 'defective'][feature]
            healthy_label = 'Normal'
            sick_label = 'Defective'
        else:
            healthy_vals = df[df[label_column] == 'sehat'][feature]
            sick_vals = df[df[label_column] == 'sakit'][feature]
            healthy_label = 'Sehat'
            sick_label = 'Sakit'

        axes[idx].hist(healthy_vals, alpha=0.5, label=healthy_label, bins=20, color='green')
        axes[idx].hist(sick_vals, alpha=0.5, label=sick_label, bins=20, color='red')
        axes[idx].set_title(feature)
        axes[idx].legend()

    for idx in range(len(feature_columns), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig('results/feature_distribution.png', dpi=150)
    plt.close()

    # Create statistical summary
    stats = df.groupby(label_column)[feature_columns].agg(['mean', 'std', 'min', 'max'])
    stats.to_csv('results/feature_statistics.csv')

    print("Analisis fitur disimpan di folder 'results/'")