import os
import joblib
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from utils.preprocessing import preprocess_image, preprocess_pipeline
from utils.feature_extraction import FeatureExtractor


def _save_preprocessed_training_images(img_path, class_name, original_filename, img_resized, gray_processed,
                                       resize_dir, threshold_dir):
    """Save resized and thresholded training images for later inspection."""
    os.makedirs(resize_dir, exist_ok=True)
    os.makedirs(threshold_dir, exist_ok=True)

    base_name, ext = os.path.splitext(original_filename)
    safe_base = f"{class_name}_{base_name}" if class_name else base_name
    safe_base = safe_base.replace(' ', '_')
    ext = ext.lower() if ext else '.png'

    resized_path = os.path.join(resize_dir, f"{safe_base}{ext}")
    threshold_path = os.path.join(threshold_dir, f"{safe_base}{ext}")

    try:
        cv2.imwrite(resized_path, img_resized)
    except Exception as e:
        print(f"  ⚠️ Gagal menyimpan resized training image {img_path}: {e}")

    try:
        cv2.imwrite(threshold_path, gray_processed)
    except Exception as e:
        print(f"  ⚠️ Gagal menyimpan threshold training image {img_path}: {e}")

def prepare_dataset(data_dir='dataset'):
    """
    Prepare dataset from directory structure using enhanced preprocessing pipeline
    
    Auto-detects class directories:
      - 'healthy' → label 'sehat'
      - Any directory starting with 'pmk_' → label = directory name
      - 'sick'    → label 'sakit' (backward compat)
    """
    features = []
    labels = []
    image_paths = []
    
    extractor = FeatureExtractor()
    
    # Auto-detect class directories
    class_dirs = {}
    if not os.path.exists(data_dir):
        raise ValueError(f"Directory {data_dir} tidak ditemukan!")
    for entry in sorted(os.listdir(data_dir)):
        entry_path = os.path.join(data_dir, entry)
        if os.path.isdir(entry_path):
            if entry == 'healthy':
                class_dirs[entry] = 'sehat'
            elif entry.startswith('pmk_'):
                class_dirs[entry] = entry
            elif entry == 'sick':
                class_dirs[entry] = 'sakit'

    if not class_dirs:
        raise ValueError(f"Tidak ada folder kelas yang ditemukan di {data_dir}!")

    resize_dir = os.path.join('uploads', 'resize')
    threshold_dir = os.path.join('uploads', 'threshold')
    
    for class_name, label in class_dirs.items():
        class_dir = os.path.join(data_dir, class_name)
        
        if os.path.exists(class_dir):
            print(f"Memproses gambar dari: {class_dir}")
            
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(class_dir, img_file)
                    try:
                        _, img_resized, _ = preprocess_image(img_path, target_size=(256, 256))
                        # Use threshold-based preprocessing pipeline
                        # Returns: img_rgb (for RGB features), gray_processed (for GLCM)
                        img_rgb, gray_eq = preprocess_pipeline(img_path, target_size=(256, 256))

                        _save_preprocessed_training_images(
                            img_path=img_path,
                            class_name=class_name,
                            original_filename=img_file,
                            img_resized=img_resized,
                            gray_processed=gray_eq,
                            resize_dir=resize_dir,
                            threshold_dir=threshold_dir,
                        )
                        
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

def save_model(model, scaler, label_encoder, prefix=''):
    """Save trained model and preprocessing objects"""
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(model, f'models/{prefix}knn_model.pkl')
    joblib.dump(scaler, f'models/{prefix}scaler.pkl')
    joblib.dump(label_encoder, f'models/{prefix}label_encoder.pkl')
    
    print(f"Model disimpan di folder 'models/' (prefix='{prefix}')")

def load_model(prefix=''):
    """Load trained model and preprocessing objects"""
    model = joblib.load(f'models/{prefix}knn_model.pkl')
    scaler = joblib.load(f'models/{prefix}scaler.pkl')
    label_encoder = joblib.load(f'models/{prefix}label_encoder.pkl')
    
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
            return float(np.clip(confidence, 50.0, 89.5))

        probabilities = model.predict_proba(features_array)[0]
        probabilities = np.asarray(probabilities, dtype=float)
        top = float(np.max(probabilities))
        if probabilities.size >= 2:
            second = float(np.partition(probabilities, -2)[-2])
        else:
            second = 0.0

        confidence = (0.9 * top + 0.1 * max(top - second, 0.0)) * 100.0
        return float(np.clip(confidence, 50.0, 89.5))
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
    n_features = min(len(feature_columns), 30)
    n_cols = 5
    n_rows = int(np.ceil(n_features / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.ravel()

    for idx in range(n_features):
        feature = feature_columns[idx]
        unique_labels = sorted(df[label_column].unique())
        cmap = plt.cm.Set1
        colors = [cmap(i % 9) for i in range(len(unique_labels))]

        for i, label_val in enumerate(unique_labels):
            vals = df[df[label_column] == label_val][feature]
            axes[idx].hist(vals, alpha=0.6, label=label_val, bins=20,
                           color=colors[i])
        axes[idx].set_title(feature)
        axes[idx].legend()

    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig('results/feature_distribution.png', dpi=150)
    plt.close()

    # Create statistical summary
    stats = df.groupby(label_column)[feature_columns].agg(['mean', 'std', 'min', 'max'])
    stats.to_csv('results/feature_statistics.csv')

    print("Analisis fitur disimpan di folder 'results/'")