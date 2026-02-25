import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from utils.preprocessing import preprocess_image
from utils.feature_extraction import FeatureExtractor

def prepare_dataset(data_dir='dataset'):
    """
    Prepare dataset from directory structure
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
                        # Preprocess (returns normalized img, resized img, mask)
                        img_preprocessed, img_resized, mask = preprocess_image(img_path)

                        # Extract features (pass mask when available)
                        img_features = extractor.extract_all_features(img_preprocessed, mask)
                        
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

def analyze_features():
    """Analyze and visualize feature distributions"""
    os.makedirs('results', exist_ok=True)
    
    # Load all features
    if os.path.exists('features/all_features.csv'):
        df = pd.read_csv('features/all_features.csv')
        
        # Create feature comparison plot
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))
        axes = axes.ravel()
        
        for idx, feature in enumerate(df.columns[:-1]):  # Exclude label column
            if idx < 15:
                healthy_vals = df[df['label'] == 'sehat'][feature]
                sick_vals = df[df['label'] == 'sakit'][feature]
                
                axes[idx].hist(healthy_vals, alpha=0.5, label='Sehat', bins=20, color='green')
                axes[idx].hist(sick_vals, alpha=0.5, label='Sakit', bins=20, color='red')
                axes[idx].set_title(feature)
                axes[idx].legend()
        
        plt.tight_layout()
        plt.savefig('results/feature_distribution.png', dpi=150)
        plt.close()
        
        # Create statistical summary
        stats = df.groupby('label').agg(['mean', 'std', 'min', 'max'])
        stats.to_csv('results/feature_statistics.csv')
        
        print("Analisis fitur disimpan di folder 'results/'")