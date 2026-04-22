import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

from utils.helpers import prepare_dataset, save_model, analyze_features
from utils.feature_extraction import FeatureExtractor


LABEL_EXPORT_MAP = {
    'sehat': ('normal', 0),
    'sakit': ('defective', 1),
}


def build_export_dataframe(feature_matrix, labels, image_paths, feature_names):
    """Build a CSV-ready dataframe with image names and export labels."""
    df = pd.DataFrame(feature_matrix, columns=feature_names)
    df['image_name'] = [os.path.basename(path) for path in image_paths]
    df['label_name'] = [LABEL_EXPORT_MAP.get(label, (label, -1))[0] for label in labels]
    df['label'] = [LABEL_EXPORT_MAP.get(label, (label, -1))[1] for label in labels]

    export_columns = ['image_name', 'label_name', 'label'] + feature_names
    df = df[export_columns]
    df = df.sort_values(by=['label', 'image_name'], kind='stable').reset_index(drop=True)
    return df


def cleanup_legacy_feature_csvs():
    """Remove legacy CSV exports so the features folder only contains the new files."""
    legacy_files = [
        'features/data_train_scaled.csv',
        'features/data_test_scaled.csv',
        'features/all_features.csv',
        'features/features_healthy.csv',
        'features/features_sick.csv',
    ]

    for file_path in legacy_files:
        if os.path.exists(file_path):
            os.remove(file_path)

def train_knn_model(data_dir='dataset', test_size=0.2, random_state=42):
    """
    Train KNN model for PMK detection
    """
    print("=" * 50)
    print("TRAINING MODEL DETEKSI PMK PADA SAPI")
    print("=" * 50)
    
    # 1. Prepare dataset
    print("\n1. MENYIAPKAN DATASET...")
    features, labels, image_paths = prepare_dataset(data_dir)
    
    print(f"\nJumlah total sampel: {len(features)}")
    print("Distribusi kelas:")
    unique, counts = np.unique(labels, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  {cls}: {count} gambar")
    
    # 2. Encode labels
    print("\n2. ENCODING LABELS...")
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    # 3. Split dataset
    print("\n3. MEMBAGI DATASET...")
    X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
        features, labels_encoded, image_paths,
        test_size=test_size, 
        random_state=random_state,
        stratify=labels_encoded
    )
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    print(f"Feature dimensionality: {X_train.shape[1]} (Average RGB 3 + GLCM 4)")
    
    # 3a. Save dataset and train-test split to CSV
    print("\n3a. MENYIMPAN DATASET DAN SPLIT TRAIN-TEST KE CSV...")
    extractor_temp = FeatureExtractor()
    os.makedirs('features', exist_ok=True)
    
    # Create DataFrames (use 7 feature names)
    feature_names_for_export = extractor_temp.feature_names

    cleanup_legacy_feature_csvs()

    df_all = build_export_dataframe(features, labels, image_paths, feature_names_for_export)
    df_all.to_csv('features/dataset.csv', index=False)
    print(f"  Dataset penuh: {len(df_all)} sampel → features/dataset.csv")

    df_train = build_export_dataframe(X_train, label_encoder.inverse_transform(y_train), paths_train, feature_names_for_export)
    df_train.to_csv('features/data_train.csv', index=False)
    print(f"  Data training: {len(df_train)} sampel → features/data_train.csv")

    df_test = build_export_dataframe(X_test, label_encoder.inverse_transform(y_test), paths_test, feature_names_for_export)
    df_test.to_csv('features/data_test.csv', index=False)
    print(f"  Data testing: {len(df_test)} sampel → features/data_test.csv")

    # 5. Feature scaling
    print("\n5. SCALING FEATURES (7 features)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Train KNN model dengan k=5 (optimized)
    print("\n6. TRAINING KNN MODEL (k=5 - optimized)...")
    
    knn = KNeighborsClassifier(
        n_neighbors=5,
        weights='distance',
        metric='euclidean',
        n_jobs=-1
    )
    
    knn.fit(X_train_scaled, y_train)
    
    # 7. Evaluate model
    print("\n7. EVALUASI MODEL...")
    y_pred = knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "=" * 50)
    print("HASIL EVALUASI")
    print("=" * 50)
    print(f"\nAkurasi Model: {accuracy:.2%}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=label_encoder.classes_))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Calculate precision, recall, F1-score
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nPrecision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1-Score: {f1:.2%}")
    
    # 8. Save model
    print("\n8. MENYIMPAN MODEL...")
    save_model(knn, scaler, label_encoder)
    
    # 9. Analyze features
    print("\n10. ANALISIS FITUR...")
    analyze_features()
    
    # Calculate feature statistics
    print("\nRATA-RATA FITUR PER KELAS:")
    stats = df_all.groupby('label_name')[feature_names_for_export].mean()
    print(stats)
    
    # Save model performance
    performance = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'training_samples': X_train.shape[0],
        'testing_samples': X_test.shape[0]
    }
    
    perf_df = pd.DataFrame([performance])
    perf_df.to_csv('results/model_performance.csv', index=False)
    
    print("\n" + "=" * 50)
    print("TRAINING SELESAI!")
    print("=" * 50)
    print(f"\nAkurasi model: {accuracy:.2%}")
    print("Model disimpan di: models/knn_model.pkl")
    print("Fitur disimpan di: features/")
    print("Hasil analisis di: results/")
    
    return knn, scaler, label_encoder, accuracy

if __name__ == "__main__":
    # Check dataset structure
    if not os.path.exists('dataset'):
        print("ERROR: Folder 'dataset' tidak ditemukan!")
        print("\nBuat struktur folder berikut:")
        print("pmk_detection_desktop/")
        print("├── dataset/")
        print("│   ├── healthy/   (isi dengan gambar sapi sehat)")
        print("│   └── sick/      (isi dengan gambar sapi sakit)")
        print("└── ...")
        
        # Create directories
        os.makedirs('dataset/healthy', exist_ok=True)
        os.makedirs('dataset/sick', exist_ok=True)
        os.makedirs('features', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
        print("\nFolder telah dibuat. Silakan tambahkan gambar ke:")
        print("  - dataset/healthy/  untuk gambar sapi sehat")
        print("  - dataset/sick/     untuk gambar sapi sakit")
        print("\nKemudian jalankan script ini kembali.")
    else:
        # Train model
        model, scaler, label_encoder, accuracy = train_knn_model()