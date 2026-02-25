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
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels_encoded, 
        test_size=test_size, 
        random_state=random_state,
        stratify=labels_encoded
    )
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    
    # 4. Feature scaling
    print("\n4. SCALING FEATURES...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Train KNN model
    print("\n5. TRAINING KNN MODEL...")
    knn = KNeighborsClassifier(
        n_neighbors=5,
        weights='distance',
        metric='euclidean',
        n_jobs=-1
    )
    
    knn.fit(X_train_scaled, y_train)
    
    # 6. Evaluate model
    print("\n6. EVALUASI MODEL...")
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
    
    # 7. Save model
    print("\n7. MENYIMPAN MODEL...")
    save_model(knn, scaler, label_encoder)
    
    # 8. Save features to separate CSV files
    print("\n8. MENYIMPAN FITUR KE CSV...")
    extractor = FeatureExtractor()
    os.makedirs('features', exist_ok=True)
    
    # Separate features by class
    healthy_features = []
    sick_features = []
    
    for feat, label in zip(features, labels):
        if label == 'sehat':
            healthy_features.append(feat)
        else:
            sick_features.append(feat)
    
    # Save to CSV
    if healthy_features:
        df_healthy = extractor.save_features_to_csv(
            healthy_features, 
            ['sehat'] * len(healthy_features),
            'features/features_healthy.csv'
        )
        print(f"  Fitur sehat: {len(df_healthy)} sampel")
    
    if sick_features:
        df_sick = extractor.save_features_to_csv(
            sick_features, 
            ['sakit'] * len(sick_features),
            'features/features_sick.csv'
        )
        print(f"  Fitur sakit: {len(df_sick)} sampel")
    
    # Save all features
    all_features_df = pd.DataFrame(features, columns=extractor.feature_names)
    all_features_df['label'] = labels
    all_features_df.to_csv('features/all_features.csv', index=False)
    
    # 9. Analyze features
    print("\n9. ANALISIS FITUR...")
    analyze_features()
    
    # Calculate feature statistics
    print("\nRATA-RATA FITUR PER KELAS:")
    stats = all_features_df.groupby('label').mean()
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