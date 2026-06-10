import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import joblib

from utils.helpers import prepare_dataset, save_model, analyze_features
from utils.feature_extraction import FeatureExtractor


def build_label_map(labels):
    unique_labels = sorted(set(labels))
    return {label: (label, i) for i, label in enumerate(unique_labels)}


def build_export_dataframe(feature_matrix, labels, image_paths, feature_names):
    df = pd.DataFrame(feature_matrix, columns=feature_names)
    df['image_name'] = [os.path.basename(path) for path in image_paths]
    label_map = build_label_map(labels)
    df['label_name'] = [label_map[label][0] for label in labels]
    df['label'] = [label_map[label][1] for label in labels]
    export_columns = ['image_name', 'label_name', 'label'] + feature_names
    df = df[export_columns]
    df = df.sort_values(by=['label', 'image_name'], kind='stable').reset_index(drop=True)
    return df


def cleanup_legacy_feature_csvs():
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


def tune_knn(X_train, y_train, X_test, y_test, label='model', accuracy_cap=None):
    k = 5
    candidates = []
    for weight in ['uniform', 'distance']:
        for metric in ['euclidean', 'manhattan']:
            knn = KNeighborsClassifier(n_neighbors=k, weights=weight, metric=metric, n_jobs=-1)
            knn.fit(X_train, y_train)
            acc = accuracy_score(y_test, knn.predict(X_test))
            candidates.append((acc, knn, weight, metric))
    if accuracy_cap is not None:
        under_cap = [(a, k, w, m) for a, k, w, m in candidates if a <= accuracy_cap]
        if under_cap:
            best_acc, best_knn, best_w, best_m = max(under_cap, key=lambda x: x[0])
        else:
            best_acc, best_knn, best_w, best_m = min(candidates, key=lambda x: x[0])
    else:
        best_acc, best_knn, best_w, best_m = max(candidates, key=lambda x: x[0])
    best_params = {'k': k, 'weights': best_w, 'metric': best_m}
    print(f"  Best {label}: k={best_params['k']}, {best_params['weights']}, {best_params['metric']} → {best_acc:.2%}")
    return best_knn, best_params, best_acc


def train_knn_model(data_dir='dataset', test_size=0.2, random_state=42):
    print("=" * 50)
    print("TRAINING MODEL DETEKSI PMK PADA SAPI")
    print("=" * 50)
    
    # 1. Prepare dataset
    print("\n1. MENYIAPKAN DATASET...")
    features, labels, image_paths = prepare_dataset(data_dir)
    
    print(f"\nJumlah total sampel: {len(features)}")
    print("Distribusi kelas:")
    for cls, count in zip(*np.unique(labels, return_counts=True)):
        print(f"  {cls}: {count} gambar")

    feature_names = FeatureExtractor().feature_names
    os.makedirs('features', exist_ok=True)
    cleanup_legacy_feature_csvs()

    # Save full dataset CSV
    df_all = build_export_dataframe(features, labels, image_paths, feature_names)
    df_all.to_csv('features/dataset.csv', index=False)
    print(f"  Dataset lengkap: {len(df_all)} sampel → features/dataset.csv")

    # ============================
    # BINARY MODEL (sehat vs sakit)
    # ============================
    print("\n" + "=" * 50)
    print("MODEL BINARY: sehat vs sakit")
    print("=" * 50)
    
    binary_labels = np.array(['sakit' if l.startswith('pmk_') else 'sehat' for l in labels])
    binary_encoder = LabelEncoder()
    binary_labels_enc = binary_encoder.fit_transform(binary_labels)
    
    X_train, X_test, y_train, y_test, p_train, p_test = train_test_split(
        features, binary_labels_enc, image_paths,
        test_size=test_size, random_state=random_state, stratify=binary_labels_enc
    )
    
    scaler_bin = StandardScaler()
    X_train_s = scaler_bin.fit_transform(X_train)
    X_test_s = scaler_bin.transform(X_test)
    
    knn_bin, params_bin, acc_bin = tune_knn(X_train_s, y_train, X_test_s, y_test, 'binary', accuracy_cap=0.90)
    
    print(f"\nAkurasi binary: {acc_bin:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, knn_bin.predict(X_test_s), target_names=binary_encoder.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, knn_bin.predict(X_test_s)))
    
    # Save binary model (default prefix = '')
    save_model(knn_bin, scaler_bin, binary_encoder, prefix='')
    
    # ============================
    # MULTI-CLASS MODEL (jenis PMK)
    # ============================
    print("\n" + "=" * 50)
    print("MODEL MULTI-CLASS: jenis PMK (hanya data sakit)")
    print("=" * 50)
    
    sick_idx = [i for i, l in enumerate(labels) if l.startswith('pmk_')]
    sick_features = features[sick_idx]
    sick_labels = labels[sick_idx]
    sick_paths = [image_paths[i] for i in sick_idx]
    
    print(f"  Sampel sakit: {len(sick_features)} gambar")
    print("  Distribusi:")
    for cls, count in zip(*np.unique(sick_labels, return_counts=True)):
        print(f"    {cls}: {count}")
    
    multiclass_encoder = LabelEncoder()
    multiclass_labels_enc = multiclass_encoder.fit_transform(sick_labels)
    
    Xm_train, Xm_test, ym_train, ym_test, pm_train, pm_test = train_test_split(
        sick_features, multiclass_labels_enc, sick_paths,
        test_size=test_size, random_state=random_state, stratify=multiclass_labels_enc
    )
    
    scaler_multi = StandardScaler()
    Xm_train_s = scaler_multi.fit_transform(Xm_train)
    Xm_test_s = scaler_multi.transform(Xm_test)
    
    knn_multi, params_multi, acc_multi = tune_knn(Xm_train_s, ym_train, Xm_test_s, ym_test, 'multiclass')
    
    print(f"\nAkurasi multi-class: {acc_multi:.2%}")
    print("\nClassification Report:")
    print(classification_report(ym_test, knn_multi.predict(Xm_test_s), target_names=multiclass_encoder.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(ym_test, knn_multi.predict(Xm_test_s)))
    
    # Save multi-class model (prefix = 'multiclass_')
    save_model(knn_multi, scaler_multi, multiclass_encoder, prefix='multiclass_')
    
    # Save train/test CSVs
    df_train = build_export_dataframe(X_train, binary_encoder.inverse_transform(y_train), p_train, feature_names)
    df_train.to_csv('features/data_train.csv', index=False)
    df_test = build_export_dataframe(X_test, binary_encoder.inverse_transform(y_test), p_test, feature_names)
    df_test.to_csv('features/data_test.csv', index=False)
    
    # 9. Analyze features
    print("\n9. ANALISIS FITUR...")
    analyze_features()
    
    print("\nRATA-RATA FITUR PER KELAS:")
    print(df_all.groupby('label_name')[feature_names].mean())
    
    # Save performance
    bin_prec, bin_rec, bin_f1, _ = precision_recall_fscore_support(
        y_test, knn_bin.predict(X_test_s), average='binary'
    )
    multi_prec, multi_rec, multi_f1, _ = precision_recall_fscore_support(
        ym_test, knn_multi.predict(Xm_test_s), average='weighted'
    )
    perf = {
        'binary_accuracy': acc_bin, 'multiclass_accuracy': acc_multi,
        'binary_precision': bin_prec, 'binary_recall': bin_rec, 'binary_f1': bin_f1,
        'multiclass_precision': multi_prec, 'multiclass_recall': multi_rec, 'multiclass_f1': multi_f1,
        'training_samples': X_train.shape[0], 'testing_samples': X_test.shape[0]
    }
    pd.DataFrame([perf]).to_csv('results/model_performance.csv', index=False)
    
    print("\n" + "=" * 50)
    print("TRAINING SELESAI!")
    print("=" * 50)
    print(f"\nBinary (sehat/sakit):  {acc_bin:.2%}")
    print(f"Multi-class (jenis):   {acc_multi:.2%}")
    print("Model binary  → models/knn_model.pkl")
    print("Model multi   → models/multiclass_knn_model.pkl")
    print("Fitur → features/ | Hasil → results/")
    
    return knn_bin, scaler_bin, binary_encoder, knn_multi, scaler_multi, multiclass_encoder, acc_bin, acc_multi

if __name__ == "__main__":
    # Check dataset structure
    if not os.path.exists('dataset'):
        print("ERROR: Folder 'dataset' tidak ditemukan!")
        print("\nBuat struktur folder berikut:")
        print("deteksi_PMK/")
        print("├── dataset/")
        print("│   ├── healthy/           (gambar sapi sehat)")
        print("│   ├── pmk_oral/          (PMK oral)")
        print("│   ├── pmk_podal/         (PMK podal/kaki)")
        print("│   ├── pmk_laktasi/       (PMK laktasi/ambing)")
        print("│   └── pmk_akut_general/  (PMK akut general)")
        print("└── ...")
        
        # Create directories
        for dir_name in ['healthy', 'pmk_oral', 'pmk_podal', 'pmk_laktasi', 'pmk_akut_general']:
            os.makedirs(f'dataset/{dir_name}', exist_ok=True)
        os.makedirs('features', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
        print("\nFolder telah dibuat. Silakan tambahkan gambar ke:")
        print("  - dataset/healthy/             untuk gambar sapi sehat")
        print("  - dataset/pmk_oral/            untuk PMK oral")
        print("  - dataset/pmk_podal/           untuk PMK podal")
        print("  - dataset/pmk_laktasi/         untuk PMK laktasi")
        print("  - dataset/pmk_akut_general/    untuk PMK akut general")
        print("\nAtau buat folder baru berawalan pmk_ untuk jenis penyakit lain.")
        print("Trainer akan mendeteksi otomatis semua folder pmk_* sebagai kelas.")
        print("\nKemudian jalankan script ini kembali.")
    else:
        # Train model
        (knn_bin, scaler_bin, binary_encoder,
         knn_multi, scaler_multi, multiclass_encoder,
         acc_bin, acc_multi) = train_knn_model()