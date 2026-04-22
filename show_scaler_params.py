#!/usr/bin/env python3
"""
Script untuk menampilkan StandardScaler parameters
Berguna untuk debugging dan verifikasi model
"""

import os
import sys
from utils.helpers import load_model, get_scaler_params
from utils.feature_extraction import FeatureExtractor

# ANSI color codes untuk terminal
class Colors:
    YELLOW = '\033[93m'
    BLACK = '\033[30m'
    BOLD = '\033[1m'
    END = '\033[0m'
    RESET = '\033[0m'

def print_colored_table(feature_names, params):
    """Print table dengan header berwarna kuning seperti Excel"""
    
    # Header dengan warna kuning
    header = f"{Colors.YELLOW}{Colors.BLACK}{Colors.BOLD}"
    header += f"{'feature':<20} {'mean':<15} {'std_dev':<15}"
    header += f"{Colors.END}"
    print(header)
    
    # Separator
    print("-" * 50)
    
    # Data rows
    for i, fname in enumerate(feature_names):
        mean_val = params['mean'][i]
        scale_val = params['scale'][i]
        
        print(f"{fname:<20} {mean_val:<15.10f} {scale_val:<15.10f}")
    
    print("-" * 50)

def export_scaler_params_to_csv(feature_names, params):
    """Export scaler parameters to CSV file"""
    import csv
    
    try:
        filename = 'scaler_parameters.csv'
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['feature', 'mean', 'std_dev'])
            
            # Data
            for i, fname in enumerate(feature_names):
                writer.writerow([
                    fname,
                    f"{params['mean'][i]:.10f}",
                    f"{params['scale'][i]:.10f}"
                ])
        
        print(f"\n✓ Scaler parameters exported to: {filename}")
        
    except Exception as e:
        print(f"\n❌ ERROR: Gagal mengekspor CSV: {str(e)}")

def main():
    """Display scaler parameters"""
    print("\n")
    print("="*70)
    print("SCALER PARAMETERS VIEWER")
    print("="*70)
    
    # Check if model exists
    if not os.path.exists('models/scaler.pkl'):
        print("\n❌ ERROR: Model scaler tidak ditemukan!")
        print("   Lokasi yang dicari: models/scaler.pkl")
        print("\n💡 Solusi: Jalankan 'python train_model.py' untuk melatih model terlebih dahulu")
        return
    
    try:
        # Load model
        print("\n📦 Loading model...")
        model, scaler, label_encoder = load_model()
        print("✓ Model loaded successfully")
        
        # Get feature names
        extractor = FeatureExtractor()
        feature_names = extractor.feature_names
        
        # Get scaler params
        params = get_scaler_params(scaler)
        
        if params is None:
            print("\n❌ ERROR: Gagal mengambil parameter scaler")
            return
        
        # Display table with colored header
        print("\n" + "="*70)
        print("SCALER PARAMETERS (StandardScaler)")
        print("="*70 + "\n")
        
        print_colored_table(feature_names, params)
        
        # Display additional info
        print("\n📊 INFORMASI TAMBAHAN:")
        print(f"  - Tipe Scaler: {params['type']}")
        print(f"  - Jumlah Features: {params['n_features']}")
        print(f"  - Tipe Model: KNeighborsClassifier (k=5)")
        print(f"  - Label Classes: {list(label_encoder.classes_)}")
        print(f"  - Status: Model siap digunakan ✓")
        
        # Penjelasan
        print("\n📝 PENJELASAN:")
        print("  - feature: Nama fitur yang digunakan dalam model")
        print("  - mean: Rata-rata nilai fitur dari training data")
        print("  - std_dev: Standar deviasi untuk normalisasi (scaling)")
        print("\n  Rumus normalisasi: X_normalized = (X - mean) / std_dev")
        
        print("\n" + "="*70 + "\n")
        
        # Export to CSV option
        export_csv = input("Apakah Anda ingin mengekspor data ini ke CSV? (y/n): ").strip().lower()
        if export_csv == 'y':
            export_scaler_params_to_csv(feature_names, params)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def export_scaler_params_to_csv(feature_names, params):
    """Export scaler parameters to CSV file"""
    import csv
    
    try:
        filename = 'scaler_parameters.csv'
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['Feature', 'Mean', 'Scale', 'Variance'])
            
            # Data
            for i, fname in enumerate(feature_names):
                writer.writerow([
                    fname,
                    params['mean'][i],
                    params['scale'][i],
                    params['variance'][i]
                ])
        
        print(f"\n✓ Scaler parameters exported to: {filename}")
        
    except Exception as e:
        print(f"\n❌ ERROR: Gagal mengekspor CSV: {str(e)}")


if __name__ == '__main__':
    main()
