import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import joblib

from utils.preprocessing import preprocess_image
from utils.feature_extraction import FeatureExtractor
from utils.helpers import load_model

class PMKDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistem Deteksi PMK pada Sapi")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')
        
        # Load model
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.extractor = FeatureExtractor()
        
        self.load_trained_model()
        
        # Variables
        self.image_path = None
        self.prediction = None
        self.confidence = None
        self.features = None
        
        # Setup UI
        self.setup_ui()
        
    def load_trained_model(self):
        """Load trained model"""
        try:
            self.model, self.scaler, self.label_encoder = load_model()
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            messagebox.showwarning("Peringatan", 
                                 "Model belum dilatih! Silakan jalankan train_model.py terlebih dahulu.")
    
    def setup_ui(self):
        """Setup user interface"""
        # Title
        title_label = tk.Label(self.root, 
                              text="SISTEM DETEKSI PENYAKIT MULUT DAN KUKU (PMK) PADA SAPI",
                              font=("Arial", 16, "bold"),
                              bg='#f0f0f0',
                              fg='#2c3e50')
        title_label.pack(pady=10)
        
        # Main frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left frame for image
        left_frame = tk.Frame(main_frame, bg='white', relief=tk.RAISED, bd=2)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Image label
        self.image_label = tk.Label(left_frame, text="Gambar akan ditampilkan di sini", 
                                   bg='white', fg='gray', font=("Arial", 10))
        self.image_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Right frame for controls and results
        right_frame = tk.Frame(main_frame, bg='#f0f0f0')
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)
        
        # Button frame
        btn_frame = tk.Frame(right_frame, bg='#f0f0f0')
        btn_frame.pack(pady=10)
        
        # Upload button
        upload_btn = tk.Button(btn_frame, text="📁 UPLOAD GAMBAR", 
                              command=self.upload_image,
                              font=("Arial", 10, "bold"),
                              bg='#3498db', fg='white',
                              padx=20, pady=10,
                              cursor="hand2")
        upload_btn.pack(pady=5)
        
        # Predict button
        predict_btn = tk.Button(btn_frame, text="🔍 PREDIKSI", 
                               command=self.predict_image,
                               font=("Arial", 10, "bold"),
                               bg='#2ecc71', fg='white',
                               padx=20, pady=10,
                               cursor="hand2",
                               state=tk.DISABLED)
        predict_btn.pack(pady=5)
        self.predict_btn = predict_btn
        
        # Results frame
        results_frame = tk.LabelFrame(right_frame, text="HASIL PREDIKSI", 
                                     font=("Arial", 11, "bold"),
                                     bg='#f0f0f0', fg='#2c3e50',
                                     padx=10, pady=10)
        results_frame.pack(pady=20, fill=tk.X)
        
        # Status label
        self.status_label = tk.Label(results_frame, 
                                    text="Status: Belum diprediksi",
                                    font=("Arial", 10),
                                    bg='#f8f9fa',
                                    relief=tk.SUNKEN,
                                    padx=10, pady=10,
                                    width=30)
        self.status_label.pack(pady=5)
        
        # Result label
        self.result_label = tk.Label(results_frame, 
                                    text="",
                                    font=("Arial", 12, "bold"),
                                    bg='#f8f9fa',
                                    padx=10, pady=5)
        self.result_label.pack(pady=5)
        
        # Confidence label
        self.confidence_label = tk.Label(results_frame,
                                        text="",
                                        font=("Arial", 10),
                                        bg='#f8f9fa',
                                        padx=10, pady=5)
        self.confidence_label.pack(pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(results_frame, 
                                       mode='indeterminate',
                                       length=200)
        
        # Features frame
        features_frame = tk.LabelFrame(right_frame, text="FITUR YANG DIEKSTRAKSI", 
                                      font=("Arial", 11, "bold"),
                                      bg='#f0f0f0', fg='#2c3e50',
                                      padx=10, pady=10)
        features_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Features text box
        self.features_text = tk.Text(features_frame, 
                                    height=10,
                                    width=40,
                                    font=("Courier", 8))
        self.features_text.pack(fill=tk.BOTH, expand=True)
        
        # Add scrollbar
        scrollbar = tk.Scrollbar(features_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.features_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.features_text.yview)
        
        # Statistics button
        stats_btn = tk.Button(right_frame, text="📊 LIHAT STATISTIK", 
                             command=self.show_statistics,
                             font=("Arial", 9),
                             bg='#9b59b6', fg='white',
                             padx=10, pady=5,
                             cursor="hand2")
        stats_btn.pack(pady=10)
        
    def upload_image(self):
        """Upload image for prediction"""
        file_path = filedialog.askopenfilename(
            title="Pilih gambar sapi",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            self.image_path = file_path
            self.display_image(file_path)
            self.predict_btn.config(state=tk.NORMAL)
            self.clear_results()
            
    def display_image(self, image_path):
        """Display selected image"""
        try:
            # Open and resize image
            img = Image.open(image_path)
            img.thumbnail((400, 400))
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)
            
            # Update label
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
            
        except Exception as e:
            messagebox.showerror("Error", f"Gagal memuat gambar: {str(e)}")
    
    def predict_image(self):
        """Predict image using trained model"""
        if not self.image_path:
            messagebox.showwarning("Peringatan", "Silakan pilih gambar terlebih dahulu!")
            return
        
        if not self.model:
            messagebox.showwarning("Peringatan", "Model belum dimuat!")
            return
        
        try:
            # Show progress
            self.progress.pack(pady=5)
            self.progress.start()
            self.status_label.config(text="Status: Memproses...")
            self.root.update()
            
            # 1. Preprocess image (returns normalized, resized, mask)
            img_preprocessed, img_original, mask = preprocess_image(self.image_path)

            # 2. Extract features (pass mask when available)
            self.features = self.extractor.extract_all_features(img_preprocessed, mask)
            
            # 3. Scale features
            features_scaled = self.scaler.transform([self.features])
            
            # 4. Predict
            prediction_encoded = self.model.predict(features_scaled)[0]
            self.prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
            
            # 5. Get prediction probabilities
            probabilities = self.model.predict_proba(features_scaled)[0]
            self.confidence = max(probabilities) * 100
            
            # 6. Display results
            self.display_results()
            
            # 7. Save prediction to CSV
            self.save_prediction_to_csv()
            
            # Stop progress
            self.progress.stop()
            self.progress.pack_forget()
            
        except Exception as e:
            self.progress.stop()
            self.progress.pack_forget()
            messagebox.showerror("Error", f"Terjadi kesalahan: {str(e)}")
            self.status_label.config(text="Status: Error")
    
    def display_results(self):
        """Display prediction results"""
        # Update status
        self.status_label.config(text="Status: Selesai")
        
        # Set result label with color
        if self.prediction == 'sehat':
            color = 'green'
            text = "✅ SAPI SEHAT"
        else:
            color = 'red'
            text = "⚠️ SAPI SAKIT (TERDETEKSI PMK)"
        
        self.result_label.config(text=text, fg=color)
        
        # Display confidence
        self.confidence_label.config(
            text=f"Tingkat Kepercayaan: {self.confidence:.2f}%",
            fg='blue' if self.confidence > 70 else 'orange'
        )
        
        # Display extracted features
        self.display_features()
    
    def display_features(self):
        """Display extracted features in text box"""
        if self.features is not None:
            self.features_text.delete(1.0, tk.END)
            
            # Create formatted string
            features_str = "Fitur yang diekstraksi:\n"
            features_str += "=" * 40 + "\n"
            
            for i, (name, value) in enumerate(zip(self.extractor.feature_names, self.features)):
                features_str += f"{name:15}: {value:10.6f}\n"
                if i == 8:  # After color moments
                    features_str += "-" * 40 + "\n"
            
            features_str += "=" * 40 + "\n"
            features_str += f"\nHasil Prediksi: {self.prediction.upper()}\n"
            features_str += f"Confidence: {self.confidence:.2f}%"
            
            self.features_text.insert(1.0, features_str)
    
    def save_prediction_to_csv(self):
        """Save prediction results to CSV"""
        try:
            os.makedirs('results', exist_ok=True)
            
            # Create data dictionary
            data = {
                'image_path': [self.image_path],
                'prediction': [self.prediction],
                'confidence': [self.confidence],
                'timestamp': [pd.Timestamp.now()]
            }
            
            # Add features
            for i, name in enumerate(self.extractor.feature_names):
                data[name] = [self.features[i]]
            
            # Save to CSV
            df = pd.DataFrame(data)
            
            # Check if file exists
            if os.path.exists('results/predictions.csv'):
                df.to_csv('results/predictions.csv', mode='a', header=False, index=False)
            else:
                df.to_csv('results/predictions.csv', index=False)
            
            print(f"Prediksi disimpan ke: results/predictions.csv")
            
        except Exception as e:
            print(f"Error saving prediction: {e}")
    
    def show_statistics(self):
        """Show feature statistics"""
        try:
            if os.path.exists('features/all_features.csv'):
                df = pd.read_csv('features/all_features.csv')
                
                # Create statistics window
                stats_window = tk.Toplevel(self.root)
                stats_window.title("Statistik Fitur")
                stats_window.geometry("600x400")
                
                # Create text widget
                text_widget = tk.Text(stats_window, font=("Courier", 9))
                text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                
                # Calculate statistics
                stats = df.groupby('label').agg(['mean', 'std', 'min', 'max'])
                
                # Format output
                output = "STATISTIK FITUR PER KELAS\n"
                output += "=" * 60 + "\n\n"
                
                for label in ['sehat', 'sakit']:
                    output += f"KELAS: {label.upper()}\n"
                    output += "-" * 40 + "\n"
                    
                    if label in stats.index:
                        label_stats = stats.loc[label]
                        for feature in self.extractor.feature_names:
                            if feature in label_stats:
                                mean = label_stats[feature]['mean']
                                std = label_stats[feature]['std']
                                output += f"{feature:15}: {mean:8.4f} ± {std:8.4f}\n"
                    
                    output += "\n"
                
                # Add comparison
                output += "PERBANDINGAN RATA-RATA\n"
                output += "-" * 40 + "\n"
                
                if 'sehat' in stats.index and 'sakit' in stats.index:
                    for feature in self.extractor.feature_names:
                        sehat_mean = stats.loc['sehat', feature]['mean']
                        sakit_mean = stats.loc['sakit', feature]['mean']
                        diff = sakit_mean - sehat_mean
                        output += f"{feature:15}: {sehat_mean:8.4f} → {sakit_mean:8.4f} ({diff:+.4f})\n"
                
                text_widget.insert(1.0, output)
                
            else:
                messagebox.showinfo("Info", "File statistik belum tersedia. Silakan train model terlebih dahulu.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Gagal memuat statistik: {str(e)}")
    
    def clear_results(self):
        """Clear previous results"""
        self.result_label.config(text="")
        self.confidence_label.config(text="")
        self.features_text.delete(1.0, tk.END)
        self.status_label.config(text="Status: Belum diprediksi")

def main():
    root = tk.Tk()
    app = PMKDetectorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()