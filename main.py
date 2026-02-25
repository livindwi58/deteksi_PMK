import os
import tkinter as tk
from tkinter import messagebox, ttk

def create_directories():
    """Create necessary directories"""
    directories = ['dataset/healthy', 'dataset/sick', 'features', 'models', 'results']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    return directories

def main_menu():
    """Main menu for the application"""
    root = tk.Tk()
    root.title("Sistem Deteksi PMK - Menu Utama")
    root.geometry("500x400")
    root.configure(bg='#2c3e50')
    
    # Title
    title_label = tk.Label(root, 
                          text="SISTEM DETEKSI PMK PADA SAPI",
                          font=("Arial", 18, "bold"),
                          bg='#2c3e50',
                          fg='white')
    title_label.pack(pady=20)
    
    subtitle_label = tk.Label(root,
                            text="Image Processing dengan Color Moment, GLCM, dan KNN",
                            font=("Arial", 10),
                            bg='#2c3e50',
                            fg='#ecf0f1')
    subtitle_label.pack(pady=5)
    
    # Menu frame
    menu_frame = tk.Frame(root, bg='#34495e', padx=20, pady=20)
    menu_frame.pack(pady=20, padx=40, fill=tk.BOTH, expand=True)
    
    # Function to run training
    def run_training():
        root.destroy()
        import train_model
        train_model.train_knn_model()
        input("\nTekan Enter untuk kembali ke menu utama...")
        main_menu()
    
    # Function to run prediction
    def run_prediction():
        root.destroy()
        import predict_image
        predict_image.main()
    
    # Function to show about
    def show_about():
        about_text = """
        SISTEM DETEKSI PENYAKIT MULUT DAN KUKU (PMK) PADA SAPI
        
        Fitur:
        1. Preprocessing: Resize gambar
        2. Feature Extraction: 
           - Color Moments (mean, std, skewness)
           - GLCM (energy, contrast, correlation, dll)
        3. Klasifikasi: K-Nearest Neighbors (KNN)
        
        Langkah penggunaan:
        1. Siapkan dataset di folder dataset/
        2. Train model dengan pilihan 1
        3. Prediksi gambar dengan pilihan 2
        
        Folder structure:
        - dataset/healthy/   : Gambar sapi sehat
        - dataset/sick/      : Gambar sapi sakit
        - features/          : File CSV fitur
        - models/            : Model machine learning
        - results/           : Hasil prediksi & analisis
        """
        messagebox.showinfo("Tentang Sistem", about_text)
    
    # Buttons
    btn_style = {
        'font': ("Arial", 11, "bold"),
        'width': 25,
        'pady': 10,
        'cursor': "hand2"
    }
    
    # Button 1: Train Model
    train_btn = tk.Button(menu_frame, 
                         text="1. 🏋️ TRAIN MODEL", 
                         command=run_training,
                         bg='#3498db', fg='white',
                         **btn_style)
    train_btn.pack(pady=10)
    
    # Button 2: Predict Image
    predict_btn = tk.Button(menu_frame, 
                           text="2. 🔍 PREDIKSI GAMBAR", 
                           command=run_prediction,
                           bg='#2ecc71', fg='white',
                           **btn_style)
    predict_btn.pack(pady=10)
    
    # Button 3: Check Structure
    def check_structure():
        dirs = create_directories()
        messagebox.showinfo("Struktur Folder", 
                          f"Folder yang tersedia:\n\n" + 
                          "\n".join([f"✓ {d}" for d in dirs]))
    
    structure_btn = tk.Button(menu_frame, 
                            text="3. 📁 CEK STRUKTUR", 
                            command=check_structure,
                            bg='#f39c12', fg='white',
                            **btn_style)
    structure_btn.pack(pady=10)
    
    # Button 4: About
    about_btn = tk.Button(menu_frame, 
                         text="4. ℹ️ TENTANG", 
                         command=show_about,
                         bg='#9b59b6', fg='white',
                         **btn_style)
    about_btn.pack(pady=10)
    
    # Button 5: Exit
    exit_btn = tk.Button(menu_frame, 
                        text="5. 🚪 KELUAR", 
                        command=root.destroy,
                        bg='#e74c3c', fg='white',
                        **btn_style)
    exit_btn.pack(pady=10)
    
    # Footer
    footer_label = tk.Label(root,
                           text="© 2024 Sistem Deteksi PMK - Computer Vision",
                           font=("Arial", 8),
                           bg='#2c3e50',
                           fg='#bdc3c7')
    footer_label.pack(side=tk.BOTTOM, pady=10)
    
    root.mainloop()

if __name__ == "__main__":
    # Create directories
    create_directories()
    
    # Show main menu
    main_menu()