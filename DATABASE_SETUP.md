# Database Setup untuk Deteksi PMK

## Persyaratan
- MySQL Server berjalan (versi 5.7 atau lebih tinggi)
- Python 3.12.5 dengan semua dependencies terinstall

## Langkah Setup

### 1. Install MySQL Server (Jika belum)

#### Windows:
- Download dari https://dev.mysql.com/downloads/mysql/
- Ikuti installer wizard
- Default Port: 3306
- Default User: root

#### Linux (Ubuntu/Debian):
```bash
sudo apt-get install mysql-server
```

#### macOS:
```bash
brew install mysql
brew services start mysql
```

### 2. Configure Environment Variables

Copy `.env.example` ke `.env` dan update konfigurasi:

```bash
cp .env.example .env
```

Edit `.env` dengan kredensial MySQL Anda:

```env
# MySQL Database Configuration
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password_here
DB_NAME=deteksi_pmk

# Flask Configuration
FLASK_SECRET=your-secret-key-here
```

**Penting:**
- `DB_HOST`: Alamat server MySQL (default: localhost)
- `DB_USER`: Username MySQL (default: root)
- `DB_PASSWORD`: Password MySQL (kosongkan jika tidak ada password)
- `DB_NAME`: Nama database yang akan dibuat

### 3. Setup Database

Jalankan script setup:

```bash
python setup_db.py
```

Script ini akan:
- Membuat database `deteksi_pmk`
- Membuat tabel-tabel yang diperlukan:
  - `predictions` - Menyimpan hasil prediksi
  - `diagnosis_history` - Menyimpan hasil diagnosis

### 4. Verify Database Connection

Anda bisa test koneksi dengan menjalankan:

```bash
python -c "from utils.mysql_db import get_engine; engine = get_engine(); print('✓ Database connected successfully!')"
```

### 5. Jalankan Aplikasi

```bash
python app.py
```

Aplikasi akan berjalan di: http://localhost:5000

## Database Schema

### Tabel: predictions

| Column | Type | Description |
|--------|------|-------------|
| id | INT PRIMARY KEY AUTO_INCREMENT | ID unik prediksi |
| original_filename | VARCHAR(255) | Nama file asli yang diupload |
| filename | VARCHAR(255) | Nama file yang disimpan |
| image_path | VARCHAR(500) | Path lengkap file |
| prediction | VARCHAR(50) | Hasil prediksi ('sehat' atau 'sakit') |
| confidence | FLOAT | Tingkat kepercayaan (0-100) |
| features | TEXT | JSON string dari fitur-fitur |
| timestamp | DATETIME | Waktu prediksi |

### Tabel: diagnosis_history

| Column | Type | Description |
|--------|------|-------------|
| id | INT PRIMARY KEY AUTO_INCREMENT | ID unik diagnosis |
| original_filename | VARCHAR(255) | Nama file asli yang diupload |
| filename | VARCHAR(255) | Nama file yang disimpan |
| image_path | VARCHAR(500) | Path lengkap file |
| diagnosis | TEXT | JSON string dari diagnosis details |
| severity | VARCHAR(50) | Tingkat keparahan ('ringan', 'sedang', 'berat') |
| confidence | FLOAT | Tingkat kepercayaan (0-100) |
| timestamp | DATETIME | Waktu diagnosis |

## Troubleshooting

### "MySQL not available at startup"
- Pastikan MySQL Server berjalan
- Periksa konfigurasi di file `.env`
- Pastikan password MySQL benar

### "Cannot import 'setuptools.build_meta'"
- Jalankan: `pip install --upgrade setuptools`

### "NumPy 2.x incompatibility"
- Sudah fixed dengan numpy<2 di requirements.txt
- Jalankan: `pip install "numpy<2"`

### Database tidak tersimpan di MySQL
- Periksa apakah MYSQL_AVAILABLE = True di terminal saat startup
- Cek file `.env` konfigurasi
- Jalankan `python setup_db.py` lagi untuk pastikan database terbuat

## Features yang Tersedia

### 1. Prediksi dengan Machine Learning
- Upload gambar (JPG, PNG, BMP)
- Deteksi PMK otomatis
- Confidence score tinggi

### 2. Riwayat Deteksi
- Lihat semua hasil deteksi yang tersimpan di database
- Filter berdasarkan hasil (Sehat/Sakit)
- Statistik real-time

### 3. Detail Riwayat
- Klik "Detail" di riwayat untuk melihat:
  - Hasil lengkap prediksi
  - Confidence score
  - Fitur-fitur yang dianalisis
  - Informasi file

### 4. Diagnosis Expertise System
- Berbasis rule-based system
- Memberikan diagnosis detail berdasarkan gejala
- Simpan hasil diagnosis ke database

## Data Storage Priority

1. **MySQL Database** (Utama)
   - Real-time
   - Queryable
   - Backup terstruktur

2. **CSV Files** (Backup)
   - Fallback jika MySQL error
   - Di folder `results/`

## Tips

- Pastikan MySQL Server running sebelum startup aplikasi
- Reguler backup database MySQL Anda
- Gunakan password yang kuat untuk MySQL di production
- Ubah FLASK_SECRET di `.env` untuk production

## Support

Jika ada error, cek:
1. `setup_db.py` untuk initialize database
2. File `.env` konfigurasi
3. MySQL Server status
4. Requirements sudah terinstall dengan benar
