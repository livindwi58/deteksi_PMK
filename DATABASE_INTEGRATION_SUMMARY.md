# MySQL Database Integration - Deteksi PMK

## ✅ Apa yang Sudah Diimplementasikan

### 1. **MySQL Database Integration** 
File: `utils/mysql_db.py`

Fitur:
- Simpan hasil prediksi ke database MySQL
- Simpan hasil diagnosis dari expert system
- Query data dari database
- JSON storage untuk fitur-fitur ekstraksi

Tabel Database:
- `predictions` - Menyimpan semua hasil prediksi dengan fitur-fitur
- `diagnosis_history` - Menyimpan hasil diagnosis dari sistem pakar

### 2. **Environment Configuration**
File: `.env`

Konfigurasi MySQL:
```
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=
DB_NAME=deteksi_pmk
```

Gunakan file `.env.example` sebagai template.

### 3. **Automatic Database Setup**
File: `setup_db.py`

Menjalankan:
```bash
python setup_db.py
```

Fitur:
- Membuat database otomatis
- Membuat tabel-tabel yang diperlukan
- Validasi koneksi

### 4. **Update Preference: MySQL First**

Priority penyimpanan data:
1. **MySQL Database** (Utama) - Real-time, queryable
2. **CSV Files** (Backup) - Fallback jika MySQL error

Priority load data:
1. **MySQL Database** (Cepat, terstruktur)
2. **CSV Files** (Fallback)

### 5. **Detail Riwayat Deteksi Page**
File: `templates/detail_deteksi.html`
Route: `/detail-deteksi/<id>`

Fitur:
- Lihat hasil lengkap satu prediksi
- Tampilkan semua fitur yang dianalisis
- Confidence score dengan progress bar
- Informasi file dan timestamp
- Link navigasi kembali

### 6. **Updated Riwayat Deteksi Page**
File: `templates/riwayat_deteksi.html`

Updated:
- Tombol "Detail" untuk setiap item riwayat
- Link ke halaman detail prediksi
- Priority load dari database

## 🚀 Cara Menggunakan

### Setup Pertama Kali

1. **Install dependencies** (jika belum):
```bash
pip install -r requirements.txt
```

2. **Setup database**:
```bash
python setup_db.py
```

3. **Jalankan aplikasi**:
```bash
python app.py
```

4. **Akses aplikasi**:
```
http://localhost:5000
```

### Workflow Penggunaan

1. **Upload Gambar** 
   - Ke halaman "Upload Gambar"
   - Hasil langsung disimpan ke MySQL database

2. **Lihat Riwayat**
   - Halaman "Riwayat Deteksi"
   - Menampilkan 10 deteksi terbaru dari database
   - Statistik otomatis dari data MySQL

3. **Detail Riwayat**
   - Klik "Detail" di setiap item
   - Lihat hasil lengkap prediksi
   - Lihat fitur-fitur yang dianalisis

## 📊 Data Structure

### Predictions Table
```
id | original_filename | filename | image_path | prediction | confidence | features (JSON) | timestamp
```

Features JSON example:
```json
{
  "area": 12345.67,
  "perimeter": 456.78,
  "circularity": 0.8765,
  "solidity": 0.9123,
  ...
}
```

### Diagnosis History Table
```
id | original_filename | filename | image_path | diagnosis (JSON) | severity | confidence | timestamp
```

## 🔧 Configuration

Edit `.env` untuk customize:
- MySQL host, port, user, password
- Database name
- Flask secret key

Default configuration sudah optimal untuk development.

## 🧪 Testing

Cek koneksi database:
```bash
python test_env.py
```

## 📝 Files yang Ditambahkan/Diubah

### Ditambahkan:
- `utils/mysql_db.py` - MySQL operations
- `setup_db.py` - Database initialization
- `test_env.py` - Environment test
- `.env` - Environment variables  
- `.env.example` - Environment template
- `DATABASE_SETUP.md` - Setup documentation
- `templates/detail_deteksi.html` - Detail page

### Diubah:
- `app.py` - Add dotenv loading, update MySQL imports, add detail route
- `requirements.txt` - Add python-dotenv
- `templates/riwayat_deteksi.html` - Update detail links

## ⚠️ Troubleshooting

### Database connection failed?
1. Pastikan MySQL server running
2. Cek `.env` configuration
3. Jalankan `python setup_db.py` lagi

### MYSQL_AVAILABLE = False?
Fallback ke CSV storage. Periksa terminal output untuk error detail.

### Data tidak ada di database?
- Pastikan database sudah di-setup dengan `setup_db.py`
- Cek MySQL server status
- Refresh halaman aplikasi

## 🎯 Next Steps (Optional)

1. **Backup automation** - Backup database secara berkala
2. **Export data** - Export hasil riwayat ke Excel/PDF
3. **Advanced analytics** - Dashboard dengan chart dan statistik
4. **User authentication** - Login system untuk multi-user
5. **API endpoints** - REST API untuk integrasi dengan sistem lain

## 📖 Documentation

- `DATABASE_SETUP.md` - Panduan setup database lengkap
- `app.py` - Kode aplikasi dengan comments
- `utils/mysql_db.py` - Dokumentasi function database

---

**Aplikasi siap! MySQL database sudah integrated dengan priority utama. Data akan otomatis tersimpan ke database saat upload gambar.**

Untuk pertanyaan lebih lanjut, cek dokumentasi atau jalankan `python setup_db.py` untuk re-initialize database.
