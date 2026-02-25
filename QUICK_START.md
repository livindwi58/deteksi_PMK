# 🚀 QUICK START - Database Integration

## Langkah 1: Setup Database (First Time Only)

Jalankan script setup database:
```bash
python setup_db.py
```

Output yang diharapkan:
```
✓ Database 'deteksi_pmk' created/verified
✓ Database tables created/verified
✓ Database setup completed successfully!
```

## Langkah 2: Jalankan Aplikasi

```bash
python app.py
```

Buka browser ke: **http://localhost:5000**

## Langkah 3: Upload Gambar & Lihat Riwayat

1. **Upload**: Klik "Upload Gambar" → Pilih file → Upload
2. **Hasil disimpan otomatis ke database MySQL**
3. **Lihat Riwayat**: Klik "Riwayat Deteksi"
4. **Detail**: Klik tombol "Detail" untuk melihat hasil lengkap

## 📋 Fitur Baru

✅ **MySQL Database** - Penyimpanan data terstruktur
✅ **Detail Riwayat** - Lihat hasil lengkap setiap deteksi
✅ **Feature Display** - Lihat fitur-fitur yang dianalisis
✅ **Automatic Backup** - CSV sebagai fallback
✅ **Real-time Stats** - Statistik dari database

## ⚙️ Konfigurasi

Edit `.env` jika ingin ubah MySQL settings:
```
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=
DB_NAME=deteksi_pmk
```

## 🔗 Useful Routes

| URL | Deskripsi |
|-----|-----------|
| `/` | Halaman Beranda |
| `/upload` | Upload Gambar |
| `/riwayat_deteksi` | Riwayat Deteksi (List) |
| `/detail-deteksi/1` | Detail Riwayat (ID=1) |
| `/expert-system` | Sistem Pakar |

## 🆘 Jika Ada Error

**Error: "MySQL not available"**
- Jalankan: `python setup_db.py`
- Pastikan MySQL server running

**Error: "Cannot connect to database"**
- Cek `.env` configuration
- Pastikan DB_HOST, DB_USER, DB_password benar

**Error: "Unknown database"**
- Jalankan: `python setup_db.py`

**Memory load error?**
- Restart Python: `python app.py`

## 📁 File Penting

- `app.py` - Main application
- `.env` - Database configuration
- `utils/mysql_db.py` - Database operations
- `setup_db.py` - Database setup script
- `templates/detail_deteksi.html` - Detail page

## ✨ Done!

Database integration sudah selesai. Aplikasi siap digunakan dengan penyimpanan data MySQL yang aman dan terstruktur.

---

Untuk dokumentasi lengkap, lihat:
- `DATABASE_SETUP.md` - Setup guide lengkap
- `DATABASE_INTEGRATION_SUMMARY.md` - Summary fitur baru
