# AGENTS.md — Panduan AI untuk Sistem Deteksi PMK

File ini adalah **satu-satunya sumber kebenaran** untuk AI agent yang bekerja pada project ini. **WAJIB diperbarui** setiap kali ada perubahan arsitektur, dependensi, struktur database, atau alur sistem.

---

## 1. GAMBARAN PROYEK

**Sistem Deteksi dan Diagnosis Penyakit Mulut dan Kuku (PMK / FMD) pada Sapi**

- **Bahasa:** Python 3.12.5
- **Framework Web:** Flask (Bootstrap 5 UI)
- **Desktop (Legacy):** Tkinter (`main.py`, `predict_image.py`)
- **Deployment:** Railway (Nixpacks + gunicorn)
- **Repository:** `https://github.com/livindra/deteksi_PMK`

Dua jalur utama:

1. **Deteksi Berbasis Gambar** — ML (KNN) multi-class (sehat + tiap jenis PMK) + Computer Vision (color moments + GLCM texture)
2. **Sistem Pakar** — Forward chaining inference engine (29 gejala, 5 penyakit, 5 aturan)

> **Dataset:** Folder `dataset/` berisi `healthy/` dan folder `pmk_*` per jenis penyakit.
> `train_model.py` otomatis mendeteksi semua folder berawalan `pmk_*` sebagai kelas terpisah.

---

## 2. STRUKTUR PROYEK

```
deteksi_PMK/
├── app.py                     # Main Flask app (~1000 baris) — routing, prediksi, sistem pakar
├── expert_system.py           # Forward chaining engine + KnowledgeBase + Evaluator
├── train_model.py             # Training KNN model (k=5, euclidean, uniform)
├── setup_db.py                # Inisialisasi database MySQL
├── main.py                    # LEGACY — Tkinter desktop menu
├── predict_image.py           # LEGACY — Tkinter prediction GUI
├── test_forward_chaining.py   # Unit test expert system
├── test_env.py                # Test environment variables & DB connection
├── show_scaler_params.py      # Utility: lihat parameter scaler
├── requirements.txt           # Python dependencies
├── Procfile                   # gunicorn app:app
├── railway.json               # Railway Nixpacks builder config
├── .env                       # Environment variables (TIDAK DI-COMMIT)
├── .env.example               # Template env
├── .gitignore
├── .python-version            # 3.12.5
│
├── QUICK_START.md             # Panduan cepat (Bahasa Indonesia)
├── AGENTS.md                  # ← FILE INI — panduan AI, HARUS DIUPDATE
│
├── dataset/                   # Dataset gambar training
│   ├── healthy/               #   ~200 gambar sapi sehat
│   ├── pmk_oral/              #   Gambar PMK oral
│   ├── pmk_podal/             #   Gambar PMK podal (kaki)
│   ├── pmk_laktasi/           #   Gambar PMK laktasi (ambing)
│   └── pmk_akut_general/      #   Gambar PMK akut general
│
├── features/                  # CSV fitur hasil ekstraksi
│   ├── dataset.csv
│   ├── data_train.csv
│   └── data_test.csv
│
├── models/                    # Model terlatih (.pkl)
│   ├── knn_model.pkl          # BINARY — KNeighborsClassifier (sehat/sakit)
│   ├── scaler.pkl             # BINARY — StandardScaler
│   ├── label_encoder.pkl      # BINARY — LabelEncoder
│   ├── multiclass_knn_model.pkl    # MULTI — KNeighborsClassifier (jenis PMK)
│   ├── multiclass_scaler.pkl       # MULTI — StandardScaler
│   ├── multiclass_label_encoder.pkl# MULTI — LabelEncoder
│   └── pca.pkl                # TIDAK DIGUNAKAN (legacy)
│
├── uploads/                   # Gambar hasil upload user
│   └── resize/                # Hasil resize
│   └── threshold/             # Hasil threshold
│
├── utils/
│   ├── __init__.py
│   ├── mysql_db.py            # SQLAlchemy ORM — models, CRUD, seed data (916 baris)
│   ├── preprocessing.py       # Validasi sapi + preprocessing pipeline (340 baris)
│   ├── feature_extraction.py   # Ekstraksi fitur: RGB avg + HSV + GLCM + histogram + Hu (44 fitur)
│   └── helpers.py             # Load/save model, dataset prep, confidence estimation (237 baris)
│
├── templates/                 # Jinja2 templates
│   ├── base.html              # Layout utama (Bootstrap 5, navbar, footer)
│   ├── index.html             # Halaman utama
│   ├── upload.html            # Form upload multi-file (JS sederhana)
│   ├── result.html            # Hasil deteksi (sehat/sakit)
│   ├── riwayat_deteksi.html   # Riwayat (client localStorage + MySQL)
│   ├── detail_deteksi.html    # Detail satu prediksi + diagnosis
│   ├── expert_system.html     # Sistem pakar — pilih gejala + hasil
│   └── diagnosis_history.html # Riwayat diagnosis sistem pakar
│
└── static/
    ├── css/style.css          # Custom CSS (gradient, animasi, responsive)
    └── js/main.js             # JS — tooltips, file preview, API fetch, dark mode
```

---

## 3. ARSITEKTUR & ALUR DATA

### 3.1 Jalur Deteksi Gambar

```
User upload gambar
  │
  ▼
[app.py] validate_cattle_image()
  ├─ Tolak wajah manusia (Haar cascade)
  ├─ Deteksi mata (HoughCircles) — bonus scoring
  ├─ Validasi warna HSV (coklat, merah, hitam, putih)
  ├─ Analisis tekstur (Laplacian variance)
  ├─ Edge detection (Canny)
  └─ Final: weighted confidence (eye 15%, color 40%, texture 30%, edge 15%), threshold ≥ 65%
  │
  ├─ Gagal → tolak dengan pesan error
  │
  ▼
[preprocessing.py] preprocess_pipeline()
  → resize 128×128 → grayscale → Otsu threshold → mask → masked RGB + masked gray
  │
  ▼
[feature_extraction.py] FeatureExtractor.extract_all_features()
  → 44 fitur: [RGB avg (3), HSV mean+std (6), GLCM (4), histogram 24, Hu (7)]
  │
  ▼
[helpers.py] scaler.transform() → [BINARY MODEL] predict()
  │
  ├  ─ 'sehat' → Simpan ke MySQL
  │
  ▼ sakit
[helpers.py] multiclass_scaler.transform() → [MULTI-CLASS MODEL] predict()
  → label_encoder.inverse_transform()
  │
  ▼
[helpers.py] estimate_prediction_confidence()
  → weighted neighbor confidence (clipped 50.0 – 98.5)
  │
  ▼
Simpan ke MySQL (primary) + CSV (fallback)
  │
  ▼
Semua gambar diproses — jika ada yang sakit:
  → Redirect ke expert-system?symptoms=G01,G02,...&mode=image
  → Gejala otomatis tercentang sesuai jenis PMK — hanya gejala yang relevan dengan body part PMK type tersebut (tidak cross-category)
Jika semua sehat:
  → Flash "Semua X gambar sehat" → /result/healthy
```

### 3.2 Jalur Sistem Pakar

```
User memilih gejala di web form
  │
  ▼
[expert_system.py] ForwardChaining.tambah_gejala(gejala_list)
  │
  ▼
ForwardChaining.inferensi()
  → Untuk setiap aturan:
    - Hitung coverage = matched_count / total_count
    - Hitung support_ratio = matched_rules / total_rules_for_disease
    - Hitung evidence_strength = min(best_matched / 4, 1.0)
    - Combined score = best_coverage*0.5 + avg_coverage*0.25 + support_ratio*0.15 + evidence_strength*0.1 + exact_bonus
    - Threshold minimum: 0.35
  │
  ▼
ForwardChaining.get_diagnosis()
  → Urutkan penyakit berdasarkan score
  → Kembalikan top diagnosis + matched rules + solusi
  → Severity map: P01=oral, P02=podal, P03=laktasi, P05=akut umum
  │
  ▼
Simpan ke MySQL (tabel predictions + diagnosis_history)
  │
  ▼
Render expert_system.html
```

### 3.3 Storage Hybrid

- **MySQL** (primary): tabel `predictions`, `diagnosis_history`, `expert_symptoms`, `expert_diseases`, `expert_rules`, join table `expert_rules_expert_symptoms`
- **localStorage** (browser): menyimpan array ID prediksi untuk ditampilkan di halaman riwayat
- **CSV fallback**: `results/predictions.csv`, `results/diagnosis_history.csv`

> **Batch upload:** Semua gambar dari satu sesi upload disimpan dalam **1 baris** tabel `predictions`. Data per-gambar (filename, prediction, pmk_type, confidence) disimpan sebagai JSON di kolom `images_data`. Field `prediction` diisi 'sakit' jika ada gambar sakit, 'sehat' jika semua sehat.

---

## 4. DATABASE (MySQL via SQLAlchemy)

### 4.1 Tabel

```sql
-- Tabel prediksi hasil deteksi gambar
predictions
  id              INT AUTO_INCREMENT PK
  original_filename VARCHAR(255)
  filename        VARCHAR(255)
  image_path      VARCHAR(500)
  prediction      VARCHAR(50)        -- 'sehat' / 'sakit'
  confidence      FLOAT
  features        TEXT               -- JSON string dari extracted features
  images_data     TEXT               -- JSON array dari per-gambar (batch upload)
  timestamp       DATETIME

-- Tabel riwayat diagnosis sistem pakar
diagnosis_history
  id              INT AUTO_INCREMENT PK
  prediction_id   INT FK → predictions.id (nullable)
  diagnosis       TEXT               -- JSON string detail diagnosis
  severity        VARCHAR(50)        -- 'ringan' / 'sedang' / 'berat'
  timestamp       DATETIME

-- Master gejala (29 gejala)
expert_symptoms
  id              INT AUTO_INCREMENT PK
  code            VARCHAR(10) UNIQUE -- G01–G29
  description     TEXT
  category        VARCHAR(50)        -- umum, mulut, kaki, ambing, berat
  display_order   INT

-- Master penyakit (5 penyakit)
expert_diseases
  id              INT AUTO_INCREMENT PK
  code            VARCHAR(10) UNIQUE -- P01–P05
  name            VARCHAR(120)
  description     TEXT
  solutions       TEXT               -- JSON array of strings
  display_order   INT

-- Aturan forward chaining (5 aturan)
expert_rules
  id              INT AUTO_INCREMENT PK
  code            VARCHAR(20) UNIQUE -- FC01–FC04
  symptom_codes   TEXT               -- JSON array of symptom codes
  result_disease_code VARCHAR(10) FK → expert_diseases.code
  description     TEXT
  is_active       BOOLEAN
  display_order   INT

-- Join table many-to-many
expert_rules_expert_symptoms
  rule_id         INT FK → expert_rules.id (ON DELETE CASCADE)
  symptom_id      INT FK → expert_symptoms.id (ON DELETE CASCADE)
```

### 4.2 Knowledge Base Default

**29 Gejala (G01–G29)** per kategori:

- **umum** (7): G01 demam >39.5°C, G11 nafsu makan turun, G12 lesu, G17 demam >40.5°C, G25 lebih sering berbaring, G28 takikardia, G29 sesak napas
- **mulut** (9): G02 air liur berlebih, G03 luka lidah/gusi, G04 nyeri setelah lepuh, G10 hidung berair, G14 lepuh moncong, G18 lepuh lidah meluas, G19 sulit mengunyah, G20 lepuh bantalan gigi, G21 bau mulut
- **kaki** (6): G05 lepuh celah kuku, G06 pincang, G15 nyeri kaki, G22 edema/radang, G23 bengkak celah kuku, G24 telapak kaki longgar
- **ambing** (5): G07 lepuh puting, G08 lesi puting, G09 produksi susu turun, G26 puting retak, G27 susu menggumpal
- **berat** (2): G13 miokarditis/kematian mendadak, G16 abortus/infertilitas

**5 Penyakit (P01–P05):**

- P01 = PMK_ORAL (gejala mulut)
- P02 = PMK_PODAL (gejala kaki/kuku)
- P03 = PMK_LAKTASI (gejala ambing)
- P05 = PMK_AKUT_GENERAL (gejala umum berat)

**4 Aturan (FC01–FC04):**

- FC01 → P01 (oral): [G01, G02, G03, G04, G11, G18, G19, G20, G21]
- FC02 → P02 (podal): [G01, G02, G05, G06, G15, G22, G23, G24, G25]
- FC03 → P03 (laktasi): [G01, G02, G07, G08, G09, G26, G27]
- FC04 → P05 (akut umum): [G01, G02, G03, G04, G05, G06, G07, G09, G11, G12, G14, G18, G20, G22, G23, G24, G26]

> **Auto-check gejala dari hasil deteksi gambar** (di `app.py:pmk_to_symptoms`):
> Mapping ini digunakan saat redirect ke expert system via mode=image. Hanya gejala spesifik body part yang diikutkan (tanpa gejala umum):
> - `pmk_oral` → gejala mulut: [G02, G03, G04, G10, G14, G18, G19, G20, G21]
> - `pmk_podal` → gejala kaki: [G05, G06, G15, G22, G23, G24]

> **Jika menambah/mengubah gejala, penyakit, atau aturan**, update data di:
>
> 1. `utils/mysql_db.py` — constants `DEFAULT_EXPERT_SYMPTOMS`, `DEFAULT_EXPERT_DISEASES`, `DEFAULT_EXPERT_RULES`
> 2. File ini (AGENTS.md) — bagian ini
> 3. Jalankan ulang `seed_expert_knowledge(force=True)` atau hapus tabel agar di-reseed

---

## 5. API ENDPOINTS (Flask)

| Method   | Route                            | Deskripsi                                          |
| -------- | -------------------------------- | -------------------------------------------------- |
| GET      | `/`                              | Homepage — model status + recent history           |
| GET      | `/upload`                        | Halaman upload gambar                              |
| POST     | `/predict`                       | Upload + prediksi gambar multi-file. Semua gambar disimpan sbg 1 baris di DB. Jika sakit → redirect ke expert-system dgn gejala otomatis + semua gambar ditampilkan. Jika sehat → /result/healthy |
| GET      | `/result/healthy`                | Hasil sehat                                        |
| GET      | `/result/sick`                   | Hasil sakit (legacy, tidak dipakai di alur baru)   |
| GET      | `/riwayat_deteksi`               | Halaman riwayat deteksi                            |
| GET      | `/detail-deteksi/<int:pred_id>`  | Detail prediksi + diagnosis                        |
| POST     | `/api/validate-image`            | Validasi gambar real-time (apakah sapi)            |
| POST     | `/api/get_data_riwayat_deteksi`  | Ambil riwayat berdasarkan localStorage IDs         |
| POST     | `/api/diagnosis`                 | API diagnosis sistem pakar                         |
| GET/POST | `/expert-system`                 | Halaman sistem pakar (GET=form dgn param `symptoms` untuk pre-check, POST=diagnosis) |
| GET      | `/expert-system/from-prediction` | Redirect ke expert system mode gambar              |
| GET      | `/riwayat-diagnosis`             | Halaman riwayat diagnosis                          |
| GET      | `/export/csv`                    | Export prediksi ke CSV                             |
| GET      | `/export/excel`                  | Export prediksi ke Excel                           |
| GET      | `/clear-session`                 | Bersihkan Flask session                            |

---

## 6. ML MODEL DETAILS

**Arsitektur Hierarchical (2-level):**

```
Input image → 44 fitur (RGB+HSV+GLCM+Histogram+Hu)
                   │
            ┌──────┴──────┐
            ▼              ▼
      [BINARY MODEL]     [selamat — end]
      sehat vs sakit      
            │ (sakit)
            ▼
      [MULTI-CLASS MODEL]
      pmk_oral / pmk_podal / pmk_laktasi / pmk_akut_general
```

### Binary Model (prefix='')

| Parameter         | Value                            |
| ----------------- | -------------------------------- |
| **Task**          | sehat vs sakit                   |
| **Algorithm**     | K-Nearest Neighbors (KNN)        |
| **k**             | Hyperparameter tuned (1–15)      |
| **Distance**      | Hyperparameter tuned (euclidean/manhattan) |
| **Weight**        | Hyperparameter tuned (uniform/distance) |
| **Accuracy**      | ±98%                             |
| **Model file**    | `models/knn_model.pkl`           |

### Multi-class Model (prefix='multiclass_')

| Parameter         | Value                            |
| ----------------- | -------------------------------- |
| **Task**          | Jenis PMK (hanya data sakit)     |
| **Algorithm**     | K-Nearest Neighbors (KNN)        |
| **k**             | Hyperparameter tuned (1–15)      |
| **Distance**      | Hyperparameter tuned (euclidean/manhattan) |
| **Weight**        | Hyperparameter tuned (uniform/distance) |
| **Accuracy**      | ±60%                             |
| **Model file**    | `models/multiclass_knn_model.pkl`|

### Common

| Parameter         | Value                                                                       |
| ----------------- | --------------------------------------------------------------------------- |
| **Features**      | 44: avg_red, avg_green, avg_blue, hsv_h_mean, hsv_h_std, hsv_s_mean, hsv_s_std, hsv_v_mean, hsv_v_std, contrast, homogeneity, correlation, energy, 8-bin histogram × 3 channels (24), hu_moment_0..6 (7) |
| **Image size**      | 256×256                                                                     |
| **Preprocessing** | Otsu threshold, mask, resize                                                |
| **Scaling**       | StandardScaler                                                              |
| **Output**        | Binary: `sehat` / `sakit`; Multi-class: `pmk_oral`, `pmk_podal`, `pmk_laktasi`, `pmk_akut_general`, ... |
| **Confidence**    | Custom weighted neighbor, clipped 50.0–89.5%                                |
| **Class detection** | Auto-detect all folders under `dataset/` — `healthy` → `sehat`, `pmk_*` → nama folder |

> **Jika menambah/mengubah fitur**, update:
>
> 1. `utils/feature_extraction.py` — `FeatureExtractor.feature_names`
> 2. `train_model.py` — feature export dimensions
> 3. Retrain model (`python train_model.py`)
> 4. AGENTS.md ini
>
> **Jika menambah kelas penyakit baru** (folder `pmk_*`):
>
> 1. Buat folder `dataset/pmk_nama_baru/` — otomatis terdeteksi oleh `prepare_dataset()`
> 2. Retrain model (`python train_model.py`)
> 3. Update AGENTS.md

---

## 7. CATTLE IMAGE VALIDATION

Lokasi: `utils/preprocessing.py:validate_cattle_image()`

Layer validasi berlapis:

1. **Face rejection** — Haar cascade face detection (tolak foto manusia)
2. **Eye detection** — HoughCircles (bonus scoring)
3. **Color validation** — HSV threshold (brown, red, black, white)
4. **Texture analysis** — Laplacian variance (range 45–3500)
5. **Edge detection** — Canny edge density (range 0.01–0.35)

Final confidence = `eye_score*0.15 + color_score*0.40 + texture_score*0.30 + edge_score*0.15`
Threshold: **≥ 65%** untuk lolos sebagai sapi.

---

## 8. DEPLOYMENT

### Railway

- **Builder:** Nixpacks (`railway.json`)
- **Start:** `gunicorn app:app --bind 0.0.0.0:$PORT`
- **Env:** `FLASK_SECRET`, `DATABASE_URL` atau `MYSQLHOST/PORT/USER/PASSWORD/DATABASE`
- `opencv-python-headless` — tidak butuh GUI library

### Local

```bash
# 1. Setup database
python setup_db.py

# 2. (Opsional) Training model
python train_model.py

# 3. Jalankan app
python app.py
# Buka http://localhost:5000
```

---

## 9. TESTING

```bash
# Test environment & database
python test_env.py

# Test expert system forward chaining
python test_forward_chaining.py
```

> **Jika menambah fitur baru**, tambahkan test case di file terkait. Jika menambah test file baru, update bagian ini.

---

## 10. CONFIGURATION

### Environment Variables (`.env`)

```
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=
DB_NAME=deteksi_pmk
MYSQLHOST=          # Railway
MYSQLPORT=          # Railway
MYSQLUSER=          # Railway
MYSQLPASSWORD=      # Railway
MYSQLDATABASE=      # Railway
DATABASE_URL=       # Railway
FLASK_SECRET=deteksi-pmk-secret-key-2026
FLASK_DEBUG=0
```

---

## 11. ATURAN UNTUK AI AGENT

1. **Jangan pernah mengubah fungsionalitas tanpa memperbarui file ini.**
2. **Jika menambah variabel environment**, tambahkan ke `.env.example` dan dokumentasikan di sini.
3. **Jika mengubah struktur tabel**, update: (a) model SQLAlchemy di `utils/mysql_db.py`, (b) skema di AGENTS.md, (c) constants `DEFAULT_EXPERT_*`.
4. **Jika menambah endpoint**, dokumentasikan di bagian API ENDPOINTS.
5. **Jika mengubah fitur ML**, update `feature_names`, retrain model, update AGENTS.md.
6. **Jika menambah dependensi**, tambahkan ke `requirements.txt` dan update tabel dependensi.
7. **File ini WAJIB dibaca** sebelum melakukan perubahan signifikan.
8. **Gunakan Bahasa Indonesia** untuk komentar, UI, dan dokumentasi end-user karena target pengguna adalah peternak/inspektur Indonesia.

---

## 12. DEPENDENSI (requirements.txt)

| Package                | Version  | Kegunaan                          |
| ---------------------- | -------- | --------------------------------- |
| opencv-python-headless | 4.9.0.80 | Image processing, validasi sapi   |
| numpy                  | <2       | Numerical operations              |
| pandas                 | ≥2.1.0   | Data manipulation                 |
| scikit-learn           | ≥1.3.2   | KNN, StandardScaler, LabelEncoder |
| scikit-image           | ≥0.22.0  | GLCM texture features             |
| matplotlib             | ≥3.8.0   | Feature analysis plots            |
| pillow                 | ≥10.1.0  | Image loading fallback            |
| joblib                 | ≥1.3.2   | Model serialization               |
| Flask                  | ≥2.3.3   | Web framework                     |
| SQLAlchemy             | ≥2.0.0   | ORM MySQL                         |
| PyMySQL                | ≥1.1.0   | MySQL driver                      |
| python-dotenv          | ≥1.0.0   | Environment loading               |
| gunicorn               | ≥21.2.0  | WSGI production server            |
| pytz                   | ≥2024.1  | Timezone (Asia/Jakarta)           |
| cryptography           | ≥3.4.8   | MySQL password encryption         |

---

## 13. LEGACY CODE (TIDAK DIGUNAKAN LAGI)

File berikut adalah kode legacy desktop (Tkinter) yang **tidak aktif** di web app:

- `main.py` — Menu utama desktop
- `predict_image.py` — Prediksi gambar via GUI Tkinter
- `models/pca.pkl` — PCA model tidak digunakan (legacy)

> Jika ada refactoring besar, pertimbangkan untuk menghapus file-file ini atau tandai secara eksplisit.
