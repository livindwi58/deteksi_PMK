from flask import Flask, render_template, request, redirect, url_for, send_from_directory, send_file, flash, session, jsonify
import io
import datetime
import os
from werkzeug.utils import secure_filename
import pandas as pd
import uuid
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from utils.helpers import load_model, estimate_prediction_confidence
from utils.preprocessing import preprocess_image, preprocess_pipeline, validate_cattle_image
from utils.feature_extraction import FeatureExtractor
from expert_system import ForwardChaining, KnowledgeBase  # Import sistem pakar

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


app = Flask(__name__, 
           template_folder=os.path.join(BASE_DIR, 'templates'), 
           static_folder=os.path.join(BASE_DIR, 'static'))
app.secret_key = os.environ.get('FLASK_SECRET', 'change-me')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Inisialisasi sistem pakar dan knowledge base
expert_system = ForwardChaining()
kb = KnowledgeBase()

# Jinja filter untuk mendapatkan deskripsi gejala dari kode
@app.template_filter('get_symptom_desc')
def get_symptom_desc(symptom_code):
    """Convert symptom code to description"""
    return kb.gejala.get(symptom_code, symptom_code)


try:
    from utils.mysql_db import (
        save_prediction_mysql,
        get_recent_predictions_mysql,
        get_prediction_by_id,
        init_mysql_tables,
        save_diagnosis_mysql,
        get_diagnosis_history_mysql,
        get_diagnosis_by_id,
        get_diagnosis_by_prediction_id,
        get_engine,
    )
  
    try:
        engine = get_engine()
        # quick connect test
        with engine.connect() as conn:
            # initialize tables if needed
            try:
                init_mysql_tables()
            except Exception as init_err:
               
                print(f"⚠️  Warning: Database tables initialization failed: {init_err}")
        print("✓ MySQL Database connected successfully")
        MYSQL_AVAILABLE = True
    except Exception as e:
        import traceback
        print('❌ MySQL not available at startup:')
        print(f"   Error: {e}")
        print("   Cek: 1) MySQL Server berjalan? 2) .env credentials benar? 3) Database 'deteksi_pmk' ada?")
        traceback.print_exc()
        MYSQL_AVAILABLE = False
except Exception as import_err:
    print(f"❌ Failed to import MySQL utilities: {import_err}")
    MYSQL_AVAILABLE = False

model = None
scaler = None
label_encoder = None
extractor = FeatureExtractor()
model_loading = False  
model_loaded = False   

def _load_model_background():
    global model, scaler, label_encoder, model_loading, model_loaded
    
    # Prevent double-loading
    if model_loading or model_loaded:
        return
    
    model_loading = True
    try:
        from utils.helpers import load_model
        print('[APP] Loading ML model in background...')
        m, s, le = load_model()
        model, scaler, label_encoder = m, s, le
        model_loaded = True
        print('[APP] Model loaded successfully.')
    except Exception as e:
        model = None
        scaler = None
        label_encoder = None
        model_loaded = True  # Mark as done to prevent retry loop
        print(f"[APP] ❌ Background model load failed: {e}")
    finally:
        model_loading = False

import threading
threading.Thread(target=_load_model_background, daemon=True).start()


def is_model_ready():
    """Check if model is loaded and ready to use"""
    return model is not None and scaler is not None and label_encoder is not None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    # provide safe context values so templates don't receive Undefined
    model_loaded = (model is not None and scaler is not None and label_encoder is not None)
    model_info = None
    try:
        # if your load_model provides info, adapt accordingly
        if model_loaded and hasattr(model, 'score'):
            model_info = {'akurasi': None}
    except Exception:
        model_info = None

    # Get recent predictions from MySQL database
    history = []
    if MYSQL_AVAILABLE:
        try:
            rows = get_recent_predictions_mysql(limit=5)
            if rows:
                history = rows
        except Exception as e:
            print(f"✗ Error reading recent predictions from MySQL: {e}")

    return render_template('index.html', 
                         model_loaded=bool(model_loaded), 
                         model_info=model_info, 
                         history=history)


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/riwayat_deteksi')
def riwayat_deteksi():
    # Halaman ditampilkan dulu, data diambil kemudian lewat API berdasarkan localStorage.
    return render_template(
        'riwayat_deteksi.html',
        recent=[],
        stats={'total': 0, 'positif': 0, 'negatif': 0, 'akurasi_rata_rata': 0.0},
        data_source='client'
    )


def _serialize_prediction_row(prediction_row):
    if not prediction_row:
        return None

    pred_id = prediction_row.get('id')
    prediction = str(prediction_row.get('prediction') or '').lower()
    source = str(prediction_row.get('source') or 'image_processing')
    diagnosis_label = prediction_row.get('diagnosis_label')
    confidence_value = prediction_row.get('confidence')
    confidence = float(confidence_value or 0.0)
    show_confidence = source != 'manual_expert_system'
    display_label = diagnosis_label if source == 'manual_expert_system' and diagnosis_label else ('Positif PMK' if prediction == 'sakit' else 'Sehat')

    return {
        'id': pred_id,
        'original_filename': prediction_row.get('original_filename') or '',
        'filename': prediction_row.get('filename') or '',
        'image_path': prediction_row.get('image_path') or '',
        'prediction': prediction,
        'source': source,
        'diagnosis_label': diagnosis_label,
        'display_label': display_label,
        'prediction_label': 'Positif PMK' if prediction == 'sakit' else 'Sehat',
        'confidence': round(confidence, 1) if show_confidence else None,
        'show_confidence': show_confidence,
        'timestamp': prediction_row.get('timestamp'),
        'detail_url': url_for('detail_deteksi', pred_id=pred_id),
        'image_url': url_for('uploaded_file', filename=prediction_row.get('filename') or '') if prediction_row.get('filename') else None,
    }


def _history_sort_timestamp(value):
    parsed = pd.to_datetime(value, errors='coerce')
    if pd.isna(parsed):
        return pd.Timestamp.min
    return parsed


def _history_sort_id(value):
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


@app.route('/get_data_riwayat_deteksi', methods=['POST'])
@app.route('/api/get_data_riwayat_deteksi', methods=['POST'])
def get_data_riwayat_deteksi():
    """Ambil data riwayat deteksi berdasarkan ID yang disimpan di localStorage."""
    if not MYSQL_AVAILABLE:
        return jsonify({
            'success': False,
            'message': 'Database tidak tersedia',
            'recent': [],
            'stats': {'total': 0, 'positif': 0, 'negatif': 0, 'akurasi_rata_rata': 0.0}
        }), 503

    payload = request.get_json(silent=True) or {}
    raw_ids = payload.get('ids', [])
    if isinstance(raw_ids, (str, int, float)):
        raw_ids = [raw_ids]

    normalized_ids = []
    for item in raw_ids if isinstance(raw_ids, list) else []:
        try:
            normalized_ids.append(int(item))
        except (TypeError, ValueError):
            continue

    ordered_ids = sorted(set(normalized_ids), reverse=True)

    recent = []
    total_confidence = 0.0
    confidence_count = 0
    positif = 0
    negatif = 0

    for pred_id in ordered_ids:
        try:
            row = get_prediction_by_id(pred_id)
        except Exception as e:
            print(f"✗ Error getting prediction {pred_id} from MySQL: {e}")
            row = None

        if not row:
            continue

        serialized = _serialize_prediction_row(row)
        if not serialized:
            continue

        recent.append(serialized)
        if serialized.get('show_confidence') and serialized.get('confidence') is not None:
            total_confidence += float(serialized.get('confidence') or 0.0)
            confidence_count += 1
        if serialized['prediction'] == 'sakit':
            positif += 1
        elif serialized['prediction'] == 'sehat':
            negatif += 1

    recent = sorted(
        recent,
        key=lambda item: (
            _history_sort_id(item.get('id')),
            _history_sort_timestamp(item.get('timestamp')),
        ),
        reverse=True,
    )

    total = len(recent)
    stats = {
        'total': total,
        'positif': positif,
        'negatif': negatif,
        'akurasi_rata_rata': round(total_confidence / confidence_count, 1) if confidence_count else 0.0,
    }

    return jsonify({
        'success': True,
        'recent': recent,
        'stats': stats,
        'message': 'Data riwayat berhasil dimuat' if recent else 'Tidak ada data riwayat pada localStorage'
    })


@app.route('/detail-deteksi/<int:pred_id>')
def detail_deteksi(pred_id):
    """Halaman detail riwayat deteksi"""
    prediction = None
    diagnosis = None
    
    # Get from MySQL database only
    if MYSQL_AVAILABLE:
        try:
            prediction = get_prediction_by_id(pred_id)
            if prediction:
                # Also get related diagnosis if exists
                diagnosis = get_diagnosis_by_prediction_id(pred_id)
                return render_template('detail_deteksi.html', 
                                     prediction=prediction, 
                                     diagnosis=diagnosis,
                                     data_source='mysql')
            else:
                print(f"✗ Prediction dengan ID {pred_id} tidak ditemukan di database")
        except Exception as e:
            import traceback
            print(f"✗ Error getting prediction from MySQL: {e}")
            traceback.print_exc()
    else:
        print("✗ MYSQL_AVAILABLE = False, database tidak terhubung saat startup")
        flash('⚠️ Database MySQL tidak tersedia. Pastikan:<br>'
              '1. MySQL Server sedang berjalan<br>'
              '2. Kredensial database di .env sudah benar<br>'
              '3. Jalankan: <code>python setup_db.py</code>', 'danger')
        return redirect(url_for('riwayat_deteksi'))
    
    # Not found or database error
    flash(f"❌ Prediksi dengan ID {pred_id} tidak ditemukan di database", 'danger')
    return redirect(url_for('riwayat_deteksi'))


@app.route('/upload')
def upload():
    """Halaman upload gambar"""
    return render_template('upload.html', error=None)


@app.route('/api/validate-image', methods=['POST'])
def api_validate_image():
    """API endpoint untuk validasi gambar real-time"""
    if 'image' not in request.files:
        return {'is_cattle': False, 'message': '❌ File tidak ditemukan'}, 400
    
    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return {'is_cattle': False, 'message': '❌ Format file tidak didukung (gunakan JPG/PNG/BMP)'}, 400
    
    temp_filepath = None
    try:
        # Simpan file temporary
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            temp_filepath = tmp.name
            file.save(temp_filepath)
        
        # Validasi gambar
        is_cattle, confidence, reason = validate_cattle_image(temp_filepath, confidence_threshold=0.75)
        
        if is_cattle:
            result = {
                'is_cattle': True,
                'message': f'✅ Gambar DITERIMA! Sapi terdeteksi dengan confidence {confidence*100:.0f}%',
                'confidence': float(confidence)
            }
        else:
            result = {
                'is_cattle': False,
                'message': reason or '❌ Ini bukan foto sapi',
                'confidence': float(confidence)
            }
        
        return result, 200
    
    except Exception as e:
        print(f"[VALIDATE API] Error: {e}")
        return {
            'is_cattle': False,
            'message': f'❌ Error saat validasi: {str(e)}'
        }, 500
    
    finally:
        # SELALU hapus temporary file
        if temp_filepath:
            try:
                if os.path.exists(temp_filepath):
                    os.remove(temp_filepath)
                    print(f"[VALIDATE API] ✓ Temporary file dihapus: {temp_filepath}")
            except Exception as e:
                print(f"[VALIDATE API] ⚠️ Error menghapus temp file: {e}")


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        flash('File tidak ditemukan', 'error')
        return redirect(url_for('index'))

    file = request.files['image']
    if file.filename == '':
        flash('Tidak ada file yang dipilih', 'error')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        # Generate unique filename to avoid conflicts
        original_filename = secure_filename(file.filename)
        ext = original_filename.rsplit('.', 1)[1].lower()
        filename = f"{uuid.uuid4()}.{ext}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # File sudah di-validasi di frontend (/api/validate-image)
        # Jangan validasi lagi, langsung lanjut ke diagnosis
        print(f"[PREDICT] ✅ File diterima dari frontend validation: {filename}")

        if not is_model_ready():
            # Model belum siap - HAPUS FILE
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    print(f"[PREDICT] ✓ File DIHAPUS (model belum ready): {filename}")
                except Exception as del_err:
                    print(f"[PREDICT] ⚠️ Error menghapus file: {del_err}")
            
            flash('Model belum tersedia. Jalankan training terlebih dahulu.', 'error')
            return redirect(url_for('index'))

        # Preprocess and extract
        img_rgb, gray_processed = preprocess_pipeline(filepath)
        features = extractor.extract_all_features(img_rgb, gray_processed)
        features_scaled = scaler.transform([features])

        pred_encoded = model.predict(features_scaled)[0]
        prediction = label_encoder.inverse_transform([pred_encoded])[0]
        confidence = estimate_prediction_confidence(model, features_scaled)
        if confidence is None:
            probabilities = model.predict_proba(features_scaled)[0]
            confidence = float(max(probabilities) * 100)

        # Prepare feature dictionary for database
        features_dict = {name: float(features[i]) for i, name in enumerate(extractor.feature_names)}
        
        # Try to save to MySQL first (primary storage)
        rowid = None
        if MYSQL_AVAILABLE:
            try:
                rowid = save_prediction_mysql(original_filename, filename, filepath, prediction, confidence, features_dict)
                print(f"✓ Saved prediction to MySQL, id={rowid}")
                # store DB id in session so result page can link to the new history row
                try:
                    session.setdefault('last_prediction', {})
                    session['last_prediction']['db_id'] = int(rowid) if rowid is not None else None
                except Exception:
                    pass
            except Exception as e:
                import traceback
                print(f"✗ Gagal menyimpan ke MySQL: {e}")
                traceback.print_exc()

        # Also save to CSV as fallback/backup (ensure consistent columns, migrate old files if needed)
        try:
            os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)
            data = {
                'image_path': [original_filename],  # Simpan nama asli
                'filename': [filename],  # Simpan nama unik
                'prediction': [prediction],
                'confidence': [confidence],
                'timestamp': [pd.Timestamp.now()]
            }
            for i, name in enumerate(extractor.feature_names):
                data[name] = [float(features[i])]

            df = pd.DataFrame(data)
            csv_path = os.path.join(BASE_DIR, 'results', 'predictions.csv')

            if os.path.exists(csv_path):
                # Check existing columns
                try:
                    existing_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
                except Exception:
                    existing_cols = []

                desired_cols = list(df.columns)

                # If existing file missing any desired columns, migrate by adding empty columns and re-saving
                if not set(desired_cols).issubset(set(existing_cols)):
                    try:
                        df_existing = pd.read_csv(csv_path)
                        for c in desired_cols:
                            if c not in df_existing.columns:
                                df_existing[c] = ''
                        # Reorder columns to desired order
                        df_existing = df_existing.reindex(columns=desired_cols)
                        df_existing.to_csv(csv_path, index=False)
                    except Exception as e:
                        print(f"Gagal memigrasi CSV lama: {e}")

                # Append new row without header
                df.to_csv(csv_path, mode='a', header=False, index=False)
            else:
                df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"Gagal menyimpan prediksi ke CSV: {e}")

        # Compact features table for session (round values to reduce size)
        features_table = list(zip(extractor.feature_names, [round(float(x), 4) for x in features]))

        # Simpan hasil prediksi ke session untuk digunakan di halaman result
        existing_db_id = None
        try:
            existing_db_id = session.get('last_prediction', {}).get('db_id')
        except Exception:
            existing_db_id = None

        session['last_prediction'] = {
            'filename': filename,
            'original_filename': original_filename,
            'prediction': prediction,
            'confidence': confidence,
            'features_table': features_table,
            'filepath': filepath,
            'source': 'image_processing',
            'db_id': existing_db_id
        }

        # Tentukan template berdasarkan hasil prediksi
        # Tambahkan jeda singkat agar tampilan hasil tidak muncul terlalu cepat
        if prediction.lower() == 'sakit':
            time.sleep(1.5)
            return redirect(url_for('result_sick'))
        else:
            time.sleep(1.5)
            return redirect(url_for('result_healthy'))

    flash('Tipe file tidak didukung', 'error')
    return redirect(url_for('index'))


@app.route('/result/healthy')
def result_healthy():
    """Halaman hasil untuk prediksi sehat"""
    if 'last_prediction' not in session:
        flash('Tidak ada hasil prediksi terbaru', 'error')
        return redirect(url_for('index'))
    
    pred = session['last_prediction']
    
    # Siapkan data untuk ditampilkan
    result = {
        'filename': pred['original_filename'],
        'saved_filename': pred['filename'],
        'prediction': 'Negatif PMK (Sehat)',
        'is_sick': False,
        'page_title': 'Hasil Deteksi - Sehat',
        'confidence': pred['confidence'] / 100.0,
        'expert_analysis': {
            'primary_conclusion': '',
            'confidence': 0.0,
            'symptoms_detected': [],
            'recommendations': [
                'Tetap jaga kebersihan kandang',
                'Berikan pakan bergizi',
                'Lakukan vaksinasi rutin',
                'Pantau kesehatan sapi secara berkala'
            ]
        }
    }
    
    # Dapatkan informasi file
    filepath = pred['filepath']
    img_norm, img_resized, _ = preprocess_image(filepath)
    
    result['timestamp'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    result['size'] = f"{os.path.getsize(filepath)/1024:.2f} KB"
    result['dimensions'] = f"{img_resized.shape[1]} x {img_resized.shape[0]} px"
    result['format'] = os.path.splitext(pred['filename'])[1].lstrip('.')
    
    return render_template('result.html', 
                         result=result, 
                         features_table=pred['features_table'])


@app.route('/result/sick')
def result_sick():
    """Halaman hasil untuk prediksi sakit dengan link ke sistem pakar"""
    if 'last_prediction' not in session:
        flash('Tidak ada hasil prediksi terbaru', 'error')
        return redirect(url_for('index'))
    
    pred = session['last_prediction']
    
    # Siapkan data untuk ditampilkan
    result = {
        'filename': pred['original_filename'],
        'saved_filename': pred['filename'],
        'prediction': 'Positif PMK (Terindikasi Sakit)',
        'is_sick': True,
        'page_title': 'Hasil Deteksi - Terindikasi Sakit',
        'confidence': pred['confidence'] / 100.0,
        'expert_analysis': {
            'primary_conclusion': 'Gambar menunjukkan indikasi infeksi PMK',
            'confidence': pred['confidence'] / 100.0,
            'symptoms_detected': [
                'Terdeteksi lesi pada area mulut/kaki',
                'Indikasi demam berdasarkan analisis visual',
                'Perubahan tekstur kulit terdeteksi'
            ],
            'recommendations': [
                'Segera isolasi sapi yang terindikasi sakit',
                'Lakukan konsultasi dengan sistem pakar untuk diagnosis lebih lanjut',
                'Bersihkan dan desinfeksi kandang',
                'Laporkan ke Dinas Peternakan setempat'
            ]
        }
    }
    
    # Dapatkan informasi file
    filepath = pred['filepath']
    img_norm, img_resized, _ = preprocess_image(filepath)
    
    result['timestamp'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    result['size'] = f"{os.path.getsize(filepath)/1024:.2f} KB"
    result['dimensions'] = f"{img_resized.shape[1]} x {img_resized.shape[0]} px"
    result['format'] = os.path.splitext(pred['filename'])[1].lstrip('.')
    
    return render_template('result.html', 
                         result=result, 
                         features_table=pred['features_table'])


@app.route('/expert-system', methods=['GET', 'POST'])
def expert_system_page():
    """Halaman sistem pakar forward chaining"""
    mode = request.form.get('mode') if request.method == 'POST' else request.args.get('mode', 'manual')
    mode = (mode or 'manual').strip().lower()
    use_image_context = mode != 'manual'

    if request.method == 'POST':
        # Dapatkan gejala yang dipilih
        gejala_terpilih = request.form.getlist('gejala')
        
        # Reset dan tambahkan gejala
        expert_system.reset()
        expert_system.tambah_gejala(gejala_terpilih)
        
        # Dapatkan diagnosis
        diagnosis = expert_system.get_diagnosis()

        prediction_id = session.get('last_prediction', {}).get('db_id') if use_image_context else None

        # Save diagnosis ke database. Jalur manual tidak lagi dipaksa terikat ke gambar.
        if MYSQL_AVAILABLE and diagnosis.get('status') == 'terdiagnosis' and diagnosis.get('diagnosis'):
            try:
                diag_list = diagnosis['diagnosis']
                main_diag = diag_list[0]
                
                # Ambil severity dari diagnosis result (sudah di-map di expert_system)
                severity = main_diag.get('severity', 'sedang')
                score = main_diag.get('score', 0)
                
                # Prepare diagnosis details dengan severity yang tepat
                diagnosis_details = {
                    'nama': main_diag.get('nama', ''),
                    'deskripsi': main_diag.get('deskripsi', ''),
                    'solusi': main_diag.get('solusi', []),
                    'score': float(score),
                    'gejala_teramati': main_diag.get('gejala_teramati', gejala_terpilih),
                    'semua_diagnosis': [
                        {
                            'nama': d.get('nama', ''),
                            'severity': d.get('severity', ''),
                            'score': float(d.get('score', 0))
                        }
                        for d in diag_list
                    ]
                }
                
                # Save diagnosis dengan FK ke predictions
                diag_id = save_diagnosis_mysql(
                    prediction_id=prediction_id,
                    diagnosis_dict=diagnosis_details,
                    severity=severity,
                    timestamp=datetime.datetime.utcnow()
                )
                diagnosis['saved_diagnosis_id'] = diag_id
                if prediction_id:
                    diagnosis['saved_prediction_id'] = int(prediction_id)
                    print(f"✓ Diagnosis saved to database, id={diag_id}, linked to prediction_id={prediction_id}, severity={severity}")
                else:
                    print(f"✓ Manual diagnosis saved to database, id={diag_id}, severity={severity}")
            except Exception as e:
                print(f"✗ Error saving diagnosis to MySQL: {e}")
                import traceback
                traceback.print_exc()
        
        # Jika ada hasil prediksi berbasis image processing sebelumnya, tambahkan ke konteks
        last_prediction = session.get('last_prediction', {})
        image_info = None
        if use_image_context and last_prediction.get('source') == 'image_processing':
            image_info = {
                'filename': last_prediction.get('original_filename', ''),
                'prediction': last_prediction.get('prediction', ''),
                'confidence': last_prediction.get('confidence', 0)
            }
        
        return render_template('expert_system.html', 
                             gejala_list=expert_system.get_gejala_list(),
                             gejala_groups=expert_system.get_gejala_groups(),
                             diagnosis=diagnosis,
                             selected_gejala=gejala_terpilih,
                             image_info=image_info,
                             context_mode='image' if use_image_context else 'manual')
    
    # GET request - tampilkan form
    # Ambil informasi gambar dari session jika hasil sebelumnya berasal dari image processing
    last_prediction = session.get('last_prediction', {})
    image_info = None
    if use_image_context and last_prediction.get('source') == 'image_processing':
        image_info = {
            'filename': last_prediction.get('original_filename', ''),
            'prediction': last_prediction.get('prediction', ''),
            'confidence': last_prediction.get('confidence', 0)
        }
    
    return render_template('expert_system.html', 
                         gejala_list=expert_system.get_gejala_list(),
                         gejala_groups=expert_system.get_gejala_groups(),
                         image_info=image_info,
                         context_mode='image' if use_image_context else 'manual')


@app.route('/expert-system/from-prediction')
def expert_system_from_prediction():
    """Redirect ke sistem pakar dari hasil prediksi"""
    # Bisa tambahkan logika untuk pre-fill gejala berdasarkan hasil prediksi
    return redirect(url_for('expert_system_page', mode='image'))


@app.route('/api/diagnosis', methods=['POST'])
def api_diagnosis():
    """API endpoint untuk diagnosis - Process dan Save ke database"""
    data = request.get_json()
    gejala = data.get('gejala', [])
    prediction_id = data.get('prediction_id', None)  # FK dari predictions table
    
    expert_system.reset()
    expert_system.tambah_gejala(gejala)
    diagnosis = expert_system.get_diagnosis()
    
    # Save diagnosis ke database jika MYSQL_AVAILABLE
    if MYSQL_AVAILABLE:
        try:
            # Prepare diagnosis data
            if diagnosis.get('status') == 'terdiagnosis' and diagnosis.get('diagnosis'):
                diag_list = diagnosis['diagnosis']
                # Save first (main) diagnosis
                main_diag = diag_list[0]
                
                # Ambil severity dari diagnosis result
                severity = main_diag.get('severity', 'sedang')
                score = main_diag.get('score', 0)
                
                # Prepare diagnosis details for storage dengan severity yang tepat
                diagnosis_details = {
                    'nama': main_diag.get('nama', ''),
                    'deskripsi': main_diag.get('deskripsi', ''),
                    'solusi': main_diag.get('solusi', []),
                    'score': float(score),
                    'gejala_teramati': main_diag.get('gejala_teramati', gejala),
                    'semua_diagnosis': [
                        {
                            'nama': d.get('nama', ''),
                            'severity': d.get('severity', ''),
                            'score': float(d.get('score', 0))
                        }
                        for d in diag_list
                    ]
                }
                
                # Save to database dengan FK ke predictions table
                diag_id = save_diagnosis_mysql(
                    prediction_id=prediction_id,
                    diagnosis_dict=diagnosis_details,
                    severity=severity,
                    timestamp=datetime.datetime.utcnow()
                )
                diagnosis['saved_diagnosis_id'] = diag_id
                if prediction_id:
                    print(f"✓ Diagnosis saved to database, id={diag_id}, linked to prediction_id={prediction_id}, severity={severity}")
                else:
                    print(f"✓ Manual diagnosis saved to database, id={diag_id}, severity={severity}")
        except Exception as e:
            print(f"✗ Error saving diagnosis to database: {e}")
            import traceback
            traceback.print_exc()
    
    return diagnosis  # Flask akan otomatis mengkonversi dict ke JSON


@app.route('/clear-session')
def clear_session():
    """Bersihkan session"""
    session.clear()
    flash('Session telah dibersihkan', 'success')
    return redirect(url_for('index'))


@app.route('/riwayat-diagnosis')
def riwayat_diagnosis():
    """Halaman riwayat diagnosis dari sistem pakar"""
    diagnosis_history = []
    data_source = 'csv'
    if MYSQL_AVAILABLE:
        try:
            rows = get_diagnosis_history_mysql(limit=50)
            if rows:
                # Extract data from diagnosis JSON and map to template keys
                for r in rows:
                    diagnosis_obj = r.get('diagnosis', {})
                    gejala_list = diagnosis_obj.get('gejala_teramati', [])
                    solusi_list = diagnosis_obj.get('solusi', [])
                    
                    diagnosis_history.append({
                        'id': r.get('id'),
                        'prediction_id': r.get('prediction_id'),
                        'timestamp': r.get('timestamp'),
                        'gejala': ','.join(gejala_list) if isinstance(gejala_list, list) else str(gejala_list),
                        'diagnosis': diagnosis_obj.get('nama', 'Tidak diketahui'),
                        'severity': r.get('severity', 'sedang'),
                        'rekomendasi': '|'.join(solusi_list) if isinstance(solusi_list, list) else str(solusi_list)
                    })
                diagnosis_history = sorted(
                    diagnosis_history,
                    key=lambda item: _history_sort_timestamp(item.get('timestamp')),
                    reverse=True,
                )
                data_source = 'mysql'
                return render_template('diagnosis_history.html', history=diagnosis_history, data_source=data_source)
        except Exception as e:
            print(f"MySQL read failed for diagnosis history: {e}")

    csv_path = os.path.join(BASE_DIR, 'results', 'diagnosis_history.csv')
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                df_recent = df.sort_values('timestamp', ascending=False).head(50)
                diagnosis_history = df_recent.to_dict(orient='records')
                diagnosis_history = sorted(
                    diagnosis_history,
                    key=lambda item: _history_sort_timestamp(item.get('timestamp')),
                    reverse=True,
                )
        except Exception as e:
            print(f"Error reading diagnosis history: {e}")

    return render_template('diagnosis_history.html', history=diagnosis_history, data_source=data_source)


@app.route('/export/csv')
def export_csv():
    csv_path = os.path.join(BASE_DIR, 'results', 'predictions.csv')
    if not os.path.exists(csv_path):
        flash('Tidak ada data untuk diekspor', 'error')
        return redirect(url_for('riwayat_deteksi'))

    try:
        df = pd.read_csv(csv_path)
        buf = io.BytesIO()
        buf.write(df.to_csv(index=False).encode('utf-8'))
        buf.seek(0)
        filename = f"riwayat_deteksi_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        return send_file(buf, mimetype='text/csv', as_attachment=True, attachment_filename=filename)
    except Exception as e:
        print(f"Gagal mengekspor CSV: {e}")
        flash('Gagal mengekspor CSV', 'error')
        return redirect(url_for('riwayat_deteksi'))


@app.route('/export/excel')
def export_excel():
    csv_path = os.path.join(BASE_DIR, 'results', 'predictions.csv')
    if not os.path.exists(csv_path):
        flash('Tidak ada data untuk diekspor', 'error')
        return redirect(url_for('riwayat_deteksi'))

    try:
        df = pd.read_csv(csv_path)
        buf = io.BytesIO()
        try:
            with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Riwayat')
            buf.seek(0)
            filename = f"riwayat_deteksi_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            return send_file(buf, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', as_attachment=True, attachment_filename=filename)
        except Exception:
            # fallback to CSV if excel writer not available
            buf = io.BytesIO()
            buf.write(df.to_csv(index=False).encode('utf-8'))
            buf.seek(0)
            filename = f"riwayat_deteksi_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            return send_file(buf, mimetype='text/csv', as_attachment=True, attachment_filename=filename)
    except Exception as e:
        print(f"Gagal mengekspor Excel: {e}")
        flash('Gagal mengekspor data', 'error')
        return redirect(url_for('riwayat_deteksi'))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(host='0.0.0.0', port=port, debug=debug)