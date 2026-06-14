from flask import Flask, render_template, request, redirect, url_for, send_from_directory, send_file, flash, session, jsonify
import io
import datetime
import os
import cv2
from werkzeug.utils import secure_filename
import pandas as pd
import uuid
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from utils.helpers import load_model, estimate_prediction_confidence
from utils.preprocessing import preprocess_image, preprocess_pipeline, validate_cattle_image, detect_udder
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
UPLOAD_RESIZE_FOLDER = os.path.join(UPLOAD_FOLDER, 'resize')
os.makedirs(UPLOAD_RESIZE_FOLDER, exist_ok=True)
UPLOAD_THRESHOLD_FOLDER = os.path.join(UPLOAD_FOLDER, 'threshold')
os.makedirs(UPLOAD_THRESHOLD_FOLDER, exist_ok=True)

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
        save_batch_prediction_mysql,
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
multiclass_model = None
multiclass_scaler = None
multiclass_label_encoder = None
extractor = FeatureExtractor()
model_loading = False  
model_loaded = False   

def _load_model_background():
    global model, scaler, label_encoder, multiclass_model, multiclass_scaler, multiclass_label_encoder
    global model_loading, model_loaded
    
    if model_loading or model_loaded:
        return
    
    model_loading = True
    try:
        from utils.helpers import load_model
        print('[APP] Loading ML models in background...')
        m, s, le = load_model()
        model, scaler, label_encoder = m, s, le
        try:
            mm, ms, mle = load_model(prefix='multiclass_')
            multiclass_model, multiclass_scaler, multiclass_label_encoder = mm, ms, mle
        except Exception:
            print('[APP] ⚠️ Multi-class model not loaded (binary-only mode)')
        model_loaded = True
        print('[APP] Models loaded successfully.')
    except Exception as e:
        model = None
        scaler = None
        label_encoder = None
        model_loaded = True
        print(f"[APP] ❌ Background model load failed: {e}")
    finally:
        model_loading = False

import threading
threading.Thread(target=_load_model_background, daemon=True).start()


def is_model_ready():
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

    images_data = prediction_row.get('images_data')
    image_count = prediction_row.get('image_count', 1)
    sick_count = sum(1 for img in images_data if str(img.get('prediction', '')).lower() == 'sakit') if images_data else (1 if prediction == 'sakit' else 0)

    if image_count > 1:
        display_label = f'{sick_count} sakit / {image_count - sick_count} sehat ({image_count} gambar)'

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
        'image_count': image_count,
        'images_data': images_data,
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
        is_cattle, confidence, reason = validate_cattle_image(temp_filepath, confidence_threshold=0.65)
        
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


def _predict_single(filepath, original_filename):
    """Process a single image and return prediction result dict."""
    features = extractor.extract_all_features(*preprocess_pipeline(filepath))
    features_scaled = scaler.transform([features])

    pred_encoded = model.predict(features_scaled)[0]
    prediction = label_encoder.inverse_transform([pred_encoded])[0]
    confidence = estimate_prediction_confidence(model, features_scaled)
    if confidence is None:
        confidence = float(max(model.predict_proba(features_scaled)[0]) * 100)

    pmk_type = None
    if prediction.lower() == 'sakit' and multiclass_model is not None:
        multi_scaled = multiclass_scaler.transform([features])
        multi_enc = multiclass_model.predict(multi_scaled)[0]
        pmk_type = multiclass_label_encoder.inverse_transform([multi_enc])[0]

    # Fallback heuristic: jika model menandai 'sehat' tapi gambar mengandung ciri ambing/puting,
    # anggap sebagai laktasi (pmk_laktasi) dan ubah prediksi agar sistem pakar terpicu.
    if prediction.lower() == 'sehat':
        try:
            udder_detected, udder_score = detect_udder(filepath)
            if udder_detected:
                prediction = 'sakit'
                pmk_type = 'pmk_laktasi'
                # update confidence minimal berdasarkan score heuristik
                confidence = max(confidence, min(udder_score * 100.0, 95.0))
                print(f"[HEURISTIC] Udder heuristic triggered for {original_filename} | score={udder_score:.2f}")
        except Exception as e:
            print(f"[HEURISTIC] Udder heuristic error: {e}")

    features_dict = {name: float(features[i]) for i, name in enumerate(extractor.feature_names)}
    features_table = list(zip(extractor.feature_names, [round(float(x), 4) for x in features]))

    return {
        'original_filename': original_filename,
        'prediction': prediction,
        'pmk_type': pmk_type,
        'confidence': confidence,
        'features': features,
        'features_dict': features_dict,
        'features_table': features_table,
    }


def _save_prediction_to_mysql(original_filename, filename, filepath, prediction, confidence, features_dict):
    """Save a single prediction to MySQL and return rowid."""
    rowid = None
    if MYSQL_AVAILABLE:
        try:
            rowid = save_prediction_mysql(original_filename, filename, filepath, prediction, confidence, features_dict)
            print(f"✓ Saved prediction to MySQL, id={rowid}")
        except Exception as e:
            import traceback
            print(f"✗ Gagal menyimpan ke MySQL: {e}")
            traceback.print_exc()
    return rowid


def _append_prediction_to_csv(original_filename, filename, prediction, confidence, features, features_dict):
    """Append a single prediction row to CSV fallback."""
    try:
        os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)
        data = {'image_path': [original_filename], 'filename': [filename],
                'prediction': [prediction], 'confidence': [confidence],
                'timestamp': [pd.Timestamp.now()]}
        for i, name in enumerate(extractor.feature_names):
            data[name] = [float(features[i])]
        df = pd.DataFrame(data)
        csv_path = os.path.join(BASE_DIR, 'results', 'predictions.csv')
        if os.path.exists(csv_path):
            try:
                existing_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
            except Exception:
                existing_cols = []
            desired_cols = list(df.columns)
            if not set(desired_cols).issubset(set(existing_cols)):
                try:
                    df_existing = pd.read_csv(csv_path)
                    for c in desired_cols:
                        if c not in df_existing.columns:
                            df_existing[c] = ''
                    df_existing = df_existing.reindex(columns=desired_cols)
                    df_existing.to_csv(csv_path, index=False)
                except Exception as e:
                    print(f"Gagal memigrasi CSV lama: {e}")
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"Gagal menyimpan prediksi ke CSV: {e}")


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        flash('File tidak ditemukan', 'error')
        return redirect(url_for('index'))

    files = request.files.getlist('image')
    if not files or (len(files) == 1 and files[0].filename == ''):
        flash('Tidak ada file yang dipilih', 'error')
        return redirect(url_for('index'))

    if not is_model_ready():
        flash('Model belum tersedia. Jalankan training terlebih dahulu.', 'error')
        return redirect(url_for('index'))

    results = []

    for file in files:
        if file.filename == '' or not allowed_file(file.filename):
            continue

        original_filename = secure_filename(file.filename)
        ext = original_filename.rsplit('.', 1)[1].lower()
        filename = f"{uuid.uuid4()}.{ext}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        is_cattle, validation_confidence, validation_reason = validate_cattle_image(filepath, confidence_threshold=0.65)
        if not is_cattle:
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception:
                    pass
            continue

        result = _predict_single(filepath, original_filename)
        result['filename'] = filename
        result['filepath'] = filepath

        results.append(result)

    if not results:
        flash('Tidak ada gambar sapi yang valid untuk diproses', 'error')
        return redirect(url_for('index'))

    # Save ALL images as ONE prediction row
    images_for_db = []
    for r in results:
        images_for_db.append({
            'original_filename': r['original_filename'],
            'filename': r['filename'],
            'image_path': r['filepath'],
            'prediction': r['prediction'],
            'pmk_type': r['pmk_type'],
            'confidence': r['confidence'],
        })

    pred_id = None
    if MYSQL_AVAILABLE:
        try:
            pred_id = save_batch_prediction_mysql(images_for_db)
            print(f"✓ Saved batch prediction to MySQL, id={pred_id} ({len(results)} images)")
        except Exception as e:
            import traceback
            print(f"✗ Gagal menyimpan batch ke MySQL: {e}")
            traceback.print_exc()

    # Also save to CSV (batch summary)
    _append_prediction_to_csv(
        f"{len(results)} images batch",
        f"batch_{pred_id}" if pred_id else "batch_unknown",
        'sakit' if any(r['prediction'].lower() == 'sakit' for r in results) else 'sehat',
        max(r['confidence'] for r in results),
        results[0]['features'],
        results[0]['features_dict']
    )

    # Build image data for session/template
    upload_images = []
    for r in results:
        upload_images.append({
            'filename': r['filename'],
            'original_filename': r['original_filename'],
            'filepath': r['filepath'],
            'prediction': r['prediction'],
            'pmk_type': r['pmk_type'],
            'confidence': r['confidence'],
            'features_table': r['features_table'],
        })

    # Determine if any sick results
    sick_pmk_types = set()
    for r in results:
        if r['prediction'].lower() == 'sakit' and r['pmk_type']:
            sick_pmk_types.add(r['pmk_type'])

    pmk_to_symptoms = {
        'pmk_oral': ['G02', 'G03', 'G04', 'G10', 'G14', 'G18', 'G19', 'G20', 'G21'],
        'pmk_podal': ['G05', 'G06', 'G15', 'G22', 'G23', 'G24'],
        # Udder (laktasi) — gunakan hanya gejala spesifik ambing/puting
        'pmk_laktasi': ['G07', 'G08', 'G09', 'G26', 'G27'],
        # General akut: kombinasi gejala yang menunjukkan PMK akut/menular luas
        'pmk_akut_general': ['G01', 'G02', 'G03', 'G04', 'G05', 'G06', 'G07', 'G09', 'G11', 'G12', 'G14', 'G18', 'G20', 'G22', 'G23', 'G24', 'G26']
    }

    preselected = set()
    for pmk_type in sick_pmk_types:
        key = pmk_type.lower()
        if key in pmk_to_symptoms:
            preselected.update(pmk_to_symptoms[key])

    # Jika lebih dari satu body-part berbeda terdeteksi sakit (misal: mulut+kaki, mulut+puting, kaki+puting),
    # anggap kemungkinan PMK akut/general dan preselect gejala umum akut juga.
    body_part_keys = {k for k in sick_pmk_types if k in {'pmk_oral', 'pmk_podal', 'pmk_laktasi'}}
    if len(body_part_keys) >= 2:
        # tambahkan gejala PMK akut general
        preselected.update(pmk_to_symptoms.get('pmk_akut_general', []))
        # juga tambahkan tipe pmk_akut_general ke hasil sehingga UI/riwayat menampilkan tipe ini
        sick_pmk_types.add('pmk_akut_general')

    # Store all images in session for expert system display
    session['last_prediction'] = {
        'db_id': pred_id,
        'filename': upload_images[0]['filename'],
        'original_filename': upload_images[0]['original_filename'],
        'filepath': upload_images[0]['filepath'],
        'prediction': 'sakit' if sick_pmk_types else 'sehat',
        'confidence': upload_images[0]['confidence'],
        'features_table': upload_images[0]['features_table'],
        'source': 'image_processing',
    }
    session['upload_images_data'] = upload_images

    time.sleep(1.0)

    if preselected:
        symptoms_param = ','.join(sorted(preselected))
        return redirect(url_for('expert_system_page', mode='image', symptoms=symptoms_param))
    else:
        flash(f'Semua {len(results)} gambar terdeteksi SEHAT', 'success')
        return redirect(url_for('result_healthy'))


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

        if MYSQL_AVAILABLE and diagnosis.get('status') == 'terdiagnosis' and diagnosis.get('diagnosis') and not use_image_context:
            try:
                diag_list = diagnosis['diagnosis']
                main_diag = diag_list[0]
                severity = main_diag.get('severity', 'sedang')
                score = main_diag.get('score', 0)

                features_dict = {
                    'diagnosis_method': 'manual_expert_system',
                    'mode': 'manual',
                    'severity': severity,
                    'gejala_selected': gejala_terpilih,
                }

                prediction_id = save_prediction_mysql(
                    original_filename='Diagnosis Sistem Pakar (Manual)',
                    filename='manual_expert_system',
                    image_path='manual_expert_system',
                    prediction='sakit',
                    confidence=float(score),
                    features_dict=features_dict,
                )

                session['last_prediction'] = {
                    'db_id': int(prediction_id),
                    'filename': 'manual_expert_system',
                    'original_filename': 'Diagnosis Sistem Pakar (Manual)',
                    'prediction': 'sakit',
                    'confidence': float(score),
                    'source': 'manual_expert_system'
                }
                diagnosis['saved_prediction_id'] = int(prediction_id)
                print(f"✓ Manual prediction saved to database, id={prediction_id}, severity={severity}")
            except Exception as e:
                print(f"✗ Error creating manual prediction entry: {e}")
                import traceback
                traceback.print_exc()

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
        upload_images = session.get('upload_images_data', []) if use_image_context else []
        image_info = None
        if use_image_context and last_prediction.get('source') == 'image_processing':
            image_info = {
                'filename': last_prediction.get('original_filename', ''),
                'prediction': last_prediction.get('prediction', ''),
                'confidence': last_prediction.get('confidence', 0),
                'db_id': last_prediction.get('db_id'),
            }
        
        return render_template('expert_system.html', 
                             gejala_list=expert_system.get_gejala_list(),
                             gejala_groups=expert_system.get_gejala_groups(),
                             diagnosis=diagnosis,
                             selected_gejala=gejala_terpilih,
                             image_info=image_info,
                             upload_images=upload_images,
                             context_mode='image' if use_image_context else 'manual')
    
    # GET request - tampilkan form
    # Parse preselected symptoms from query param
    preselected_symptoms = []
    symptoms_param = request.args.get('symptoms', '')
    if symptoms_param:
        preselected_symptoms = [s.strip() for s in symptoms_param.split(',') if s.strip()]

    # Ambil informasi gambar dari session jika hasil sebelumnya berasal dari image processing
    last_prediction = session.get('last_prediction', {})
    upload_images = session.get('upload_images_data', []) if use_image_context else []
    image_info = None
    if use_image_context and last_prediction.get('source') == 'image_processing':
        image_info = {
            'filename': last_prediction.get('original_filename', ''),
            'prediction': last_prediction.get('prediction', ''),
            'confidence': last_prediction.get('confidence', 0),
            'db_id': last_prediction.get('db_id'),
        }
    
    return render_template('expert_system.html', 
                         gejala_list=expert_system.get_gejala_list(),
                         gejala_groups=expert_system.get_gejala_groups(),
                         image_info=image_info,
                         upload_images=upload_images,
                         selected_gejala=preselected_symptoms,
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