from flask import Flask, render_template, request, redirect, url_for, send_from_directory, send_file, flash, session
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

from utils.helpers import load_model
from utils.preprocessing import preprocess_image
from utils.feature_extraction import FeatureExtractor
from expert_system import ForwardChaining, KnowledgeBase  # Import sistem pakar

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ensure Flask uses the correct absolute template/static folders
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

# Attempt MySQL integration (optional). If env var not set, fall back to CSV storage.
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
    # Test DB connection now; only enable MySQL features if connect succeeds
    try:
        engine = get_engine()
        # quick connect test
        with engine.connect() as conn:
            # initialize tables if needed
            try:
                init_mysql_tables()
            except Exception:
                # ignore init errors; will fallback to CSV reads/writes at runtime
                pass
        MYSQL_AVAILABLE = True
    except Exception as e:
        print('MySQL not available at startup:', e)
        MYSQL_AVAILABLE = False
except Exception:
    MYSQL_AVAILABLE = False
# Defer heavy model imports/loading to a background thread so startup remains snappy
model = None
scaler = None
label_encoder = None
extractor = FeatureExtractor()

def _load_model_background():
    global model, scaler, label_encoder
    try:
        from utils.helpers import load_model
        print('Loading ML model in background...')
        m, s, le = load_model()
        model, scaler, label_encoder = m, s, le
        print('Model loaded (background).')
    except Exception as e:
        model = None
        scaler = None
        label_encoder = None
        print(f"Background model load failed: {e}")

import threading
threading.Thread(target=_load_model_background, daemon=True).start()


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


@app.route('/dashboard')
@app.route('/riwayat-deteksi')
@app.route('/riwayat_deteksi')
def riwayat_deteksi():
    recent = []
    stats = {'total': 0, 'positif': 0, 'negatif': 0}

    # Get data from MySQL database only
    if MYSQL_AVAILABLE:
        try:
            rows = get_recent_predictions_mysql(limit=10)
            # Query successful - return dengan data (bisa kosong)
            recent = rows if rows else []
            stats['total'] = len(recent)
            stats['positif'] = int(sum(1 for r in recent if (str(r.get('prediction') or '').lower() == 'sakit')))
            stats['negatif'] = int(sum(1 for r in recent if (str(r.get('prediction') or '').lower() == 'sehat')))
            return render_template('riwayat_deteksi.html', recent=recent, stats=stats, data_source='mysql')
        except Exception as e:
            print(f"✗ Error reading from MySQL: {e}")
            flash('Error memuat riwayat deteksi dari database', 'danger')
            return render_template('riwayat_deteksi.html', recent=recent, stats=stats, data_source='error')
    else:
        flash('Database tidak tersedia. Silakan setup database terlebih dahulu.', 'warning')
        return render_template('riwayat_deteksi.html', recent=recent, stats=stats, data_source='database_error')


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
        except Exception as e:
            print(f"✗ Error getting prediction from MySQL: {e}")
    else:
        flash('Database tidak tersedia. Silakan setup database terlebih dahulu.', 'danger')
    
    # Not found or database error
    flash(f"Prediksi dengan ID {pred_id} tidak ditemukan", 'danger')
    return redirect(url_for('riwayat_deteksi'))


@app.route('/upload')
def upload():
    """Halaman upload gambar"""
    return render_template('upload.html', error=None)


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

        if model is None or scaler is None or label_encoder is None:
            flash('Model belum tersedia. Jalankan training terlebih dahulu.', 'error')
            return redirect(url_for('index'))

        # Preprocess and extract
        img_norm, img_resized, mask = preprocess_image(filepath)
        features = extractor.extract_all_features(img_norm, mask)
        features_scaled = scaler.transform([features])

        pred_encoded = model.predict(features_scaled)[0]
        prediction = label_encoder.inverse_transform([pred_encoded])[0]
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
    
    return render_template('result_healthy.html', 
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
    
    return render_template('result_sick.html', 
                         result=result, 
                         features_table=pred['features_table'])


@app.route('/expert-system', methods=['GET', 'POST'])
def expert_system_page():
    """Halaman sistem pakar forward chaining"""
    # Batasi akses: hanya setelah ada hasil prediksi terakhir
    if 'last_prediction' not in session:
        flash('Akses hanya tersedia setelah melakukan deteksi gambar.', 'error')
        return redirect(url_for('index'))
    if request.method == 'POST':
        # Dapatkan gejala yang dipilih
        gejala_terpilih = request.form.getlist('gejala')
        
        # Reset dan tambahkan gejala
        expert_system.reset()
        expert_system.tambah_gejala(gejala_terpilih)
        
        # Dapatkan diagnosis
        diagnosis = expert_system.get_diagnosis()
        
        # Get prediction_id dari session untuk Foreign Key
        prediction_id = session.get('last_prediction', {}).get('db_id')
        
        # Save diagnosis ke database dengan FK ke predictions
        if MYSQL_AVAILABLE and prediction_id:
            try:
                if diagnosis.get('status') == 'terdiagnosis' and diagnosis.get('diagnosis'):
                    diag_list = diagnosis['diagnosis']
                    main_diag = diag_list[0]
                    
                    # Determine severity from CF
                    cf = main_diag.get('cf', 0)
                    if cf >= 70:
                        severity = 'berat'
                    elif cf >= 50:
                        severity = 'sedang'
                    else:
                        severity = 'ringan'
                    
                    # Prepare diagnosis details
                    diagnosis_details = {
                        'nama': main_diag.get('nama', ''),
                        'deskripsi': main_diag.get('deskripsi', ''),
                        'solusi': main_diag.get('solusi', []),
                        'cf': float(cf),
                        'gejala_teramati': gejala_terpilih,
                        'semua_diagnosis': [
                            {
                                'nama': d.get('nama', ''),
                                'cf': float(d.get('cf', 0))
                            }
                            for d in diag_list
                        ]
                    }
                    
                    # Save diagnosis dengan FK ke predictions
                    diag_id = save_diagnosis_mysql(
                        prediction_id=prediction_id,
                        diagnosis_dict=diagnosis_details,
                        severity=severity,
                        confidence=float(cf),
                        timestamp=datetime.datetime.utcnow()
                    )
                    print(f"✓ Diagnosis saved to database, id={diag_id}, linked to prediction_id={prediction_id}")
            except Exception as e:
                print(f"✗ Error saving diagnosis to MySQL: {e}")
                import traceback
                traceback.print_exc()
        
        # Jika ada hasil prediksi sebelumnya, tambahkan ke konteks
        image_info = None
        if 'last_prediction' in session:
            image_info = {
                'filename': session['last_prediction']['original_filename'],
                'prediction': session['last_prediction']['prediction'],
                'confidence': session['last_prediction']['confidence']
            }
        
        return render_template('expert_system.html', 
                             gejala_list=expert_system.get_gejala_list(),
                             diagnosis=diagnosis,
                             selected_gejala=gejala_terpilih,
                             image_info=image_info)
    
    # GET request - tampilkan form
    # Ambil informasi gambar dari session jika ada
    image_info = None
    if 'last_prediction' in session:
        image_info = {
            'filename': session['last_prediction']['original_filename'],
            'prediction': session['last_prediction']['prediction'],
            'confidence': session['last_prediction']['confidence']
        }
    
    return render_template('expert_system.html', 
                         gejala_list=expert_system.get_gejala_list(),
                         image_info=image_info)


@app.route('/expert-system/from-prediction')
def expert_system_from_prediction():
    """Redirect ke sistem pakar dari hasil prediksi"""
    # Bisa tambahkan logika untuk pre-fill gejala berdasarkan hasil prediksi
    return redirect(url_for('expert_system_page'))


@app.route('/api/diagnosis', methods=['POST'])
def api_diagnosis():
    """API endpoint untuk diagnosis - Process dan Save ke database"""
    data = request.get_json()
    gejala = data.get('gejala', [])
    prediction_id = data.get('prediction_id', None)  # FK dari predictions table
    
    expert_system.reset()
    expert_system.tambah_gejala(gejala)
    diagnosis = expert_system.get_diagnosis()
    
    # Save diagnosis ke database jika MYSQL_AVAILABLE dan ada prediction_id
    if MYSQL_AVAILABLE and prediction_id:
        try:
            # Prepare diagnosis data
            if diagnosis.get('status') == 'terdiagnosis' and diagnosis.get('diagnosis'):
                diag_list = diagnosis['diagnosis']
                # Save first (main) diagnosis
                main_diag = diag_list[0]
                
                # Determine severity from CF (Certainty Factor)
                cf = main_diag.get('cf', 0)
                if cf >= 70:
                    severity = 'berat'
                elif cf >= 50:
                    severity = 'sedang'
                else:
                    severity = 'ringan'
                
                # Prepare diagnosis details for storage
                diagnosis_details = {
                    'nama': main_diag.get('nama', ''),
                    'deskripsi': main_diag.get('deskripsi', ''),
                    'solusi': main_diag.get('solusi', []),
                    'cf': float(main_diag.get('cf', 0)),
                    'gejala_teramati': gejala,
                    'semua_diagnosis': [
                        {
                            'nama': d.get('nama', ''),
                            'cf': float(d.get('cf', 0))
                        }
                        for d in diag_list
                    ]
                }
                
                # Save to database dengan FK ke predictions table
                diag_id = save_diagnosis_mysql(
                    prediction_id=prediction_id,
                    diagnosis_dict=diagnosis_details,
                    severity=severity,
                    confidence=float(cf),
                    timestamp=datetime.datetime.utcnow()
                )
                print(f"✓ Diagnosis saved to database, id={diag_id}, linked to prediction_id={prediction_id}")
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
                        'confidence': r.get('confidence') or 0.0,
                        'rekomendasi': '|'.join(solusi_list) if isinstance(solusi_list, list) else str(solusi_list)
                    })
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
    app.run(host='0.0.0.0', port=5000, debug=True)