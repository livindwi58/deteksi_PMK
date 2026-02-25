from flask import Flask, render_template, request, redirect, url_for, send_from_directory, send_file, flash, session
import io
import datetime
import os
from werkzeug.utils import secure_filename
import pandas as pd
import uuid
import time

from utils.helpers import load_model
from utils.preprocessing import preprocess_image
from utils.feature_extraction import FeatureExtractor
from expert_system import ForwardChaining  # Import sistem pakar

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ensure Flask uses the correct absolute template/static folders
app = Flask(__name__, 
           template_folder=os.path.join(BASE_DIR, 'templates'), 
           static_folder=os.path.join(BASE_DIR, 'static'))
app.secret_key = os.environ.get('FLASK_SECRET', 'change-me')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Inisialisasi sistem pakar
expert_system = ForwardChaining()

# Attempt MySQL integration (optional). If env var not set, fall back to CSV storage.
try:
    from utils.mysql_db import (
        save_prediction_mysql,
        get_recent_predictions_mysql,
        init_mysql_tables,
        save_diagnosis_mysql,
        get_diagnosis_history_mysql,
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

    # Prefer CSV (written synchronously on predict) so recent detection appears immediately;
    # fallback to MySQL only if CSV not present.
    history = []
    csv_path = os.path.join(BASE_DIR, 'results', 'predictions.csv')
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                df_recent = df.sort_values('timestamp', ascending=False).head(5)
                history = df_recent.to_dict(orient='records')
        except Exception as e:
            print(f"Error reading history for index: {e}")

    if not history and MYSQL_AVAILABLE:
        try:
            rows = get_recent_predictions_mysql(limit=5)
            if rows:
                history = rows
        except Exception as e:
            print(f"MySQL read failed for index: {e}")

    return render_template('index.html', 
                         model_loaded=bool(model_loaded), 
                         model_info=model_info, 
                         history=history)


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/dashboard')
@app.route('/riwayat-deteksi')
def riwayat_deteksi():
    recent = []
    stats = {'total': 0, 'positif': 0, 'negatif': 0}

    # Prefer CSV so new predictions are visible immediately; fallback to MySQL if CSV absent
    csv_path = os.path.join(BASE_DIR, 'results', 'predictions.csv')
    if os.path.exists(csv_path):
        try:
            # Robust CSV parsing: handle rows with extra/missing columns (old format variations)
            import csv as _csv
            from collections import Counter

            def _robust_read(path):
                with open(path, newline='', encoding='utf-8') as fh:
                    reader = _csv.reader(fh)
                    rows = [r for r in reader]
                if not rows:
                    return pd.DataFrame()
                header = rows[0]
                data_rows = rows[1:]
                if not data_rows:
                    return pd.DataFrame(columns=header)
                lengths = [len(r) for r in data_rows]
                cnt = Counter(lengths)
                most_common_len, _ = cnt.most_common(1)[0]

                # If header matches most common row length, use it
                if len(header) == most_common_len:
                    return pd.DataFrame(data_rows, columns=header)

                # If rows commonly have one extra column, assume missing 'filename' header at index 1
                if most_common_len == len(header) + 1:
                    new_header = header.copy()
                    new_header.insert(1, 'filename')
                    norm_rows = []
                    for r in data_rows:
                        if len(r) == len(header):
                            r2 = r.copy()
                            r2.insert(1, '')
                            norm_rows.append(r2)
                        elif len(r) >= len(new_header):
                            norm_rows.append(r[:len(new_header)])
                        else:
                            r2 = r + [''] * (len(new_header) - len(r))
                            norm_rows.append(r2)
                    return pd.DataFrame(norm_rows, columns=new_header)

                # Fallback to pandas (will create unnamed columns for extras)
                return pd.read_csv(path, low_memory=False)

            df = _robust_read(csv_path)

            # Normalize/match columns even if CSV format changed over time
            # 1) Find timestamp-like column
            timestamp_col = None
            best_ts_count = 0
            for col in df.columns:
                try:
                    parsed = pd.to_datetime(df[col], errors='coerce')
                    cnt = parsed.notna().sum()
                    if cnt > best_ts_count:
                        best_ts_count = cnt
                        timestamp_col = col
                except Exception:
                    continue

            if timestamp_col is not None and best_ts_count > 0:
                df['timestamp'] = pd.to_datetime(df[timestamp_col], errors='coerce')
            else:
                df['timestamp'] = pd.NaT

            # 2) Find prediction column by matching common labels
            prediction_col = None
            best_pred_count = 0
            for col in df.columns:
                try:
                    vals = df[col].astype(str).str.lower()
                    cnt = vals.isin(['sakit', 'sehat']).sum()
                    if cnt > best_pred_count:
                        best_pred_count = cnt
                        prediction_col = col
                except Exception:
                    continue

            if prediction_col:
                df['prediction'] = df[prediction_col].astype(str).str.lower()

            # 3) Ensure filename and image_path exist
            if 'filename' not in df.columns:
                # Heuristic: second column often contains filename when present
                if len(df.columns) >= 2:
                    possible = df.columns[1]
                    # if many values look like a uuid or end with image ext, use it
                    vals = df[possible].astype(str)
                    score = vals.str.contains(r"\.(jpg|jpeg|png|bmp)$", case=False, regex=True).sum()
                    if score > 0:
                        df['filename'] = df[possible]
                    else:
                        df['filename'] = ''
                else:
                    df['filename'] = ''

            if 'image_path' not in df.columns:
                # find a column containing file paths starting with drive letter or slash
                img_col = None
                for col in df.columns:
                    vals = df[col].astype(str)
                    if vals.str.contains(r'^[A-Za-z]:\\|^/|\\').any():
                        img_col = col
                        break
                df['image_path'] = df[img_col] if img_col else ''

            # Ensure confidence is numeric so templates can round it safely
            if 'confidence' in df.columns:
                try:
                    df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
                except Exception:
                    df['confidence'] = pd.NA

            # Compute stats safely
            stats['total'] = len(df)
            stats['positif'] = int((df.get('prediction', '') == 'sakit').sum()) if 'prediction' in df else 0
            stats['negatif'] = int((df.get('prediction', '') == 'sehat').sum()) if 'prediction' in df else 0

            # Sort by parsed timestamp (NaT go to end)
            if 'timestamp' in df.columns:
                df_recent = df.sort_values('timestamp', ascending=False, na_position='last').head(10)
            else:
                df_recent = df.tail(10)

            recent = df_recent.to_dict(orient='records')
            return render_template('riwayat_deteksi.html', recent=recent, stats=stats, data_source='csv')
        except Exception as e:
            print(f"Error reading CSV for dashboard: {e}")

    # CSV not available or failed; try MySQL
    if MYSQL_AVAILABLE:
        try:
            rows = get_recent_predictions_mysql(limit=10)
            if rows:
                recent = rows
                stats['total'] = len(rows)
                stats['positif'] = int(sum(1 for r in rows if (str(r.get('prediction') or '').lower() == 'sakit')))
                stats['negatif'] = int(sum(1 for r in rows if (str(r.get('prediction') or '').lower() == 'sehat')))
                return render_template('riwayat_deteksi.html', recent=recent, stats=stats, data_source='mysql')
        except Exception as e:
            print(f"MySQL read failed for dashboard: {e}")

    return render_template('riwayat_deteksi.html', recent=recent, stats=stats, data_source='none')


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

        # Save to CSV (ensure consistent columns, migrate old files if needed)
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
            print(f"Gagal menyimpan prediksi: {e}")

        # Also attempt to save to MySQL when available (non-fatal)
        if MYSQL_AVAILABLE:
            try:
                features_dict = {name: float(features[i]) for i, name in enumerate(extractor.feature_names)}
                # save_prediction_mysql(original_filename, filename, image_path, prediction, confidence, features, timestamp=None)
                rowid = save_prediction_mysql(original_filename, filename, filepath, prediction, confidence, features_dict)
                print(f"Saved prediction to MySQL, id={rowid}")
                # store DB id in session so result page can link to the new history row
                try:
                    session.setdefault('last_prediction', {})
                    session['last_prediction']['db_id'] = int(rowid) if rowid is not None else None
                except Exception:
                    pass
            except Exception as e:
                import traceback
                print(f"Gagal menyimpan ke MySQL: {e}")
                traceback.print_exc()

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
        
        # Jika ada hasil prediksi sebelumnya, tambahkan ke konteks
        image_info = None
        if 'last_prediction' in session:
            image_info = {
                'filename': session['last_prediction']['original_filename'],
                'prediction': session['last_prediction']['prediction'],
                'confidence': session['last_prediction']['confidence']
            }
        
        # Persist diagnosis to CSV and MySQL (if available)
        try:
            os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)
            # Normalize fields for storage
            if diagnosis.get('status') == 'terdiagnosis' and diagnosis.get('diagnosis'):
                top = diagnosis['diagnosis'][0]
                diag_text = top.get('nama')
                confidence = float(top.get('cf', 0.0))
                rekom = top.get('solusi', [])
            else:
                diag_text = diagnosis.get('message', str(diagnosis))
                confidence = 0.0
                rekom = []

            gejala_str = ','.join(diagnosis.get('gejala_teramati', []))

            row = {
                'timestamp': pd.Timestamp.now(),
                'gejala': gejala_str,
                'diagnosis': diag_text,
                'confidence': confidence,
                'rekomendasi': '|'.join(rekom)
            }
            csv_path = os.path.join(BASE_DIR, 'results', 'diagnosis_history.csv')
            df_row = pd.DataFrame([row])
            if os.path.exists(csv_path):
                df_row.to_csv(csv_path, mode='a', header=False, index=False)
            else:
                df_row.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"Gagal menyimpan diagnosis ke CSV: {e}")

        if MYSQL_AVAILABLE:
            try:
                save_diagnosis_mysql(diagnosis.get('gejala_teramati', []), diag_text, confidence=confidence, rekomendasi=rekom, related_prediction_id=None)
            except Exception as e:
                print(f"Gagal menyimpan diagnosis ke MySQL: {e}")

        return render_template('expert_system.html', 
                             gejala_list=expert_system.get_gejala_list(),
                             diagnosis=diagnosis,
                             selected_gejala=gejala_terpilih,
                             image_info=image_info,
                             data_source=('mysql' if MYSQL_AVAILABLE else 'csv'))
    
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
    """API endpoint untuk diagnosis"""
    data = request.get_json()
    gejala = data.get('gejala', [])
    
    expert_system.reset()
    expert_system.tambah_gejala(gejala)
    diagnosis = expert_system.get_diagnosis()
    
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
                # normalize to expected template keys
                for r in rows:
                    diagnosis_history.append({
                        'id': r.get('id'),
                        'timestamp': r.get('timestamp'),
                        'gejala': ','.join(r.get('gejala') or []),
                        'diagnosis': r.get('diagnosis'),
                        'confidence': r.get('confidence') or 0.0,
                        'rekomendasi': '|'.join(r.get('rekomendasi') or [])
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