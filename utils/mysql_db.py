"""
MySQL Database Integration for Deteksi PMK
Handles saving and retrieving predictions and diagnosis history
"""

import os
import json
import datetime
from collections import OrderedDict
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey, Table, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.exc import SQLAlchemyError

# Timezone setting untuk waktu lokal Indonesia
import pytz
TZ_INDONESIA = pytz.timezone('Asia/Jakarta')

DATABASE_URL = os.environ.get('DATABASE_URL')


def _build_database_uri():
    """Build a SQLAlchemy database URI from Railway or local environment variables.

    Priority:
    1. Individual Railway reference variables (MYSQLHOST, MYSQLPORT, MYSQLUSER,
       MYSQLPASSWORD, MYSQLDATABASE) — most reliable when set via Railway reference vars.
    2. DATABASE_URL — used as a fallback if the individual variables are absent.
    """
    mysql_host = os.environ.get('MYSQLHOST')
    mysql_port = os.environ.get('MYSQLPORT')
    mysql_user = os.environ.get('MYSQLUSER')
    mysql_password = os.environ.get('MYSQLPASSWORD')  # may be empty string — that is valid
    mysql_database = os.environ.get('MYSQLDATABASE')

    if mysql_host and mysql_user and mysql_database:
        db_port = mysql_port or '3306'
        db_password = mysql_password if mysql_password is not None else ''
        if db_password:
            return f"mysql+pymysql://{mysql_user}:{db_password}@{mysql_host}:{db_port}/{mysql_database}"
        return f"mysql+pymysql://{mysql_user}@{mysql_host}:{db_port}/{mysql_database}"

    if DATABASE_URL:
        url = make_url(DATABASE_URL)
        if url.drivername == 'mysql':
            url = url.set(drivername='mysql+pymysql')
        return str(url)

    # Last-resort local defaults
    db_host = os.environ.get('DB_HOST', 'localhost')
    db_port = os.environ.get('DB_PORT', '3306')
    db_user = os.environ.get('DB_USER', 'root')
    db_password = os.environ.get('DB_PASSWORD', '')
    db_name = os.environ.get('DB_NAME', 'deteksi_pmk')

    if db_password:
        return f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    return f"mysql+pymysql://{db_user}@{db_host}:{db_port}/{db_name}"


# Create database URI
DATABASE_URI = _build_database_uri()

# Create engine
engine = create_engine(DATABASE_URI, echo=False, pool_pre_ping=True)
Base = declarative_base()
Session = sessionmaker(bind=engine)


def _prediction_source_from_features(features):
    """Infer the prediction source from stored features metadata."""
    if isinstance(features, dict) and features.get('diagnosis_method') == 'manual_expert_system':
        return 'manual_expert_system'
    return 'image_processing'


expert_rules_expert_symptoms = Table(
    'expert_rules_expert_symptoms',
    Base.metadata,
    Column('rule_id', Integer, ForeignKey('expert_rules.id', ondelete='CASCADE'), primary_key=True),
    Column('symptom_id', Integer, ForeignKey('expert_symptoms.id', ondelete='CASCADE'), primary_key=True),
)


class Prediction(Base):
    """Model untuk menyimpan hasil prediksi"""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    original_filename = Column(String(255))
    filename = Column(String(255))
    image_path = Column(String(500))
    prediction = Column(String(50))  # 'sehat' or 'sakit'
    confidence = Column(Float)
    features = Column(Text)  # JSON string of features
    timestamp = Column(DateTime, default=lambda: datetime.datetime.now(TZ_INDONESIA).replace(tzinfo=None))


class DiagnosisHistory(Base):
    """Model untuk menyimpan hasil diagnosis dari expert system"""
    __tablename__ = 'diagnosis_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(Integer, ForeignKey('predictions.id'), nullable=True)  # FK ke predictions table
    diagnosis = Column(Text)  # JSON string of diagnosis details
    severity = Column(String(50))  # 'ringan', 'sedang', 'berat'
    timestamp = Column(DateTime, default=lambda: datetime.datetime.now(TZ_INDONESIA).replace(tzinfo=None))


class ExpertSymptom(Base):
    """Master gejala sistem pakar"""
    __tablename__ = 'expert_symptoms'

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(10), unique=True, nullable=False)
    description = Column(Text, nullable=False)
    category = Column(String(50), nullable=True)
    display_order = Column(Integer, default=0)
    rules = relationship('ExpertRule', secondary=expert_rules_expert_symptoms, back_populates='symptoms')


class ExpertDisease(Base):
    """Master penyakit sistem pakar"""
    __tablename__ = 'expert_diseases'

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(10), unique=True, nullable=False)
    name = Column(String(120), nullable=False)
    description = Column(Text, nullable=False)
    solutions = Column(Text, nullable=False)
    display_order = Column(Integer, default=0)
    rules = relationship('ExpertRule', back_populates='disease')


class ExpertRule(Base):
    """Aturan forward chaining"""
    __tablename__ = 'expert_rules'

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(20), unique=True, nullable=False)
    symptom_codes = Column(Text, nullable=False)
    result_disease_code = Column(String(10), ForeignKey('expert_diseases.code'), nullable=False)
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    display_order = Column(Integer, default=0)
    disease = relationship('ExpertDisease', back_populates='rules')
    symptoms = relationship('ExpertSymptom', secondary=expert_rules_expert_symptoms, back_populates='rules')


DEFAULT_EXPERT_SYMPTOMS = [
    {'code': 'G01', 'description': 'Demam lebih dari 39.5°C', 'category': 'umum', 'display_order': 1},
    {'code': 'G02', 'description': 'Air liur keluar berlebihan', 'category': 'mulut', 'display_order': 2},
    {'code': 'G03', 'description': 'Luka pada lidah, gusi, bantalan gigi, atau bibir', 'category': 'mulut', 'display_order': 3},
    {'code': 'G04', 'description': 'Nyeri setelah lepuh pecah', 'category': 'mulut', 'display_order': 4},
    {'code': 'G05', 'description': 'Lepuh pada celah kuku', 'category': 'kaki', 'display_order': 5},
    {'code': 'G06', 'description': 'Pincang', 'category': 'kaki', 'display_order': 6},
    {'code': 'G07', 'description': 'Lepuh pada puting', 'category': 'ambing', 'display_order': 7},
    {'code': 'G08', 'description': 'Lesi atau komplikasi pada puting', 'category': 'ambing', 'display_order': 8},
    {'code': 'G09', 'description': 'Produksi susu menurun', 'category': 'ambing', 'display_order': 9},
    {'code': 'G10', 'description': 'Hidung berair', 'category': 'mulut', 'display_order': 10},
    {'code': 'G11', 'description': 'Nafsu makan turun', 'category': 'umum', 'display_order': 11},
    {'code': 'G12', 'description': 'Lesu', 'category': 'umum', 'display_order': 12},
    {'code': 'G13', 'description': 'Miokarditis atau kematian mendadak', 'category': 'berat', 'display_order': 13},
    {'code': 'G14', 'description': 'Lepuh pada moncong', 'category': 'mulut', 'display_order': 14},
    {'code': 'G15', 'description': 'Nyeri kaki', 'category': 'kaki', 'display_order': 15},
    {'code': 'G16', 'description': 'Abortus atau infertilitas', 'category': 'berat', 'display_order': 16},
    {'code': 'G17', 'description': 'Demam lebih dari 40.5°C', 'category': 'umum', 'display_order': 17},
    {'code': 'G18', 'description': 'Lepuh pada lidah meluas', 'category': 'mulut', 'display_order': 18},
    {'code': 'G19', 'description': 'Sulit mengunyah atau menelan', 'category': 'mulut', 'display_order': 19},
    {'code': 'G20', 'description': 'Lepuh pada bantalan gigi', 'category': 'mulut', 'display_order': 20},
    {'code': 'G21', 'description': 'Bau mulut', 'category': 'mulut', 'display_order': 21},
    {'code': 'G22', 'description': 'Edema atau radang', 'category': 'kaki', 'display_order': 22},
    {'code': 'G23', 'description': 'Bengkak pada celah kuku', 'category': 'kaki', 'display_order': 23},
    {'code': 'G24', 'description': 'Telapak kaki longgar atau terlepas', 'category': 'kaki', 'display_order': 24},
    {'code': 'G25', 'description': 'Lebih sering berbaring', 'category': 'umum', 'display_order': 25},
    {'code': 'G26', 'description': 'Puting retak', 'category': 'ambing', 'display_order': 26},
    {'code': 'G27', 'description': 'Susu menggumpal', 'category': 'ambing', 'display_order': 27},
    {'code': 'G28', 'description': 'Takikardia atau irama jantung tidak normal', 'category': 'umum', 'display_order': 28},
    {'code': 'G29', 'description': 'Sesak napas atau gagal jantung', 'category': 'umum', 'display_order': 29},
]

DEFAULT_EXPERT_DISEASES = [
    {
        'code': 'P01',
        'name': 'PMK_ORAL',
        'description': 'PMK yang terutama menyerang bagian mulut. Biasanya terlihat luka, lepuh, air liur berlebih, atau sulit makan dan menelan.',
        'solutions': [
            'Pisahkan sapi yang sakit dari sapi yang sehat',
            'Berikan pakan yang lunak dan mudah dimakan',
            'Sediakan air minum yang cukup',
            'Periksa kondisi mulut sapi setiap hari',
            'Bersihkan luka dengan obat antiseptik sesuai anjuran dokter hewan',
            'Hubungi dokter hewan jika sapi susah makan atau minum',
        ],
        'display_order': 1,
    },
    {
        'code': 'P02',
        'name': 'PMK_PODAL',
        'description': 'PMK yang terutama menyerang kaki dan kuku. Biasanya sapi pincang, ada lepuh di celah kuku, atau kaki terasa sakit.',
        'solutions': [
            'Pisahkan sapi yang pincang dari kelompoknya',
            'Jaga kandang tetap kering dan bersih',
            'Bersihkan serta obati luka pada kaki sesuai anjuran dokter hewan',
            'Kurangi sapi berjalan terlalu jauh',
            'Periksa celah kuku dan telapak kaki secara rutin',
            'Segera minta bantuan dokter hewan jika pincang makin parah',
        ],
        'display_order': 2,
    },
    {
        'code': 'P03',
        'name': 'PMK_LAKTASI',
        'description': 'PMK yang menyerang ambing dan produksi susu. Biasanya puting retak, ada lepuh pada puting, atau susu menggumpal dan turun.',
        'solutions': [
            'Hentikan dulu pemerahan yang memicu sakit pada puting',
            'Bersihkan puting dan ambing dengan hati-hati',
            'Jaga kebersihan alat pemerahan',
            'Pantau produksi susu setiap hari',
            'Hubungi dokter hewan bila susu menggumpal atau puting luka',
            'Pisahkan sapi sakit agar tidak menular ke sapi lain',
        ],
        'display_order': 3,
    },
    {
        'code': 'P04',
        'name': 'PMK_JUVENIL',
        'description': 'PMK pada hewan muda yang biasanya terlihat dengan gejala berat seperti lemas, mudah berbaring, gangguan jantung, atau kematian mendadak.',
        'solutions': [
            'Pisahkan hewan muda yang terlihat lemah',
            'Pantau suhu tubuh dan detak jantung',
            'Segera hubungi dokter hewan karena kondisi bisa cepat memburuk',
            'Berikan pakan dan minum yang cukup bila masih mau makan',
            'Jangan biarkan hewan muda bercampur dengan ternak lain',
        ],
        'display_order': 4,
    },
    {
        'code': 'P05',
        'name': 'PMK_AKUT_GENERAL',
        'description': 'PMK yang muncul sangat cepat dan berat, biasanya dengan demam tinggi, lesu, nafsu makan turun, dan tanda gangguan tubuh yang umum.',
        'solutions': [
            'Pisahkan sapi yang sakit dari kelompoknya',
            'Hubungi dokter hewan secepatnya',
            'Pantau suhu tubuh, napas, dan detak jantung',
            'Berikan pakan yang mudah dimakan bila masih mau makan',
            'Jaga kebersihan kandang dan alat agar penularan tidak meluas',
        ],
        'display_order': 5,
    },
]

DEFAULT_EXPERT_RULES = [
    {
        'code': 'FC01',
        'symptom_codes': ['G01', 'G02', 'G03', 'G04', 'G11', 'G18', 'G19', 'G20', 'G21'],
        'result_disease_code': 'P01',
        'description': 'PMK oral: demam, air liur berlebihan, luka mulut, nyeri setelah lepuh pecah, nafsu makan turun, luka meluas, sulit mengunyah/menelan, dan bau mulut',
        'display_order': 1,
    },
    {
        'code': 'FC02',
        'symptom_codes': ['G01', 'G02', 'G05', 'G06', 'G15', 'G22', 'G23', 'G24', 'G25'],
        'result_disease_code': 'P02',
        'description': 'PMK podal: demam, air liur berlebihan, lepuh celah kuku, pincang, nyeri kaki, edema/radang, bengkak celah kuku, telapak kaki longgar, dan lebih sering berbaring',
        'display_order': 2,
    },
    {
        'code': 'FC03',
        'symptom_codes': ['G01', 'G02', 'G07', 'G08', 'G09', 'G26', 'G27'],
        'result_disease_code': 'P03',
        'description': 'PMK laktasi: demam, air liur berlebihan, lepuh puting, lesi/komplikasi puting, produksi susu menurun, puting retak, dan susu menggumpal',
        'display_order': 3,
    },
    {
        'code': 'FC04',
        'symptom_codes': ['G01', 'G13', 'G28', 'G29'],
        'result_disease_code': 'P04',
        'description': 'PMK juvenil: demam, miokarditis/kematian mendadak, takikardia atau irama jantung tidak normal, serta sesak napas atau gagal jantung',
        'display_order': 4,
    },
    {
        'code': 'FC05',
        'symptom_codes': ['G01', 'G02', 'G03', 'G04', 'G05', 'G06', 'G07', 'G09', 'G11', 'G12', 'G14', 'G18', 'G20', 'G22', 'G23', 'G24', 'G26'],
        'result_disease_code': 'P05',
        'description': 'PMK akut: demam, air liur berlebihan, luka mulut, nyeri setelah lepuh pecah, lepuh kaki/kuku, lepuh puting, produksi susu menurun, nafsu makan turun, lesu, lepuh moncong, luka meluas, edema/radang, bengkak celah kuku, telapak kaki longgar, dan puting retak',
        'display_order': 5,
    },
]

_GROUP_TITLES = OrderedDict([
    ('umum', 'Gejala Umum / Sistemik'),
    ('mulut', 'Gejala Mulut / Oral'),
    ('kaki', 'Gejala Kaki / Kuku'),
    ('ambing', 'Gejala Ambing / Laktasi'),
    ('berat', 'Gejala Berat / Khusus'),
])


def get_engine():
    """Return SQLAlchemy engine"""
    return engine


def init_mysql_tables():
    """Initialize database tables"""
    try:
        Base.metadata.create_all(engine)
        migrate_diagnosis_history_schema()
        migrate_expert_rule_disease_fk()
        seed_expert_knowledge(force=False)
        sync_expert_rule_symptom_relations()
        print("Database tables initialized successfully")
    except SQLAlchemyError as e:
        print(f"Error initializing database tables: {e}")
        raise


def migrate_diagnosis_history_schema():
    """Remove legacy diagnosis_history columns that are no longer used."""
    try:
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        if 'diagnosis_history' not in table_names:
            return

        column_names = {column['name'] for column in inspector.get_columns('diagnosis_history')}
        if 'confidence' not in column_names:
            return

        with engine.begin() as connection:
            connection.execute(text('ALTER TABLE diagnosis_history DROP COLUMN confidence'))
        print("✓ Removed legacy confidence column from diagnosis_history")
    except SQLAlchemyError as e:
        print(f"Warning: unable to migrate diagnosis_history schema: {e}")


def migrate_expert_rule_disease_fk():
    """Add the missing foreign key from expert_rules.result_disease_code to expert_diseases.code."""
    try:
        inspector = inspect(engine)
        if 'expert_rules' not in inspector.get_table_names() or 'expert_diseases' not in inspector.get_table_names():
            return

        existing_fks = inspector.get_foreign_keys('expert_rules')
        for fk in existing_fks:
            if fk.get('constrained_columns') == ['result_disease_code']:
                return

        with engine.begin() as connection:
            connection.execute(text(
                'ALTER TABLE expert_rules '
                'ADD CONSTRAINT fk_expert_rules_disease '
                'FOREIGN KEY (result_disease_code) REFERENCES expert_diseases(code) '
                'ON UPDATE CASCADE ON DELETE RESTRICT'
            ))
        print("✓ Added foreign key expert_rules.result_disease_code -> expert_diseases.code")
    except SQLAlchemyError as e:
        print(f"Warning: unable to add expert_rules disease foreign key: {e}")


def sync_expert_rule_symptom_relations():
    """Backfill the expert_rules_expert_symptoms join table from stored symptom codes."""
    session = Session()
    try:
        session.execute(expert_rules_expert_symptoms.delete())

        rules = session.query(ExpertRule).all()
        symptoms_by_code = {row.code: row for row in session.query(ExpertSymptom).all()}

        for rule in rules:
            try:
                symptom_codes = json.loads(rule.symptom_codes) if rule.symptom_codes else []
                if not isinstance(symptom_codes, list):
                    symptom_codes = []
            except Exception:
                symptom_codes = []

            for symptom_code in symptom_codes:
                symptom = symptoms_by_code.get(symptom_code)
                if symptom:
                    session.execute(
                        expert_rules_expert_symptoms.insert().values(
                            rule_id=rule.id,
                            symptom_id=symptom.id,
                        )
                    )

        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        print(f"Warning: unable to sync expert rule symptom relations: {e}")
    finally:
        session.close()


def seed_expert_knowledge(force=False):
    """Isi data default gejala, penyakit, dan aturan sistem pakar."""
    session = Session()
    try:
        existing_symptoms = session.query(ExpertSymptom).count()
        existing_diseases = session.query(ExpertDisease).count()
        existing_rules = session.query(ExpertRule).count()

        if not force and existing_symptoms > 0 and existing_diseases > 0 and existing_rules > 0:
            return {
                'seeded': False,
                'symptoms': existing_symptoms,
                'diseases': existing_diseases,
                'rules': existing_rules,
            }

        for item in DEFAULT_EXPERT_SYMPTOMS:
            row = session.query(ExpertSymptom).filter(ExpertSymptom.code == item['code']).first()
            if not row:
                row = ExpertSymptom(code=item['code'])
                session.add(row)
            row.description = item['description']
            row.category = item.get('category')
            row.display_order = item.get('display_order', 0)

        for item in DEFAULT_EXPERT_DISEASES:
            row = session.query(ExpertDisease).filter(ExpertDisease.code == item['code']).first()
            if not row:
                row = ExpertDisease(code=item['code'])
                session.add(row)
            row.name = item['name']
            row.description = item['description']
            row.solutions = json.dumps(item.get('solutions', []), ensure_ascii=False)
            row.display_order = item.get('display_order', 0)

        for item in DEFAULT_EXPERT_RULES:
            row = session.query(ExpertRule).filter(ExpertRule.code == item['code']).first()
            if not row:
                row = ExpertRule(code=item['code'])
                session.add(row)
            row.symptom_codes = json.dumps(item.get('symptom_codes', []), ensure_ascii=False)
            row.result_disease_code = item['result_disease_code']
            row.description = item.get('description', '')
            row.is_active = True
            row.display_order = item.get('display_order', 0)

        session.commit()
        sync_expert_rule_symptom_relations()

        return {
            'seeded': True,
            'symptoms': session.query(ExpertSymptom).count(),
            'diseases': session.query(ExpertDisease).count(),
            'rules': session.query(ExpertRule).count(),
        }
    except SQLAlchemyError as e:
        session.rollback()
        print(f"Error seeding expert knowledge: {e}")
        raise
    finally:
        session.close()


def get_expert_knowledge_mysql():
    """Ambil knowledge base sistem pakar (gejala, penyakit, aturan) dari MySQL."""
    session = Session()
    try:
        symptoms = session.query(ExpertSymptom).order_by(ExpertSymptom.display_order.asc(), ExpertSymptom.code.asc()).all()
        diseases = session.query(ExpertDisease).order_by(ExpertDisease.display_order.asc(), ExpertDisease.code.asc()).all()
        rules = session.query(ExpertRule).filter(ExpertRule.is_active == True).order_by(ExpertRule.display_order.asc(), ExpertRule.code.asc()).all()

        if not symptoms or not diseases or not rules:
            seed_expert_knowledge(force=False)
            session.close()
            session = Session()
            symptoms = session.query(ExpertSymptom).order_by(ExpertSymptom.display_order.asc(), ExpertSymptom.code.asc()).all()
            diseases = session.query(ExpertDisease).order_by(ExpertDisease.display_order.asc(), ExpertDisease.code.asc()).all()
            rules = session.query(ExpertRule).filter(ExpertRule.is_active == True).order_by(ExpertRule.display_order.asc(), ExpertRule.code.asc()).all()

        gejala = OrderedDict()
        grouped_codes = {k: [] for k in _GROUP_TITLES.keys()}
        for row in symptoms:
            gejala[row.code] = row.description
            category = (row.category or 'umum').lower()
            if category not in grouped_codes:
                grouped_codes[category] = []
            grouped_codes[category].append(row.code)

        penyakit = OrderedDict()
        for row in diseases:
            try:
                solusi = json.loads(row.solutions) if row.solutions else []
                if not isinstance(solusi, list):
                    solusi = []
            except Exception:
                solusi = []

            penyakit[row.code] = {
                'nama': row.name,
                'deskripsi': row.description,
                'solusi': solusi,
            }

        aturan = []
        for row in rules:
            try:
                gejala_codes = json.loads(row.symptom_codes) if row.symptom_codes else []
                if not isinstance(gejala_codes, list):
                    gejala_codes = []
            except Exception:
                gejala_codes = []

            aturan.append({
                'kode': row.code,
                'gejala': gejala_codes,
                'hasil': row.result_disease_code,
                'deskripsi': row.description or '',
            })

        gejala_groups = OrderedDict()
        for key, title in _GROUP_TITLES.items():
            gejala_groups[key] = {
                'title': title,
                'codes': grouped_codes.get(key, []),
            }

        return {
            'gejala': dict(gejala),
            'penyakit': dict(penyakit),
            'aturan': aturan,
            'gejala_groups': gejala_groups,
        }
    except SQLAlchemyError as e:
        print(f"Error loading expert knowledge from MySQL: {e}")
        return None
    finally:
        session.close()


def save_prediction_mysql(original_filename, filename, image_path, prediction, confidence, features_dict, timestamp=None):
    """
    Save prediction result to MySQL database
    
    Args:
        original_filename: Original filename uploaded
        filename: Saved filename
        image_path: Full path to saved image
        prediction: 'sehat' or 'sakit'
        confidence: Confidence score (0-1)
        features_dict: Dictionary of extracted features
        timestamp: Optional datetime, defaults to now
    
    Returns:
        ID of saved prediction
    """
    try:
        session = Session()
        
        # Convert features dict to JSON string
        features_json = json.dumps(features_dict, default=str)
        
        # Create prediction record
        pred = Prediction(
            original_filename=original_filename,
            filename=filename,
            image_path=image_path,
            prediction=prediction.lower(),
            confidence=float(confidence),
            features=features_json,
            timestamp=timestamp or datetime.datetime.now(TZ_INDONESIA).replace(tzinfo=None)
        )
        
        session.add(pred)
        session.commit()
        pred_id = pred.id
        session.close()
        
        return pred_id
    except SQLAlchemyError as e:
        print(f"Error saving prediction to database: {e}")
        raise


def get_recent_predictions_mysql(limit=10):
    """
    Get recent predictions from database
    
    Args:
        limit: Number of recent predictions to retrieve
    
    Returns:
        List of dictionaries containing prediction data
    """
    try:
        session = Session()
        
        # Query recent predictions, ordered by timestamp descending
        predictions = session.query(Prediction).order_by(
            Prediction.timestamp.desc()
        ).limit(limit).all()
        
        result = []
        for pred in predictions:
            try:
                features = json.loads(pred.features) if pred.features else {}
            except:
                features = {}

            source = _prediction_source_from_features(features)
            diagnosis = get_diagnosis_by_prediction_id(pred.id)

            diagnosis_label = None
            if diagnosis and isinstance(diagnosis, dict):
                diagnosis_label = diagnosis.get('diagnosis', {}).get('nama') or diagnosis.get('diagnosis', {}).get('name')
            
            result.append({
                'id': pred.id,
                'original_filename': pred.original_filename,
                'filename': pred.filename,
                'image_path': pred.image_path,
                'prediction': pred.prediction,
                'confidence': round(float(pred.confidence), 4),
                'features': features,
                'source': source,
                'diagnosis': diagnosis,
                'diagnosis_label': diagnosis_label,
                'timestamp': pred.timestamp.isoformat() if pred.timestamp else None
            })
        
        session.close()
        return result
    except SQLAlchemyError as e:
        print(f"Error retrieving predictions from database: {e}")
        return []


def get_prediction_by_id(pred_id):
    """
    Get specific prediction by ID
    
    Args:
        pred_id: Prediction ID
    
    Returns:
        Dictionary with prediction details or None
    """
    try:
        session = Session()
        
        pred = session.query(Prediction).filter(Prediction.id == pred_id).first()
        
        if not pred:
            session.close()
            return None
        
        try:
            features = json.loads(pred.features) if pred.features else {}
        except:
            features = {}

        source = _prediction_source_from_features(features)
        diagnosis = get_diagnosis_by_prediction_id(pred.id)
        diagnosis_label = None
        if diagnosis and isinstance(diagnosis, dict):
            diagnosis_label = diagnosis.get('diagnosis', {}).get('nama') or diagnosis.get('diagnosis', {}).get('name')
        
        result = {
            'id': pred.id,
            'original_filename': pred.original_filename,
            'filename': pred.filename,
            'image_path': pred.image_path,
            'prediction': pred.prediction,
            'confidence': round(float(pred.confidence), 4),
            'features': features,
            'source': source,
            'diagnosis': diagnosis,
            'diagnosis_label': diagnosis_label,
            'timestamp': pred.timestamp.isoformat() if pred.timestamp else None
        }
        
        session.close()
        return result
    except SQLAlchemyError as e:
        print(f"Error retrieving prediction by ID: {e}")
        return None


def save_diagnosis_mysql(prediction_id=None, diagnosis_dict=None, severity='sedang', timestamp=None):
    """
    Save diagnosis result from expert system to database
    
    Args:
        prediction_id: Foreign Key ke predictions table (prediction yang di-diagnosa). Optional.
        diagnosis_dict: Dictionary of diagnosis details from expert system
        severity: 'ringan', 'sedang', or 'berat'
        timestamp: Optional datetime, defaults to now
    
    Returns:
        ID of saved diagnosis
    """
    try:
        session = Session()
        
        diagnosis_dict = diagnosis_dict or {}

        # Convert diagnosis dict to JSON string
        diagnosis_json = json.dumps(diagnosis_dict, default=str)
        
        # Create diagnosis record with FK to predictions
        diag = DiagnosisHistory(
            prediction_id=prediction_id,
            diagnosis=diagnosis_json,
            severity=severity,
            timestamp=timestamp or datetime.datetime.now(TZ_INDONESIA).replace(tzinfo=None)
        )
        
        session.add(diag)
        session.commit()
        diag_id = diag.id
        session.close()
        
        return diag_id
    except SQLAlchemyError as e:
        print(f"Error saving diagnosis to database: {e}")
        raise


def get_diagnosis_history_mysql(limit=50, order_by='timestamp'):
    """
    Get diagnosis history from database
    
    Args:
        limit: Number of records to retrieve
        order_by: Column to order by
    
    Returns:
        List of dictionaries containing diagnosis data
    """
    try:
        session = Session()
        
        # Query diagnosis history, ordered by timestamp descending
        diagnoses = session.query(DiagnosisHistory).order_by(
            DiagnosisHistory.timestamp.desc()
        ).limit(limit).all()
        
        result = []
        for diag in diagnoses:
            try:
                diagnosis = json.loads(diag.diagnosis) if diag.diagnosis else {}
            except:
                diagnosis = {}
            
            result.append({
                'id': diag.id,
                'prediction_id': diag.prediction_id,
                'diagnosis': diagnosis,
                'severity': diag.severity,
                'timestamp': diag.timestamp.isoformat() if diag.timestamp else None
            })
        
        session.close()
        return result
    except SQLAlchemyError as e:
        print(f"Error retrieving diagnosis history from database: {e}")
        return []


def get_diagnosis_by_id(diag_id):
    """
    Get specific diagnosis by ID
    
    Args:
        diag_id: Diagnosis ID
    
    Returns:
        Dictionary with diagnosis details or None
    """
    try:
        session = Session()
        
        diag = session.query(DiagnosisHistory).filter(DiagnosisHistory.id == diag_id).first()
        
        if not diag:
            session.close()
            return None
        
        try:
            diagnosis = json.loads(diag.diagnosis) if diag.diagnosis else {}
        except:
            diagnosis = {}
        
        result = {
            'id': diag.id,
            'prediction_id': diag.prediction_id,
            'diagnosis': diagnosis,
            'severity': diag.severity,
            'timestamp': diag.timestamp.isoformat() if diag.timestamp else None
        }
        
        session.close()
        return result
    except SQLAlchemyError as e:
        print(f"Error retrieving diagnosis by ID: {e}")
        return None


def get_diagnosis_by_prediction_id(pred_id):
    """
    Get diagnosis for a specific prediction (by prediction_id FK)
    
    Args:
        pred_id: Prediction ID (Foreign Key)
    
    Returns:
        Dictionary with most recent diagnosis or None if no diagnosis exists
    """
    try:
        session = Session()
        
        # Get most recent diagnosis for this prediction
        diag = session.query(DiagnosisHistory).filter(
            DiagnosisHistory.prediction_id == pred_id
        ).order_by(DiagnosisHistory.timestamp.desc()).first()
        
        if not diag:
            session.close()
            return None
        
        try:
            diagnosis = json.loads(diag.diagnosis) if diag.diagnosis else {}
        except:
            diagnosis = {}
        
        result = {
            'id': diag.id,
            'prediction_id': diag.prediction_id,
            'diagnosis': diagnosis,
            'severity': diag.severity,
            'timestamp': diag.timestamp.isoformat() if diag.timestamp else None
        }
        
        session.close()
        return result
    except SQLAlchemyError as e:
        print(f"Error retrieving diagnosis by prediction ID: {e}")
        return None


def delete_prediction_mysql(pred_id):
    """Delete a prediction by ID"""
    try:
        session = Session()
        session.query(Prediction).filter(Prediction.id == pred_id).delete()
        session.commit()
        session.close()
        return True
    except SQLAlchemyError as e:
        print(f"Error deleting prediction: {e}")
        return False


def delete_diagnosis_mysql(diag_id):
    """Delete a diagnosis by ID"""
    try:
        session = Session()
        session.query(DiagnosisHistory).filter(DiagnosisHistory.id == diag_id).delete()
        session.commit()
        session.close()
        return True
    except SQLAlchemyError as e:
        print(f"Error deleting diagnosis: {e}")
        return False


def get_statistics_mysql():
    """Get statistics from database"""
    try:
        session = Session()
        
        # Count predictions
        total_predictions = session.query(Prediction).count()
        sakit_predictions = session.query(Prediction).filter(Prediction.prediction == 'sakit').count()
        sehat_predictions = session.query(Prediction).filter(Prediction.prediction == 'sehat').count()
        
        # Count diagnoses
        total_diagnoses = session.query(DiagnosisHistory).count()
        
        # Average confidence
        avg_confidence = None
        result = session.query(Prediction).first()
        if result:
            from sqlalchemy import func
            avg_conf = session.query(func.avg(Prediction.confidence)).scalar()
            avg_confidence = round(float(avg_conf), 4) if avg_conf else None
        
        session.close()
        
        return {
            'total_predictions': total_predictions,
            'sakit_predictions': sakit_predictions,
            'sehat_predictions': sehat_predictions,
            'total_diagnoses': total_diagnoses,
            'average_confidence': avg_confidence
        }
    except SQLAlchemyError as e:
        print(f"Error getting statistics: {e}")
        return {}
