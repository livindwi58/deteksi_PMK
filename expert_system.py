import os

# Basis pengetahuan untuk penyakit PMK pada sapi
class KnowledgeBase:
    def __init__(self):
        self.gejala = {
            # Gejala Umum
            'G01': 'Demam tinggi (>40°C)',
            'G02': 'Sapi terlihat lesu dan tidak nafsu makan',
            'G03': 'Produksi susu menurun drastis',
            'G04': 'Sapi pincak atau sulit berjalan',
            'G05': 'Sapi lebih sering berbaring',
            
            # Gejala Mulut
            'G06': 'Lepuh pada lidah dan gusi',
            'G07': 'Luka terbuka pada mulut',
            'G08': 'Air liur berlebihan (hipersalivasi)',
            'G09': 'Sulit mengunyah makanan',
            'G10': 'Bau mulut tidak sedap',
            
            # Gejala Kaki
            'G11': 'Lepuh pada sela-sela kuku',
            'G12': 'Pembengkakan pada kaki',
            'G13': 'Luka pada kuku',
            'G14': 'Kuku terlepas (kasus berat)',
            'G15': 'Kaki terasa hangat saat disentuh',
            
            # Gejala Ambing
            'G16': 'Lepuh pada ambing',
            'G17': 'Luka pada puting susu',
            'G18': 'Radang ambing (mastitis)',
            'G19': 'Warna ambing kemerahan',
            'G20': 'Nyeri saat pemerahan'
        }
        
        self.penyakit = {
            'P01': {
                'nama': 'PMK Ringan',
                'deskripsi': 'Tahap awal infeksi PMK dengan gejala ringan',
                'solusi': [
                    'Isolasi sapi yang terinfeksi',
                    'Berikan pakan lunak yang mudah dikonsumsi',
                    'Bersihkan dan obati luka dengan antiseptik',
                    'Konsultasikan dengan dokter hewan',
                    'Berikan vitamin untuk meningkatkan daya tahan tubuh'
                ]
            },
            'P02': {
                'nama': 'PMK Sedang',
                'deskripsi': 'Infeksi PMK dengan gejala yang jelas dan meluas',
                'solusi': [
                    'Isolasi total sapi yang terinfeksi',
                    'Lakukan desinfeksi kandang secara menyeluruh',
                    'Pengobatan simtomatis untuk demam dan nyeri',
                    'Perawatan intensif pada luka di mulut dan kaki',
                    'Vaksinasi untuk sapi yang sehat di sekitar',
                    'Lapor ke Dinas Peternakan setempat'
                ]
            },
            'P03': {
                'nama': 'PMK Berat dengan Komplikasi',
                'deskripsi': 'Infeksi PMK lanjut dengan komplikasi pada kaki dan ambing',
                'solusi': [
                    'Perawatan intensif oleh dokter hewan',
                    'Antibiotik untuk mencegah infeksi sekunder',
                    'Terapi cairan dan nutrisi jika tidak mau makan/minum',
                    'Perawatan luka profesional',
                    'Karantina ketat',
                    'Pelaporan wajib ke otoritas veteriner',
                    'Pembersihan dan desinfeksi total lingkungan'
                ]
            }
        }
        
        self.aturan = [
            # Aturan untuk PMK Ringan
            {
                'kode': 'R01',
                'gejala': ['G01', 'G02', 'G03'],
                'penyakit': 'P01',
                'cf': 0.6
            },
            {
                'kode': 'R02',
                'gejala': ['G01', 'G05', 'G08'],
                'penyakit': 'P01',
                'cf': 0.7
            },
            {
                'kode': 'R03',
                'gejala': ['G02', 'G04', 'G06'],
                'penyakit': 'P01',
                'cf': 0.8
            },
            
            # Aturan untuk PMK Sedang
            {
                'kode': 'R04',
                'gejala': ['G01', 'G06', 'G07', 'G08'],
                'penyakit': 'P02',
                'cf': 0.8
            },
            {
                'kode': 'R05',
                'gejala': ['G01', 'G11', 'G12', 'G15'],
                'penyakit': 'P02',
                'cf': 0.85
            },
            {
                'kode': 'R06',
                'gejala': ['G02', 'G06', 'G07', 'G09', 'G11'],
                'penyakit': 'P02',
                'cf': 0.9
            },
            
            # Aturan untuk PMK Berat
            {
                'kode': 'R07',
                'gejala': ['G01', 'G07', 'G11', 'G13', 'G14'],
                'penyakit': 'P03',
                'cf': 0.9
            },
            {
                'kode': 'R08',
                'gejala': ['G01', 'G02', 'G08', 'G11', 'G12', 'G14'],
                'penyakit': 'P03',
                'cf': 0.95
            },
            {
                'kode': 'R09',
                'gejala': ['G01', 'G08', 'G13', 'G16', 'G17', 'G19'],
                'penyakit': 'P03',
                'cf': 0.85
            },
            {
                'kode': 'R10',
                'gejala': ['G02', 'G09', 'G14', 'G18', 'G20'],
                'penyakit': 'P03',
                'cf': 0.9
            }
        ]


class ForwardChaining:
    def __init__(self):
        self.kb = KnowledgeBase()
        self.fakta = set()
        self.hasil = {}
        
    def reset(self):
        """Reset fakta dan hasil"""
        self.fakta = set()
        self.hasil = {}
        
    def tambah_gejala(self, gejala_list):
        """Menambahkan gejala yang teramati"""
        for gejala in gejala_list:
            self.fakta.add(gejala)
            
    def hitung_cf_combined(self, cf1, cf2):
        """Menghitung kombinasi Certainty Factor"""
        return cf1 + cf2 * (1 - cf1)
    
    def inferensi(self):
        """Melakukan inferensi forward chaining"""
        self.hasil = {}
        fakta_baru = True
        
        while fakta_baru:
            fakta_baru = False
            
            for aturan in self.kb.aturan:
                # Cek apakah semua gejala dalam aturan ada di fakta
                gejala_terpenuhi = all(g in self.fakta for g in aturan['gejala'])
                
                if gejala_terpenuhi:
                    penyakit = aturan['penyakit']
                    cf = aturan['cf']
                    
                    if penyakit not in self.hasil:
                        self.hasil[penyakit] = cf
                        fakta_baru = True
                    else:
                        # Kombinasikan CF jika penyakit sudah ada
                        cf_lama = self.hasil[penyakit]
                        cf_baru = self.hitung_cf_combined(cf_lama, cf)
                        if cf_baru > cf_lama:
                            self.hasil[penyakit] = cf_baru
                            fakta_baru = True
        
        return self.hasil
    
    def get_diagnosis(self):
        """Mendapatkan hasil diagnosis"""
        hasil_inferensi = self.inferensi()
        
        if not hasil_inferensi:
            return {
                'status': 'Tidak terdiagnosis',
                'message': 'Gejala yang diberikan tidak cukup untuk mendiagnosis PMK'
            }
        
        # Urutkan hasil berdasarkan CF tertinggi
        hasil_urut = sorted(hasil_inferensi.items(), key=lambda x: x[1], reverse=True)
        
        diagnosis = []
        for kode_penyakit, cf in hasil_urut:
            penyakit = self.kb.penyakit[kode_penyakit]
            diagnosis.append({
                'kode': kode_penyakit,
                'nama': penyakit['nama'],
                'deskripsi': penyakit['deskripsi'],
                'solusi': penyakit['solusi'],
                'cf': round(cf * 100, 2)
            })
        
        return {
            'status': 'terdiagnosis',
            'diagnosis': diagnosis,
            'gejala_teramati': [self.kb.gejala[g] for g in self.fakta]
        }
    
    def get_gejala_list(self):
        """Mendapatkan daftar semua gejala"""
        return self.kb.gejala


class Evaluator:
    def __init__(self, predictions_csv_path='results/predictions.csv'):
        self.predictions_csv_path = predictions_csv_path

    def _infer_true_label(self, image_path):
        if not image_path:
            return None
        p = image_path.lower()
        if 'healthy' in p or '/healthy/' in p or '\\healthy\\' in p:
            return 'sehat'
        # treat other dataset folders (e.g., FMD) as sick
        if 'fmd' in p or 'sakit' in p or 'fmd (' in p:
            return 'sakit'
        # fallback: if filename contains 'healthy'
        if 'healthy' in os.path.basename(p):
            return 'sehat'
        return 'sakit'

    def compute_confusion_matrix(self):
        import csv
        from collections import Counter

        counts = Counter()

        try:
            with open(self.predictions_csv_path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Some CSVs may not have the expected headers; be defensive
                    image_path = row.get('image_path') or row.get('image') or ''
                    pred = (row.get('prediction') or row.get('prediksi') or '').strip().lower()

                    true = self._infer_true_label(image_path)
                    if true is None or pred == '':
                        continue

                    # normalize labels to 'sehat'/'sakit'
                    if pred not in ('sehat', 'sakit'):
                        # try to map common variants
                        if 'sehat' in pred:
                            pred = 'sehat'
                        elif 'sakit' in pred:
                            pred = 'sakit'
                        else:
                            # skip unknown labels
                            continue

                    if true == 'sakit' and pred == 'sakit':
                        counts['TP'] += 1
                    elif true == 'sehat' and pred == 'sehat':
                        counts['TN'] += 1
                    elif true == 'sehat' and pred == 'sakit':
                        counts['FP'] += 1
                    elif true == 'sakit' and pred == 'sehat':
                        counts['FN'] += 1

        except FileNotFoundError:
            return None

        TP = counts.get('TP', 0)
        TN = counts.get('TN', 0)
        FP = counts.get('FP', 0)
        FN = counts.get('FN', 0)

        total = TP + TN + FP + FN
        accuracy = (TP + TN) / total if total else 0.0
        precision = TP / (TP + FP) if (TP + FP) else 0.0
        recall = TP / (TP + FN) if (TP + FN) else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

        metrics = {
            'confusion_matrix': {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN},
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'total': total
        }

        # save to results/model_performance.csv
        try:
            import csv
            out_path = os.path.join('results', 'model_performance.csv')
            with open(out_path, 'w', newline='', encoding='utf-8') as outf:
                writer = csv.writer(outf)
                writer.writerow(['accuracy', 'precision', 'recall', 'f1_score', 'TP', 'TN', 'FP', 'FN', 'total'])
                writer.writerow([metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score'], TP, TN, FP, FN, total])
        except Exception:
            pass

        return metrics
