import os
from collections import OrderedDict, defaultdict

DISEASE_ALIAS_MAP = {
    'P01': {'ORAL', 'ORAL_KUAT', 'P01'},
    'P02': {'PODAL', 'PODAL_KUAT', 'P02'},
    'P03': {'LAKTASI', 'LAKTASI_KUAT', 'P03'},
    'P05': {'AKUT_GENERAL', 'AKUT_GENERAL_KUAT', 'P05'},
}

DEFAULT_GEJALA_GROUP_TITLES = OrderedDict([
    ('umum', 'Gejala Umum / Sistemik'),
    ('mulut', 'Gejala Mulut / Oral'),
    ('kaki', 'Gejala Kaki / Kuku'),
    ('ambing', 'Gejala Ambing / Laktasi'),
    ('berat', 'Gejala Berat / Khusus'),
])


class KnowledgeBase:
    def __init__(self):
        self.gejala = {}
        self.penyakit = {}
        self.aturan = []
        self.gejala_groups = self._empty_groups()

        loaded = self._load_from_database()
        if not loaded:
            self._set_fallback_knowledge()

    def _empty_groups(self):
        return OrderedDict(
            (group_code, {'title': group_title, 'codes': []})
            for group_code, group_title in DEFAULT_GEJALA_GROUP_TITLES.items()
        )

    def _set_fallback_knowledge(self):
        self.gejala = {}
        self.penyakit = {}
        self.aturan = []
        self.gejala_groups = self._empty_groups()

    def _load_from_database(self):
        try:
            from utils.mysql_db import get_expert_knowledge_mysql

            knowledge = get_expert_knowledge_mysql()
            if not knowledge:
                return False

            gejala = knowledge.get('gejala') or {}
            penyakit = knowledge.get('penyakit') or {}
            aturan = knowledge.get('aturan') or []
            gejala_groups = knowledge.get('gejala_groups')

            if gejala and penyakit and aturan:
                self.gejala = gejala
                self.penyakit = penyakit
                self.aturan = aturan
                if gejala_groups:
                    self.gejala_groups = gejala_groups
                print('[EXPERT_SYSTEM] Knowledge base loaded from MySQL.')
                return True
        except Exception as e:
            print(f"[EXPERT_SYSTEM] Using fallback knowledge base (DB unavailable): {e}")
        return False

    def get_gejala_groups(self):
        return self.gejala_groups


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
    
    def inferensi(self):
        """Mencocokkan gejala dengan aturan (exact + partial matching)."""
        self.hasil = {}
        self.matched_rules = []
        known_facts = set(self.fakta)
        if not known_facts:
            self.disease_evidence = defaultdict(list)
            return self.hasil

        disease_stats = {}
        disease_evidence = defaultdict(list)

        disease_rule_stats = {}
        for aturan in self.kb.aturan:
            hasil = aturan['hasil']
            if hasil in self.kb.penyakit:
                disease_rule_stats.setdefault(hasil, {'total_rules': 0, 'max_rule_size': 0})
                disease_rule_stats[hasil]['total_rules'] += 1
                disease_rule_stats[hasil]['max_rule_size'] = max(
                    disease_rule_stats[hasil]['max_rule_size'],
                    len(aturan['gejala'])
                )

        for aturan in self.kb.aturan:
            hasil = aturan['hasil']
            if hasil not in self.kb.penyakit:
                continue

            rule_gejala = aturan.get('gejala', [])
            if not rule_gejala:
                continue

            matched_gejala = [g for g in rule_gejala if g in known_facts]
            matched_count = len(matched_gejala)
            total_count = len(rule_gejala)
            coverage = matched_count / total_count

            if matched_count == 0:
                continue

            disease_stats.setdefault(
                hasil,
                {
                    'matched_rules': 0,
                    'exact_rules': 0,
                    'best_coverage': 0.0,
                    'coverage_sum': 0.0,
                    'best_matched_count': 0,
                },
            )

            disease_stats[hasil]['matched_rules'] += 1
            disease_stats[hasil]['coverage_sum'] += coverage
            disease_stats[hasil]['best_coverage'] = max(disease_stats[hasil]['best_coverage'], coverage)
            disease_stats[hasil]['best_matched_count'] = max(disease_stats[hasil]['best_matched_count'], matched_count)

            if coverage >= 1.0:
                disease_stats[hasil]['exact_rules'] += 1

            rule_obj = {
                'kode': aturan['kode'],
                'hasil': hasil,
                'gejala': rule_gejala,
                'gejala_cocok': matched_gejala,
                'matched_count': matched_count,
                'total_count': total_count,
                'coverage': coverage,
                'rule_size': total_count,
                'deskripsi': aturan.get('deskripsi', ''),
            }

            if coverage >= 1.0:
                self.matched_rules.append(rule_obj)

            disease_evidence[hasil].append(rule_obj)

        min_score_threshold = 0.35
        for penyakit, stats in disease_stats.items():
            total_rules = disease_rule_stats.get(penyakit, {}).get('total_rules', 1) or 1
            matched_rules = stats['matched_rules']
            avg_coverage = stats['coverage_sum'] / matched_rules if matched_rules else 0.0
            support_ratio = matched_rules / total_rules
            evidence_strength = stats['best_matched_count'] / max(len(self.kb.gejala), 1)
            exact_bonus = 0.10 if stats['exact_rules'] > 0 else 0.0

            combined_score = (
                (stats['best_coverage'] * 0.50)
                + (avg_coverage * 0.25)
                + (support_ratio * 0.15)
                + (evidence_strength * 0.10)
                + exact_bonus
            )
            combined_score = min(combined_score, 0.99)

            if combined_score >= min_score_threshold:
                self.hasil[penyakit] = (combined_score, stats['best_matched_count'])

        self.disease_evidence = disease_evidence

        return self.hasil
    
    def get_diagnosis(self):
        """Mengambil hasil diagnosis yang paling sesuai."""
        if not self.kb.aturan:
            return {
                'status': 'Belum terdeteksi',
                'message': 'Data aturan sistem pakar belum tersedia di database. Silakan setup/seed database terlebih dahulu.'
            }

        hasil_inferensi = self.inferensi()
        
        if not hasil_inferensi:
            return {
                'status': 'Belum terdeteksi',
                'message': 'Gejala yang dipilih masih belum cukup untuk menentukan PMK.'
            }
        
        # Urutkan hasil dari yang paling cocok
        hasil_urut = sorted(hasil_inferensi.items(), key=lambda x: (x[1][0], x[1][1]), reverse=True)
        
        # Pemetaan tingkat keparahan
        severity_map = {
            'P01': 'oral',
            'P02': 'podal',
            'P03': 'laktasi',
            'P05': 'akut umum'
        }
        
        diagnosis = []
        for kode_penyakit, (score, _matched_count) in hasil_urut:
            penyakit = self.kb.penyakit[kode_penyakit]
            relevant_aliases = DISEASE_ALIAS_MAP.get(kode_penyakit, {kode_penyakit})
            evidence_rules = [rule for rule in self.matched_rules if rule['hasil'] in relevant_aliases]

            if not evidence_rules:
                all_evidence = list(self.disease_evidence.get(kode_penyakit, []))
                all_evidence.sort(key=lambda r: (r.get('coverage', 0.0), r.get('matched_count', 0)), reverse=True)
                evidence_rules = all_evidence[:3]

            diagnosis.append({
                'kode': kode_penyakit,
                'severity': severity_map.get(kode_penyakit, 'unknown'),
                'nama': penyakit['nama'],
                'deskripsi': penyakit['deskripsi'],
                'solusi': penyakit['solusi'],
                'score': score,
                'gejala_teramati': [g for g in self.fakta if g in self.kb.gejala],
                'bukti_aturan': evidence_rules,
                'jumlah_bukti': len(evidence_rules),
                'semua_diagnosis': diagnosis  # Will be populated after all diagnosis generated
            })
        
        filtered_diagnosis = diagnosis[:5]

        # Isi daftar hasil untuk ditampilkan ke pengguna
        for diag in filtered_diagnosis:
            diag['persentase'] = round(float(diag['score']) * 100.0, 2)
            diag['semua_diagnosis'] = [
                {
                    'nama': d['nama'],
                    'severity': d['severity'],
                    'score': d['score']
                }
                for d in diagnosis
            ]
        
        return {
            'status': 'terdiagnosis',
            'diagnosis': filtered_diagnosis,
            'gejala_teramati': [self.kb.gejala.get(g, g) for g in sorted(self.fakta)]
        }
    
    def get_gejala_list(self):
        """Mendapatkan daftar semua gejala"""
        return self.kb.gejala

    def get_gejala_groups(self):
        """Mendapatkan daftar gejala yang dikelompokkan per kategori"""
        return self.kb.get_gejala_groups()


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
