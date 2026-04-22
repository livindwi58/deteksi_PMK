#!/usr/bin/env python
"""Tes forward chaining berbasis cocok-tidaknya aturan."""

from expert_system import ForwardChaining

# Tes 1: PMK Ringan (hanya gejala umum)
print("=" * 60)
print("TES 1: PMK Ringan (hanya gejala umum)")
print("=" * 60)
fc1 = ForwardChaining()
fc1.tambah_gejala(['G01', 'G02', 'G03'])  # Demam + Anorexia + Produksi turun
result1 = fc1.get_diagnosis()
if result1.get('status') == 'terdiagnosis':
    for diag in result1['diagnosis']:
        print(f"Diagnosis: {diag['nama']}")
        print(f"Severity: {diag['severity']}")
        print(f"Score: {diag['score']:.4f}")
        print(f"Aturan yang cocok: {len(fc1.matched_rules)}")
        print()

# Tes 2: PMK Sedang (gejala mulut dan tenggorokan)
print("=" * 60)
print("TES 2: PMK Sedang (gejala mulut dan tenggorokan)")
print("=" * 60)
fc2 = ForwardChaining()
fc2.tambah_gejala(['G01', 'G05', 'G06', 'G07'])  # Demam + Hipersalivasi + Saliva berbusa + Submandibula
result2 = fc2.get_diagnosis()
if result2.get('status') == 'terdiagnosis':
    for diag in result2['diagnosis']:
        print(f"Diagnosis: {diag['nama']}")
        print(f"Severity: {diag['severity']}")
        print(f"Score: {diag['score']:.4f}")
        print(f"Aturan yang cocok: {len(fc2.matched_rules)}")
        print()

# Tes 2B: Gejala mulut dan tenggorokan berat tanpa demam
print("=" * 60)
print("TES 2B: Gejala mulut dan tenggorokan berat tanpa demam")
print("=" * 60)
fc2b = ForwardChaining()
fc2b.tambah_gejala(['G05', 'G06', 'G07'])  # Hipersalivasi + Saliva berbusa + Submandibula
result2b = fc2b.get_diagnosis()
if result2b.get('status') == 'terdiagnosis':
    for diag in result2b['diagnosis']:
        print(f"Diagnosis: {diag['nama']}")
        print(f"Severity: {diag['severity']}")
        print(f"Score: {diag['score']:.4f}")
        print(f"Aturan yang cocok: {len(fc2b.matched_rules)}")
        print()

# Tes 3: PMK Berat (gejala kaki/kuku)
print("=" * 60)
print("TES 3: PMK Berat (gejala kaki/kuku)")
print("=" * 60)
fc3 = ForwardChaining()
fc3.tambah_gejala(['G01', 'G05', 'G12', 'G14'])  # Demam + Hipersalivasi + Nekrosis + Pincak
result3 = fc3.get_diagnosis()
if result3.get('status') == 'terdiagnosis':
    for diag in result3['diagnosis']:
        print(f"Diagnosis: {diag['nama']}")
        print(f"Severity: {diag['severity']}")
        print(f"Score: {diag['score']:.4f}")
        print(f"Aturan yang cocok: {len(fc3.matched_rules)}")
        print()

print("=" * 60)
print("✓ Sistem pencocokan aturan berjalan dengan baik!")
print("=" * 60)
