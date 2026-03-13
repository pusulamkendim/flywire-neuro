"""
FlyWire Analiz 11 - Glutamat Analizi
Üçüncü büyük NT: hem uyarıcı hem inhibitör çift rolü var.
"""
import pandas as pd

conn = pd.read_feather("data/proofread_connections_783.feather")
ann = pd.read_csv("data/neuron_annotations.tsv", sep="\t", low_memory=False)

glut_neurons = ann[ann['top_nt'] == 'glutamate']
glut_ids = set(glut_neurons['root_id'])

print("=" * 80)
print("GLUTAMAT NÖRONLARİ — KİMLER?")
print("=" * 80)
print(f"\nToplam: {len(glut_neurons):,} / {len(ann):,} ({len(glut_neurons)/len(ann)*100:.1f}%)")
print(f"Güven skoru: {glut_neurons['top_nt_conf'].mean():.3f}")

print("\n--- Süper Sınıf ---")
for cls, count in glut_neurons['super_class'].value_counts().items():
    pct = count / len(glut_neurons) * 100
    bar = "█" * int(pct / 2)
    print(f"  {cls:<25} {count:>6}  ({pct:>5.1f}%)  {bar}")

print("\n--- Flow ---")
for f, count in glut_neurons['flow'].value_counts().items():
    pct = count / len(glut_neurons) * 100
    print(f"  {str(f):<25} {count:>6}  ({pct:>5.1f}%)")

print("\n--- Hücre Sınıfı (ilk 20) ---")
for cls, count in glut_neurons['cell_class'].value_counts().head(20).items():
    pct = count / len(glut_neurons) * 100
    print(f"  {str(cls):<25} {count:>6}  ({pct:>5.1f}%)")

print("\n--- Hücre Tipi (ilk 15) ---")
for ct, count in glut_neurons['cell_type'].value_counts().head(15).items():
    print(f"  {str(ct):<25} {count:>6}")

# Glutamat sinyali nereye
print("\n" + "=" * 80)
print("GLUTAMAT SİNYALİ NEREYE GİDİYOR?")
print("=" * 80)

glut_conn = conn[conn['glut_avg'] > 0.5]
print(f"\nGlutamat-baskın bağlantı: {len(glut_conn):,}")
print(f"Toplam sinaps: {glut_conn['syn_count'].sum():,}")

print("\n--- Hedef bölgeler ---")
glut_targets = glut_conn.groupby('neuropil')['syn_count'].sum().sort_values(ascending=False)
for region, syns in glut_targets.head(20).items():
    pct = syns / glut_targets.sum() * 100
    bar = "█" * int(pct / 2)
    print(f"  {region:<15} {syns:>10,} sinaps  ({pct:>5.1f}%)  {bar}")

# Glutamat bölge baskınlığı
print("\n" + "=" * 80)
print("GLUTAMAT BASKINLIĞI — Hangi bölgelerde Glutamat önde?")
print("=" * 80)

ach_conn = conn[conn['ach_avg'] > 0.5]
gaba_conn = conn[conn['gaba_avg'] > 0.5]
ach_by_region = ach_conn.groupby('neuropil')['syn_count'].sum()
glut_by_region = glut_conn.groupby('neuropil')['syn_count'].sum()
gaba_by_region = gaba_conn.groupby('neuropil')['syn_count'].sum()

all_regions = set(glut_by_region.index) | set(ach_by_region.index)
region_data = []
for r in all_regions:
    a = ach_by_region.get(r, 0)
    gl = glut_by_region.get(r, 0)
    ga = gaba_by_region.get(r, 0)
    total = a + gl + ga
    if total > 50000:
        region_data.append((r, a, gl, ga, total, gl / total * 100))

region_data.sort(key=lambda x: x[5], reverse=True)

print(f"\n{'Bölge':<15} {'Glut':>10} {'ACh':>10} {'GABA':>10} {'Glut%':>7}")
print("-" * 60)
for r, a, gl, ga, total, pct in region_data[:20]:
    print(f"  {r:<15} {gl:>8,} {a:>8,} {ga:>8,} {pct:>5.1f}%")

# Kritik devrelerde glutamat
print("\n" + "=" * 80)
print("GLUTAMAT KRİTİK DEVRELERDEKİ ETKİSİ")
print("=" * 80)

ach_ids = set(ann[ann['top_nt'] == 'acetylcholine']['root_id'])
da_ids = set(ann[ann['top_nt'] == 'dopamine']['root_id'])
ser_ids = set(ann[ann['top_nt'] == 'serotonin']['root_id'])
gaba_ids = set(ann[ann['top_nt'] == 'gaba']['root_id'])
oct_ids = set(ann[ann['top_nt'] == 'octopamine']['root_id'])

pam_ids = set(ann[ann['cell_type'].str.startswith('PAM', na=False)]['root_id'])
ppl1_ids = set(ann[ann['cell_type'].str.startswith('PPL1', na=False)]['root_id'])
kenyon_ids = set(ann[ann['cell_class'] == 'Kenyon_Cell']['root_id'])
mbon_ids = set(ann[ann['cell_class'] == 'MBON']['root_id'])
orn_ids = set(ann[ann['cell_class'] == 'olfactory']['root_id'])
alpn_ids = set(ann[ann['cell_class'] == 'ALPN']['root_id'])

targets_circuit = [
    ("Glut → Kenyon Cell", kenyon_ids),
    ("Glut → MBON (çıkış)", mbon_ids),
    ("Glut → PAM (ödül)", pam_ids),
    ("Glut → PPL1 (ceza)", ppl1_ids),
    ("Glut → ORN (koku)", orn_ids),
    ("Glut → ALPN (koku proj.)", alpn_ids),
]

print(f"\n{'Hedef':<30} {'Bağlantı':>10} {'Sinaps':>10}")
print("-" * 55)
for label, target in targets_circuit:
    c = conn[(conn['pre_pt_root_id'].isin(glut_ids)) & (conn['post_pt_root_id'].isin(target))]
    print(f"  {label:<30} {len(c):>8,} {c['syn_count'].sum():>10,}")

# Glutamat diğer NT'lerle etkileşim
print("\n" + "=" * 80)
print("GLUTAMAT DİĞER SİSTEMLERLE ETKİLEŞİM")
print("=" * 80)

pairs = [
    ("Glut → ACh", ach_ids),
    ("Glut → GABA", gaba_ids),
    ("Glut → Glut", glut_ids),
    ("Glut → DA", da_ids),
    ("Glut → SER", ser_ids),
    ("Glut → OCT", oct_ids),
]

print(f"\n{'Yön':<20} {'Bağlantı':>10} {'Sinaps':>10}")
print("-" * 45)
for label, target in pairs:
    c = conn[(conn['pre_pt_root_id'].isin(glut_ids)) & (conn['post_pt_root_id'].isin(target))]
    print(f"  {label:<20} {len(c):>8,} {c['syn_count'].sum():>10,}")

# Motor bağlantı
print("\n" + "=" * 80)
print("GLUTAMAT VE MOTOR SİSTEM")
print("=" * 80)

motor_ids = set(ann[ann['super_class'] == 'motor']['root_id'])
descending_ids = set(ann[ann['super_class'] == 'descending']['root_id'])

glut_to_motor = conn[(conn['pre_pt_root_id'].isin(glut_ids)) & (conn['post_pt_root_id'].isin(motor_ids))]
glut_to_desc = conn[(conn['pre_pt_root_id'].isin(glut_ids)) & (conn['post_pt_root_id'].isin(descending_ids))]

total_motor = glut_to_motor['syn_count'].sum() + glut_to_desc['syn_count'].sum()
print(f"\n  Glut → Motor:       {glut_to_motor['syn_count'].sum():>8,} sinaps")
print(f"  Glut → Descending:  {glut_to_desc['syn_count'].sum():>8,} sinaps")
print(f"  Nöron başına: {total_motor/len(glut_ids):.1f} sinaps/nöron")

# Glutamaterjik MBON'lar — özel analiz
print("\n" + "=" * 80)
print("GLUTAMATERJİK MBON'LAR — Özel Durum")
print("=" * 80)

mbon_all = ann[ann['cell_class'] == 'MBON']
print(f"\nTüm MBON'lar: {len(mbon_all)}")
for nt, count in mbon_all['top_nt'].value_counts().items():
    pct = count / len(mbon_all) * 100
    print(f"  {str(nt):<20} {count:>5}  ({pct:.1f}%)")

glut_mbon_ids = set(mbon_all[mbon_all['top_nt'] == 'glutamate']['root_id'])
ach_mbon_ids = set(mbon_all[mbon_all['top_nt'] == 'acetylcholine']['root_id'])
gaba_mbon_ids = set(mbon_all[mbon_all['top_nt'] == 'gaba']['root_id'])

print(f"\nGlutamaterjik MBON hedefleri:")
for label, mbon_set in [("Glut-MBON", glut_mbon_ids), ("ACh-MBON", ach_mbon_ids), ("GABA-MBON", gaba_mbon_ids)]:
    to_pam = conn[(conn['pre_pt_root_id'].isin(mbon_set)) & (conn['post_pt_root_id'].isin(pam_ids))]
    to_ppl1 = conn[(conn['pre_pt_root_id'].isin(mbon_set)) & (conn['post_pt_root_id'].isin(ppl1_ids))]
    print(f"  {label:<15} → PAM: {to_pam['syn_count'].sum():>6,}  → PPL1: {to_ppl1['syn_count'].sum():>6,}")
