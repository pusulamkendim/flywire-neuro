"""
FlyWire Analiz 10 - Asetilkolin (ACh) Analizi
Beyindeki ana uyarıcı sinyal: nöronların %62'si ACh kullanıyor.
"""
import pandas as pd

conn = pd.read_feather("data/proofread_connections_783.feather")
ann = pd.read_csv("data/neuron_annotations.tsv", sep="\t", low_memory=False)

ach_neurons = ann[ann['top_nt'] == 'acetylcholine']
ach_ids = set(ach_neurons['root_id'])

print("=" * 80)
print("ASETİLKOLİN NÖRONLARİ — KİMLER?")
print("=" * 80)
print(f"\nToplam: {len(ach_neurons):,} / {len(ann):,} ({len(ach_neurons)/len(ann)*100:.1f}%)")
print(f"Güven skoru: {ach_neurons['top_nt_conf'].mean():.3f}")

print("\n--- Süper Sınıf ---")
for cls, count in ach_neurons['super_class'].value_counts().items():
    pct = count / len(ach_neurons) * 100
    bar = "█" * int(pct / 2)
    print(f"  {cls:<25} {count:>6}  ({pct:>5.1f}%)  {bar}")

print("\n--- Flow ---")
for f, count in ach_neurons['flow'].value_counts().items():
    pct = count / len(ach_neurons) * 100
    print(f"  {str(f):<25} {count:>6}  ({pct:>5.1f}%)")

print("\n--- Hücre Sınıfı (ilk 20) ---")
for cls, count in ach_neurons['cell_class'].value_counts().head(20).items():
    pct = count / len(ach_neurons) * 100
    print(f"  {str(cls):<25} {count:>6}  ({pct:>5.1f}%)")

print("\n--- Hücre Tipi (ilk 15) ---")
for ct, count in ach_neurons['cell_type'].value_counts().head(15).items():
    print(f"  {str(ct):<25} {count:>6}")

# ACh sinyali nereye
print("\n" + "=" * 80)
print("ACh SİNYALİ NEREYE GİDİYOR?")
print("=" * 80)

ach_conn = conn[conn['ach_avg'] > 0.5]
print(f"\nACh-baskın bağlantı: {len(ach_conn):,}")
print(f"Toplam sinaps: {ach_conn['syn_count'].sum():,}")

print("\n--- Hedef bölgeler ---")
ach_targets = ach_conn.groupby('neuropil')['syn_count'].sum().sort_values(ascending=False)
for region, syns in ach_targets.head(20).items():
    pct = syns / ach_targets.sum() * 100
    bar = "█" * int(pct / 2)
    print(f"  {region:<15} {syns:>10,} sinaps  ({pct:>5.1f}%)  {bar}")

# ACh bölge baskınlığı — ACh'nin diğer uyarıcılardan baskın olduğu yerler
print("\n" + "=" * 80)
print("ACh BASKINLIĞI — Hangi bölgelerde ACh açık ara önde?")
print("=" * 80)

glut_conn = conn[conn['glut_avg'] > 0.5]
ach_by_region = ach_conn.groupby('neuropil')['syn_count'].sum()
glut_by_region = glut_conn.groupby('neuropil')['syn_count'].sum()
gaba_conn = conn[conn['gaba_avg'] > 0.5]
gaba_by_region = gaba_conn.groupby('neuropil')['syn_count'].sum()

all_regions = set(ach_by_region.index)
region_data = []
for r in all_regions:
    a = ach_by_region.get(r, 0)
    gl = glut_by_region.get(r, 0)
    ga = gaba_by_region.get(r, 0)
    total = a + gl + ga
    if total > 50000:
        region_data.append((r, a, gl, ga, total, a / total * 100))

region_data.sort(key=lambda x: x[5], reverse=True)

print(f"\n{'Bölge':<15} {'ACh':>10} {'Glut':>10} {'GABA':>10} {'ACh%':>7}")
print("-" * 60)
for r, a, gl, ga, total, pct in region_data[:20]:
    print(f"  {r:<15} {a:>8,} {gl:>8,} {ga:>8,} {pct:>5.1f}%")

# Kritik devrelerdeki ACh
print("\n" + "=" * 80)
print("ACh KRİTİK DEVRELERDEKİ ETKİSİ")
print("=" * 80)

da_ids = set(ann[ann['top_nt'] == 'dopamine']['root_id'])
ser_ids = set(ann[ann['top_nt'] == 'serotonin']['root_id'])
gaba_ids = set(ann[ann['top_nt'] == 'gaba']['root_id'])
glut_ids = set(ann[ann['top_nt'] == 'glutamate']['root_id'])
oct_ids = set(ann[ann['top_nt'] == 'octopamine']['root_id'])

pam_ids = set(ann[ann['cell_type'].str.startswith('PAM', na=False)]['root_id'])
ppl1_ids = set(ann[ann['cell_type'].str.startswith('PPL1', na=False)]['root_id'])
kenyon_ids = set(ann[ann['cell_class'] == 'Kenyon_Cell']['root_id'])
mbon_ids = set(ann[ann['cell_class'] == 'MBON']['root_id'])
orn_ids = set(ann[ann['cell_class'] == 'olfactory']['root_id'])
alpn_ids = set(ann[ann['cell_class'] == 'ALPN']['root_id'])
alln_ids = set(ann[ann['cell_class'] == 'ALLN']['root_id'])

targets_circuit = [
    ("ACh → Kenyon Cell", kenyon_ids),
    ("ACh → MBON (çıkış)", mbon_ids),
    ("ACh → PAM (ödül)", pam_ids),
    ("ACh → PPL1 (ceza)", ppl1_ids),
    ("ACh → ORN (koku)", orn_ids),
    ("ACh → ALPN (koku proj.)", alpn_ids),
    ("ACh → ALLN (koku yerel)", alln_ids),
]

print(f"\n{'Hedef':<30} {'Bağlantı':>10} {'Sinaps':>10}")
print("-" * 55)
for label, target in targets_circuit:
    c = conn[(conn['pre_pt_root_id'].isin(ach_ids)) & (conn['post_pt_root_id'].isin(target))]
    print(f"  {label:<30} {len(c):>8,} {c['syn_count'].sum():>10,}")

# ACh diğer NT'lerle etkileşim
print("\n" + "=" * 80)
print("ACh DİĞER SİSTEMLERLE ETKİLEŞİM")
print("=" * 80)

pairs = [
    ("ACh → ACh", ach_ids),
    ("ACh → GABA", gaba_ids),
    ("ACh → Glut", glut_ids),
    ("ACh → DA", da_ids),
    ("ACh → SER", ser_ids),
    ("ACh → OCT", oct_ids),
]

print(f"\n{'Yön':<20} {'Bağlantı':>10} {'Sinaps':>10}")
print("-" * 45)
for label, target in pairs:
    c = conn[(conn['pre_pt_root_id'].isin(ach_ids)) & (conn['post_pt_root_id'].isin(target))]
    print(f"  {label:<20} {len(c):>8,} {c['syn_count'].sum():>10,}")

# Motor bağlantı
print("\n" + "=" * 80)
print("ACh VE MOTOR SİSTEM")
print("=" * 80)

motor_ids = set(ann[ann['super_class'] == 'motor']['root_id'])
descending_ids = set(ann[ann['super_class'] == 'descending']['root_id'])

ach_to_motor = conn[(conn['pre_pt_root_id'].isin(ach_ids)) & (conn['post_pt_root_id'].isin(motor_ids))]
ach_to_desc = conn[(conn['pre_pt_root_id'].isin(ach_ids)) & (conn['post_pt_root_id'].isin(descending_ids))]

total_motor = ach_to_motor['syn_count'].sum() + ach_to_desc['syn_count'].sum()
print(f"\n  ACh → Motor:       {ach_to_motor['syn_count'].sum():>8,} sinaps")
print(f"  ACh → Descending:  {ach_to_desc['syn_count'].sum():>8,} sinaps")
print(f"  Nöron başına: {total_motor/len(ach_ids):.1f} sinaps/nöron")
