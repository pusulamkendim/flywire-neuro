"""
FlyWire Analiz 08 - Oktopamin Analizi
Böceklerin "adrenalin"i: savaş-kaç tepkisi, görme modülasyonu.
"""
import pandas as pd

conn = pd.read_feather("data/proofread_connections_783.feather")
ann = pd.read_csv("data/neuron_annotations.tsv", sep="\t", low_memory=False)

oct_neurons = ann[ann['top_nt'] == 'octopamine']
oct_ids = set(oct_neurons['root_id'])

print("=" * 80)
print("OKTOPAMİN NÖRONLARİ — KİMLER?")
print("=" * 80)
print(f"\nToplam: {len(oct_neurons):,} / {len(ann):,} ({len(oct_neurons)/len(ann)*100:.1f}%)")
print(f"Güven skoru: {oct_neurons['top_nt_conf'].mean():.3f}")

print("\n--- Süper Sınıf ---")
for cls, count in oct_neurons['super_class'].value_counts().items():
    pct = count / len(oct_neurons) * 100
    bar = "█" * int(pct / 2)
    print(f"  {cls:<25} {count:>5}  ({pct:>5.1f}%)  {bar}")

print("\n--- Flow ---")
for f, count in oct_neurons['flow'].value_counts().items():
    pct = count / len(oct_neurons) * 100
    print(f"  {str(f):<25} {count:>5}  ({pct:>5.1f}%)")

print("\n--- Hücre Sınıfı (ilk 15) ---")
for cls, count in oct_neurons['cell_class'].value_counts().head(15).items():
    print(f"  {str(cls):<25} {count:>5}")

print("\n--- Hücre Tipi (ilk 20) ---")
for ct, count in oct_neurons['cell_type'].value_counts().head(20).items():
    print(f"  {str(ct):<25} {count:>5}")

# Oktopamin sinyali nereye
print("\n" + "=" * 80)
print("OKTOPAMİN SİNYALİ NEREYE GİDİYOR?")
print("=" * 80)

oct_conn = conn[conn['oct_avg'] > 0.5]
print(f"\nOktopamin-baskın bağlantı: {len(oct_conn):,}")
print(f"Toplam sinaps: {oct_conn['syn_count'].sum():,}")

print("\n--- Hedef bölgeler ---")
oct_targets = oct_conn.groupby('neuropil')['syn_count'].sum().sort_values(ascending=False)
for region, syns in oct_targets.head(20).items():
    pct = syns / oct_targets.sum() * 100
    bar = "█" * int(pct / 2)
    print(f"  {region:<15} {syns:>10,} sinaps  ({pct:>5.1f}%)  {bar}")

# Diğer NT'lerle etkileşim
print("\n" + "=" * 80)
print("OKTOPAMİN DİĞER SİSTEMLERLE ETKİLEŞİM")
print("=" * 80)

da_ids = set(ann[ann['top_nt'] == 'dopamine']['root_id'])
ser_ids = set(ann[ann['top_nt'] == 'serotonin']['root_id'])
ach_ids = set(ann[ann['top_nt'] == 'acetylcholine']['root_id'])

pairs = [
    ("OCT → DA", oct_ids, da_ids),
    ("OCT → SER", oct_ids, ser_ids),
    ("OCT → ACh", oct_ids, ach_ids),
    ("OCT → OCT", oct_ids, oct_ids),
    ("DA → OCT", da_ids, oct_ids),
    ("SER → OCT", ser_ids, oct_ids),
    ("ACh → OCT", ach_ids, oct_ids),
]

print(f"\n{'Yön':<15} {'Bağlantı':>10} {'Sinaps':>10}")
print("-" * 40)
for label, pre, post in pairs:
    c = conn[(conn['pre_pt_root_id'].isin(pre)) & (conn['post_pt_root_id'].isin(post))]
    print(f"  {label:<15} {len(c):>8,} {c['syn_count'].sum():>10,}")

# Kritik hedefler
print("\n" + "=" * 80)
print("OKTOPAMİN KRİTİK BÖLGELERDEKİ ETKİSİ")
print("=" * 80)

pam_ids = set(ann[ann['cell_type'].str.startswith('PAM', na=False)]['root_id'])
ppl1_ids = set(ann[ann['cell_type'].str.startswith('PPL1', na=False)]['root_id'])
kenyon_ids = set(ann[ann['cell_class'] == 'Kenyon_Cell']['root_id'])
orn_ids = set(ann[ann['cell_class'] == 'olfactory']['root_id'])
alpn_ids = set(ann[ann['cell_class'] == 'ALPN']['root_id'])
alln_ids = set(ann[ann['cell_class'] == 'ALLN']['root_id'])

targets_check = [
    ("OCT → PAM (ödül)", pam_ids),
    ("OCT → PPL1 (ceza)", ppl1_ids),
    ("OCT → Kenyon Cell", kenyon_ids),
    ("OCT → ORN (koku reseptör)", orn_ids),
    ("OCT → ALPN (koku projeksiyon)", alpn_ids),
    ("OCT → ALLN (koku yerel)", alln_ids),
]

for label, target in targets_check:
    c = conn[(conn['pre_pt_root_id'].isin(oct_ids)) & (conn['post_pt_root_id'].isin(target))]
    print(f"  {label:<30} {len(c):>8,} bağlantı  {c['syn_count'].sum():>8,} sinaps")

# Motor nöronlarla ilişki
print("\n" + "=" * 80)
print("OKTOPAMİN VE MOTOR SİSTEM")
print("=" * 80)

motor_ids = set(ann[ann['super_class'] == 'motor']['root_id'])
descending_ids = set(ann[ann['super_class'] == 'descending']['root_id'])

oct_to_motor = conn[(conn['pre_pt_root_id'].isin(oct_ids)) & (conn['post_pt_root_id'].isin(motor_ids))]
oct_to_desc = conn[(conn['pre_pt_root_id'].isin(oct_ids)) & (conn['post_pt_root_id'].isin(descending_ids))]

print(f"\n  OCT → Motor:       {oct_to_motor['syn_count'].sum():>6,} sinaps")
print(f"  OCT → Descending:  {oct_to_desc['syn_count'].sum():>6,} sinaps")
print(f"\n  Nöron başına motor+descending: {(oct_to_motor['syn_count'].sum() + oct_to_desc['syn_count'].sum())/len(oct_ids):.1f} sinaps/nöron")
