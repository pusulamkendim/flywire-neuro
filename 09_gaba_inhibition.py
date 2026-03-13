"""
FlyWire Analiz 09 - GABA İnhibisyon Analizi
Beyindeki fren sistemi: GABA nöronları neyi, nerede ve nasıl susturuyor?
"""
import pandas as pd

conn = pd.read_feather("data/proofread_connections_783.feather")
ann = pd.read_csv("data/neuron_annotations.tsv", sep="\t", low_memory=False)

gaba_neurons = ann[ann['top_nt'] == 'gaba']
gaba_ids = set(gaba_neurons['root_id'])

print("=" * 80)
print("GABA NÖRONLARİ — KİMLER?")
print("=" * 80)
print(f"\nToplam: {len(gaba_neurons):,} / {len(ann):,} ({len(gaba_neurons)/len(ann)*100:.1f}%)")
print(f"Güven skoru: {gaba_neurons['top_nt_conf'].mean():.3f}")

print("\n--- Süper Sınıf ---")
for cls, count in gaba_neurons['super_class'].value_counts().items():
    pct = count / len(gaba_neurons) * 100
    bar = "█" * int(pct / 2)
    print(f"  {cls:<25} {count:>5}  ({pct:>5.1f}%)  {bar}")

print("\n--- Flow ---")
for f, count in gaba_neurons['flow'].value_counts().items():
    pct = count / len(gaba_neurons) * 100
    print(f"  {str(f):<25} {count:>5}  ({pct:>5.1f}%)")

print("\n--- Hücre Sınıfı (ilk 20) ---")
for cls, count in gaba_neurons['cell_class'].value_counts().head(20).items():
    pct = count / len(gaba_neurons) * 100
    print(f"  {str(cls):<25} {count:>5}  ({pct:>5.1f}%)")

print("\n--- Hücre Tipi (ilk 20) ---")
for ct, count in gaba_neurons['cell_type'].value_counts().head(20).items():
    print(f"  {str(ct):<25} {count:>5}")

# GABA sinyali nereye
print("\n" + "=" * 80)
print("GABA SİNYALİ NEREYE GİDİYOR?")
print("=" * 80)

gaba_conn = conn[conn['gaba_avg'] > 0.5]
print(f"\nGABA-baskın bağlantı: {len(gaba_conn):,}")
print(f"Toplam sinaps: {gaba_conn['syn_count'].sum():,}")

print("\n--- Hedef bölgeler ---")
gaba_targets = gaba_conn.groupby('neuropil')['syn_count'].sum().sort_values(ascending=False)
for region, syns in gaba_targets.head(20).items():
    pct = syns / gaba_targets.sum() * 100
    bar = "█" * int(pct / 2)
    print(f"  {region:<15} {syns:>10,} sinaps  ({pct:>5.1f}%)  {bar}")

# GABA kimi susturuyor (hedef NT tipi)
print("\n" + "=" * 80)
print("GABA KİMİ SUSTURUYOR? (Hedef NT tipi)")
print("=" * 80)

da_ids = set(ann[ann['top_nt'] == 'dopamine']['root_id'])
ser_ids = set(ann[ann['top_nt'] == 'serotonin']['root_id'])
ach_ids = set(ann[ann['top_nt'] == 'acetylcholine']['root_id'])
glut_ids = set(ann[ann['top_nt'] == 'glutamate']['root_id'])
oct_ids = set(ann[ann['top_nt'] == 'octopamine']['root_id'])

targets_nt = [
    ("GABA → ACh", ach_ids, len(ann[ann['top_nt'] == 'acetylcholine'])),
    ("GABA → GABA", gaba_ids, len(gaba_neurons)),
    ("GABA → Glutamate", glut_ids, len(ann[ann['top_nt'] == 'glutamate'])),
    ("GABA → Dopamine", da_ids, len(ann[ann['top_nt'] == 'dopamine'])),
    ("GABA → Serotonin", ser_ids, len(ann[ann['top_nt'] == 'serotonin'])),
    ("GABA → Octopamine", oct_ids, len(ann[ann['top_nt'] == 'octopamine'])),
]

print(f"\n{'Yön':<25} {'Bağlantı':>10} {'Sinaps':>10} {'Hedef Nöron':>12} {'Sinaps/Nöron':>14}")
print("-" * 75)
for label, target, target_count in targets_nt:
    c = conn[(conn['pre_pt_root_id'].isin(gaba_ids)) & (conn['post_pt_root_id'].isin(target))]
    syns = c['syn_count'].sum()
    per_neuron = syns / target_count if target_count > 0 else 0
    print(f"  {label:<25} {len(c):>8,} {syns:>10,} {target_count:>10,} {per_neuron:>12.1f}")

# Ters yön: Kim GABA'yı uyarıyor?
print("\n--- Kim GABA'ya sinyal gönderiyor? ---")
print(f"\n{'Yön':<25} {'Bağlantı':>10} {'Sinaps':>10}")
print("-" * 50)
for label, source in [("ACh → GABA", ach_ids), ("GABA → GABA", gaba_ids),
                       ("Glut → GABA", glut_ids), ("DA → GABA", da_ids),
                       ("SER → GABA", ser_ids), ("OCT → GABA", oct_ids)]:
    c = conn[(conn['pre_pt_root_id'].isin(source)) & (conn['post_pt_root_id'].isin(gaba_ids))]
    print(f"  {label:<25} {len(c):>8,} {c['syn_count'].sum():>10,}")

# Kritik devrelerdeki GABA etkisi
print("\n" + "=" * 80)
print("GABA KRİTİK DEVRELERDEKİ ETKİSİ")
print("=" * 80)

pam_ids = set(ann[ann['cell_type'].str.startswith('PAM', na=False)]['root_id'])
ppl1_ids = set(ann[ann['cell_type'].str.startswith('PPL1', na=False)]['root_id'])
kenyon_ids = set(ann[ann['cell_class'] == 'Kenyon_Cell']['root_id'])
mbon_ids = set(ann[ann['cell_class'] == 'MBON']['root_id'])
orn_ids = set(ann[ann['cell_class'] == 'olfactory']['root_id'])
alpn_ids = set(ann[ann['cell_class'] == 'ALPN']['root_id'])
alln_ids = set(ann[ann['cell_class'] == 'ALLN']['root_id'])

# Central complex
cx_regions = ['FB', 'EB', 'PB', 'NO']
cx_ids = set(ann[ann['cell_class'].isin(cx_regions)]['root_id'])

targets_circuit = [
    ("GABA → PAM (ödül)", pam_ids),
    ("GABA → PPL1 (ceza)", ppl1_ids),
    ("GABA → Kenyon Cell", kenyon_ids),
    ("GABA → MBON (çıkış)", mbon_ids),
    ("GABA → ORN (koku reseptör)", orn_ids),
    ("GABA → ALPN (koku proj.)", alpn_ids),
    ("GABA → ALLN (koku yerel)", alln_ids),
]

print(f"\n{'Hedef':<30} {'Bağlantı':>10} {'Sinaps':>10}")
print("-" * 55)
for label, target in targets_circuit:
    c = conn[(conn['pre_pt_root_id'].isin(gaba_ids)) & (conn['post_pt_root_id'].isin(target))]
    print(f"  {label:<30} {len(c):>8,} {c['syn_count'].sum():>10,}")

# GABA vs ACh dengesi bölge bazında
print("\n" + "=" * 80)
print("GABA vs ACh DENGESİ (Bölge bazında)")
print("=" * 80)

ach_conn = conn[conn['ach_avg'] > 0.5]
gaba_by_region = gaba_conn.groupby('neuropil')['syn_count'].sum()
ach_by_region = ach_conn.groupby('neuropil')['syn_count'].sum()

all_regions = set(gaba_by_region.index) | set(ach_by_region.index)
region_balance = []
for r in all_regions:
    g = gaba_by_region.get(r, 0)
    a = ach_by_region.get(r, 0)
    total = g + a
    if total > 10000:
        region_balance.append((r, g, a, total, g / total * 100))

region_balance.sort(key=lambda x: x[4], reverse=True)

print(f"\n{'Bölge':<15} {'GABA':>10} {'ACh':>10} {'GABA %':>8}  Denge")
print("-" * 65)
for r, g, a, total, pct in region_balance[:25]:
    bar_g = "▓" * int(pct / 5)
    bar_a = "░" * int((100 - pct) / 5)
    label = "← GABA baskın" if pct > 50 else "→ ACh baskın"
    print(f"  {r:<15} {g:>8,} {a:>8,} {pct:>6.1f}%  {bar_g}{bar_a}  {label}")

# GABA kendini ne kadar inhibe ediyor?
print("\n" + "=" * 80)
print("GABA KENDİNİ İNHİBE EDİYOR MU?")
print("=" * 80)

gaba_to_gaba = conn[(conn['pre_pt_root_id'].isin(gaba_ids)) & (conn['post_pt_root_id'].isin(gaba_ids))]
gaba_to_all = conn[conn['pre_pt_root_id'].isin(gaba_ids)]
self_pct = gaba_to_gaba['syn_count'].sum() / gaba_to_all['syn_count'].sum() * 100

print(f"\n  GABA → GABA:    {gaba_to_gaba['syn_count'].sum():>10,} sinaps")
print(f"  GABA → Tüm:     {gaba_to_all['syn_count'].sum():>10,} sinaps")
print(f"  Kendi kendini inhibisyon: {self_pct:.1f}%")

# GABA nöronlarının GABA alan/almayan karşılaştırması
gaba_inhib_neurons = ann[ann['cell_class'].isin(['Kenyon_Cell', 'MBON'])]
for cls in ['Kenyon_Cell', 'MBON']:
    cls_ids = set(ann[ann['cell_class'] == cls]['root_id'])
    gaba_in = conn[(conn['pre_pt_root_id'].isin(gaba_ids)) & (conn['post_pt_root_id'].isin(cls_ids))]
    total_in = conn[conn['post_pt_root_id'].isin(cls_ids)]
    if total_in['syn_count'].sum() > 0:
        pct = gaba_in['syn_count'].sum() / total_in['syn_count'].sum() * 100
        print(f"\n  {cls} aldığı toplam sinapsın %{pct:.1f}'i GABA'dan")

# Motor sisteme GABA etkisi
print("\n" + "=" * 80)
print("GABA VE MOTOR SİSTEM")
print("=" * 80)

motor_ids = set(ann[ann['super_class'] == 'motor']['root_id'])
descending_ids = set(ann[ann['super_class'] == 'descending']['root_id'])

gaba_to_motor = conn[(conn['pre_pt_root_id'].isin(gaba_ids)) & (conn['post_pt_root_id'].isin(motor_ids))]
gaba_to_desc = conn[(conn['pre_pt_root_id'].isin(gaba_ids)) & (conn['post_pt_root_id'].isin(descending_ids))]

print(f"\n  GABA → Motor:       {gaba_to_motor['syn_count'].sum():>8,} sinaps")
print(f"  GABA → Descending:  {gaba_to_desc['syn_count'].sum():>8,} sinaps")
print(f"\n  Nöron başına motor+descending: {(gaba_to_motor['syn_count'].sum() + gaba_to_desc['syn_count'].sum())/len(gaba_ids):.1f} sinaps/nöron")

# Karşılaştırma
print("\n--- NT Motor Karşılaştırması ---")
for nt_name, nt_ids in [("Dopamine", da_ids), ("Serotonin", ser_ids), ("Octopamine", oct_ids), ("GABA", gaba_ids)]:
    to_motor = conn[(conn['pre_pt_root_id'].isin(nt_ids)) & (conn['post_pt_root_id'].isin(motor_ids))]
    to_desc = conn[(conn['pre_pt_root_id'].isin(nt_ids)) & (conn['post_pt_root_id'].isin(descending_ids))]
    total = to_motor['syn_count'].sum() + to_desc['syn_count'].sum()
    per_neuron = total / len(nt_ids) if len(nt_ids) > 0 else 0
    print(f"  {nt_name:<15} {total:>8,} sinaps  ({per_neuron:.1f}/nöron)")
