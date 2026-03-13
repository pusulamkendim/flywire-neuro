"""
FlyWire Analiz 13 - Tat Sinyal Yayılma Simülasyonu
"Sinek bir şey tattı — beyin adım adım ne yapıyor?"

Koku simülasyonuyla (12) aynı model, başlangıç noktası farklı:
- Koku: ORN (2,282 nöron, Antennal Lobe)
- Tat: Gustatory (408 nöron, GNG - alt beyin)
"""
import pandas as pd
import numpy as np
from collections import defaultdict

print("Veri yükleniyor...")
conn = pd.read_feather("data/proofread_connections_783.feather")
ann = pd.read_csv("data/neuron_annotations.tsv", sep="\t", low_memory=False)

# === Ağ yapısını hazırla ===
print("Ağ yapısı hazırlanıyor...")

neuron_nt = dict(zip(ann['root_id'], ann['top_nt']))

strong_conn = conn[conn['syn_count'] >= 3].copy()
print(f"Güçlü bağlantılar (>=3 sinaps): {len(strong_conn):,}")

pre_total = strong_conn.groupby('pre_pt_root_id')['syn_count'].sum()

def get_nt_sign(row):
    nt_scores = {
        'ach': row['ach_avg'], 'gaba': row['gaba_avg'], 'glut': row['glut_avg'],
        'oct': row['oct_avg'], 'ser': row['ser_avg'], 'da': row['da_avg'],
    }
    dominant = max(nt_scores, key=nt_scores.get)
    return -1.0 if dominant == 'gaba' else 1.0

print("NT etkileri hesaplanıyor...")
strong_conn['nt_sign'] = strong_conn.apply(get_nt_sign, axis=1)

print("Bağlantı haritası oluşturuluyor...")
adjacency = defaultdict(list)
for _, row in strong_conn.iterrows():
    pre = row['pre_pt_root_id']
    post = row['post_pt_root_id']
    weight = row['syn_count']
    sign = row['nt_sign']
    total = pre_total.get(pre, weight)
    norm_weight = (weight / total) * sign
    adjacency[pre].append((post, norm_weight))

print(f"Bağlantı haritası hazır: {len(adjacency):,} kaynak nöron")

# === Nöron gruplarını tanımla ===
gust_ids = set(ann[ann['cell_class'] == 'gustatory']['root_id'])
orn_ids = set(ann[ann['cell_class'] == 'olfactory']['root_id'])
alpn_ids = set(ann[ann['cell_class'] == 'ALPN']['root_id'])
alln_ids = set(ann[ann['cell_class'] == 'ALLN']['root_id'])
kenyon_ids = set(ann[ann['cell_class'] == 'Kenyon_Cell']['root_id'])
mbon_ids = set(ann[ann['cell_class'] == 'MBON']['root_id'])
pam_ids = set(ann[ann['cell_type'].str.startswith('PAM', na=False)]['root_id'])
ppl1_ids = set(ann[ann['cell_type'].str.startswith('PPL1', na=False)]['root_id'])
motor_ids = set(ann[ann['super_class'] == 'motor']['root_id'])
descending_ids = set(ann[ann['super_class'] == 'descending']['root_id'])

# CX (merkezi kompleks)
cx_classes = ['CX', 'FB', 'EB', 'PB']
cx_ids = set(ann[ann['cell_class'].isin(cx_classes)]['root_id'])

da_ids = set(ann[ann['top_nt'] == 'dopamine']['root_id'])
ser_ids = set(ann[ann['top_nt'] == 'serotonin']['root_id'])
gaba_ids = set(ann[ann['top_nt'] == 'gaba']['root_id'])
ach_ids = set(ann[ann['top_nt'] == 'acetylcholine']['root_id'])
glut_ids = set(ann[ann['top_nt'] == 'glutamate']['root_id'])
oct_ids = set(ann[ann['top_nt'] == 'octopamine']['root_id'])

all_tracked_groups = {
    'Gustatory (tat)': gust_ids,
    'ORN (koku)': orn_ids,
    'ALPN (koku proj.)': alpn_ids,
    'ALLN (koku yerel)': alln_ids,
    'Kenyon Cell': kenyon_ids,
    'MBON (çıkış)': mbon_ids,
    'PAM (ödül)': pam_ids,
    'PPL1 (ceza)': ppl1_ids,
    'Motor': motor_ids,
    'Descending': descending_ids,
}

nt_groups = {
    'ACh': ach_ids, 'GABA': gaba_ids, 'Glutamat': glut_ids,
    'Dopamin': da_ids, 'Serotonin': ser_ids, 'Oktopamin': oct_ids,
}

# Bölge eşleştirmesi
neuron_region = {}
region_count = conn.groupby(['post_pt_root_id', 'neuropil'])['syn_count'].sum().reset_index()
for nid, group in region_count.groupby('post_pt_root_id'):
    top_region = group.loc[group['syn_count'].idxmax(), 'neuropil']
    neuron_region[nid] = top_region

# === TAT PROFİLİ ===
print("\n" + "=" * 80)
print("TAT NÖRONLARİ PROFİLİ")
print("=" * 80)

gust_neurons = ann[ann['cell_class'] == 'gustatory']
print(f"\nToplam: {len(gust_neurons)} tat nöronu")
print(f"\n--- NT dağılımı ---")
for nt, count in gust_neurons['top_nt'].value_counts().items():
    pct = count / len(gust_neurons) * 100
    print(f"  {nt:<20} {count:>5}  ({pct:.1f}%)")
print(f"\n--- Ana tipler ---")
for ct, count in gust_neurons['cell_type'].value_counts().head(10).items():
    print(f"  {str(ct):<20} {count:>5}")

# Tat nöronlarının doğrudan hedefleri
print(f"\n--- Tat nöronlarının doğrudan hedefleri (ilk sinyal nereye?) ---")
gust_out = conn[conn['pre_pt_root_id'].isin(gust_ids)]
gust_targets = gust_out.merge(
    ann[['root_id', 'cell_class', 'super_class', 'top_nt']],
    left_on='post_pt_root_id', right_on='root_id', how='left'
)
target_class = gust_targets.groupby('cell_class')['syn_count'].sum().sort_values(ascending=False)
for cls, syns in target_class.head(15).items():
    pct = syns / target_class.sum() * 100
    print(f"  {str(cls):<25} {syns:>8,} sinaps  ({pct:.1f}%)")

print(f"\n--- Hedef bölgeler ---")
target_region = gust_out.groupby('neuropil')['syn_count'].sum().sort_values(ascending=False)
for r, syns in target_region.head(10).items():
    pct = syns / target_region.sum() * 100
    print(f"  {r:<15} {syns:>8,} sinaps  ({pct:.1f}%)")

# === SİMÜLASYON ===
print("\n" + "=" * 80)
print("SİMÜLASYON: SİNEK BİR ŞEY TATTI — BEYİN NE YAPIYOR?")
print("=" * 80)

# Başlangıç: tüm tat nöronlarını aktive et (408 nöron — kokuya göre az)
activation = defaultdict(float)
for gid in gust_ids:
    activation[gid] = 1.0

THRESHOLD = 0.1
DECAY = 0.3
NUM_STEPS = 10  # tat daha uzun sürebilir
GAIN = 2.0

print(f"\nBaşlangıç: {len(gust_ids)} tat nöronu aktive edildi (tümü)")
print(f"Parametreler: eşik={THRESHOLD}, bozunma={DECAY}, adım={NUM_STEPS}, kazanç={GAIN}")

print(f"\n{'='*90}")
print(f"{'Adım':<6} {'Aktif':>7} {'Yeni':>7} | {'Gust':>5} {'ORN':>5} {'ALPN':>5} {'ALLN':>5} {'KC':>5} {'MBON':>5} {'PAM':>4} {'PPL1':>4} {'Mot':>4} {'Desc':>4}")
print(f"{'-'*90}")

step_data = []

for step in range(NUM_STEPS + 1):
    active_neurons = {nid for nid, act in activation.items() if act > THRESHOLD}

    group_counts = {}
    for gname, gids in all_tracked_groups.items():
        group_counts[gname] = len(active_neurons & gids)

    nt_counts = {}
    for ntname, ntids in nt_groups.items():
        nt_counts[ntname] = len(active_neurons & ntids)

    region_activation = defaultdict(float)
    for nid in active_neurons:
        r = neuron_region.get(nid, 'unknown')
        region_activation[r] += activation[nid]

    avg_act = np.mean([v for v in activation.values() if v > THRESHOLD]) if active_neurons else 0

    step_info = {
        'step': step, 'total_active': len(active_neurons),
        'groups': group_counts, 'nts': nt_counts,
        'regions': dict(region_activation), 'avg_activation': avg_act,
    }
    step_data.append(step_info)

    new_this_step = len(active_neurons) - (step_data[step-1]['total_active'] if step > 0 else 0)

    gc = group_counts
    print(f"  {step:<6} {len(active_neurons):>5,} {new_this_step:>+6,} | "
          f"{gc.get('Gustatory (tat)',0):>5} "
          f"{gc.get('ORN (koku)',0):>5} "
          f"{gc.get('ALPN (koku proj.)',0):>5} "
          f"{gc.get('ALLN (koku yerel)',0):>5} "
          f"{gc.get('Kenyon Cell',0):>5} "
          f"{gc.get('MBON (çıkış)',0):>5} "
          f"{gc.get('PAM (ödül)',0):>4} "
          f"{gc.get('PPL1 (ceza)',0):>4} "
          f"{gc.get('Motor',0):>4} "
          f"{gc.get('Descending',0):>4}")

    if step == NUM_STEPS:
        break

    new_activation = defaultdict(float)
    for nid, act in activation.items():
        if act > THRESHOLD:
            new_activation[nid] += act * (1 - DECAY)

    for nid in active_neurons:
        act = activation[nid]
        if nid in adjacency:
            for post_id, weight in adjacency[nid]:
                signal = act * weight * GAIN
                new_activation[post_id] += signal

    activation = defaultdict(float)
    for nid, act in new_activation.items():
        activation[nid] = max(0.0, min(1.0, act))

# === Detaylı adım analizi ===
print(f"\n{'='*80}")
print("DETAYLI ADIM ANALİZİ")
print(f"{'='*80}")

for i, sd in enumerate(step_data):
    if i in [0, 1, 2, 3, 5, 8, NUM_STEPS]:
        print(f"\n--- Adım {i} ({sd['total_active']:,} aktif, ort: {sd['avg_activation']:.3f}) ---")

        print(f"\n  Devre Durumu:")
        for gname, count in sd['groups'].items():
            total = len(all_tracked_groups[gname])
            if total > 0:
                pct = count / total * 100
                bar = "█" * int(pct / 5)
                print(f"    {gname:<25} {count:>5} / {total:>5}  ({pct:>5.1f}%)  {bar}")

        print(f"\n  NT Aktivasyonu:")
        for ntname, count in sd['nts'].items():
            total = len(nt_groups[ntname])
            if total > 0:
                pct = count / total * 100
                bar = "▓" * int(pct / 5)
                print(f"    {ntname:<15} {count:>6} / {total:>6}  ({pct:>5.1f}%)  {bar}")

        top_regions = sorted(sd['regions'].items(), key=lambda x: x[1], reverse=True)[:10]
        if top_regions:
            print(f"\n  En Aktif Bölgeler:")
            for r, act in top_regions:
                print(f"    {r:<15} toplam aktivasyon: {act:>8.1f}")

# === Zaman çizelgesi ===
print(f"\n{'='*80}")
print("TAT İŞLEME ZAMAN ÇİZELGESİ")
print(f"{'='*80}")

events = []
for i, sd in enumerate(step_data):
    gc = sd['groups']
    if i == 0:
        events.append((i, f"Gustatory aktif ({gc['Gustatory (tat)']} tat nöronu tadi algıladı)"))
    if i > 0:
        prev = step_data[i-1]['groups']
        for gname in ['ORN (koku)', 'ALLN (koku yerel)', 'ALPN (koku proj.)', 'Kenyon Cell',
                       'MBON (çıkış)', 'PAM (ödül)', 'PPL1 (ceza)', 'Descending', 'Motor']:
            if gc[gname] > 0 and prev[gname] == 0:
                events.append((i, f"{gname} aktif oldu ({gc[gname]} nöron)"))
            elif gc[gname] > prev[gname] * 1.5 and gc[gname] - prev[gname] > 5:
                events.append((i, f"{gname} artış ({prev[gname]} → {gc[gname]})"))

print()
for step, event in events:
    marker = "⚡" if step == 0 else f"t+{step}"
    print(f"  [{marker}]  {event}")

# === NT devreye girme ===
print(f"\n{'='*80}")
print("NT SİSTEMLERİ DEVREYE GİRME SIRASI")
print(f"{'='*80}")

print(f"\n{'Adım':<6} ", end="")
for nt in ['ACh', 'GABA', 'Glutamat', 'Dopamin', 'Serotonin', 'Oktopamin']:
    print(f"{nt:>10}", end="")
print()
print("-" * 70)

for sd in step_data:
    print(f"  {sd['step']:<4} ", end="")
    for nt in ['ACh', 'GABA', 'Glutamat', 'Dopamin', 'Serotonin', 'Oktopamin']:
        count = sd['nts'][nt]
        total = len(nt_groups[nt])
        pct = count / total * 100 if total > 0 else 0
        print(f"{pct:>9.1f}%", end="")
    print()

# === Uyarıcı vs İnhibitör ===
print(f"\n{'='*80}")
print("UYARICI vs İNHİBİTÖR DENGE")
print(f"{'='*80}")

print(f"\n{'Adım':<6} {'Uyarıcı':>10} {'İnhibitör':>10} {'Oran':>10}")
print("-" * 40)
for sd in step_data:
    excit = sd['nts']['ACh'] + sd['nts']['Glutamat'] + sd['nts']['Dopamin'] + sd['nts']['Serotonin'] + sd['nts']['Oktopamin']
    inhib = sd['nts']['GABA']
    ratio = excit / inhib if inhib > 0 else float('inf')
    ratio_str = "∞" if ratio == float('inf') else f"{ratio:.1f}:1"
    print(f"  {sd['step']:<4} {excit:>8,} {inhib:>10,} {ratio_str:>10}")

# === KOKU vs TAT KARŞILAŞTIRMASI ===
print(f"\n{'='*80}")
print("KOKU vs TAT KARŞILAŞTIRMASI")
print(f"{'='*80}")

print(f"""
  Özellik                KOKU                    TAT
  ─────────────────────────────────────────────────────────
  Reseptör sayısı        2,282 ORN               408 Gustatory
  Başlangıç bölgesi      AL (Antennal Lobe)      GNG (alt beyin)
  NT profili             %28 serotonin           %77 asetilkolin
  Başlangıç aktif        684 (%30)               408 (%100)
""")

# Son adım karşılaştırması
sd_final = step_data[NUM_STEPS]
gc = sd_final['groups']
print(f"  Son adım (t+{NUM_STEPS}):")
print(f"    Toplam aktif:        {sd_final['total_active']:,}")
print(f"    Kenyon Cell:         {gc['Kenyon Cell']}")
print(f"    MBON:                {gc['MBON (çıkış)']}")
print(f"    PAM (ödül):          {gc['PAM (ödül)']}")
print(f"    PPL1 (ceza):         {gc['PPL1 (ceza)']}")
print(f"    Motor:               {gc['Motor']}")
print(f"    Descending:          {gc['Descending']}")
print(f"    Koku sistemi aktif:  ORN={gc['ORN (koku)']}, ALPN={gc['ALPN (koku proj.)']}, ALLN={gc['ALLN (koku yerel)']}")
