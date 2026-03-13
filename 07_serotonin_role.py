"""
FlyWire Analiz 07 - Serotoninin Koku İşlemedeki Rolü
Serotonin ödül/ceza devresine karışıyor mu, yoksa sadece duyusal kalibrasyon mu?
"""
import pandas as pd

conn = pd.read_feather("data/proofread_connections_783.feather")
ann = pd.read_csv("data/neuron_annotations.tsv", sep="\t", low_memory=False)

ser_neurons = ann[ann['top_nt'] == 'serotonin']
ser_ids = set(ser_neurons['root_id'])

# === Serotoninin AL'deki rolü ===
print("=" * 80)
print("SEROTONİN ANTENNAL LOBE'DA NE YAPIYOR?")
print("=" * 80)

al_ser_conn = conn[
    (conn['pre_pt_root_id'].isin(ser_ids)) &
    (conn['neuropil'].isin(['AL_L', 'AL_R']))
]

al_ser_targets = al_ser_conn.groupby('post_pt_root_id')['syn_count'].sum().reset_index()
al_ser_targets = al_ser_targets.merge(
    ann[['root_id', 'cell_class', 'cell_type', 'top_nt', 'super_class', 'flow']],
    left_on='post_pt_root_id', right_on='root_id', how='left'
)

print(f"\nAL'de serotonin bağlantısı: {len(al_ser_conn):,} bağlantı, {al_ser_conn['syn_count'].sum():,} sinaps")

print("\n--- Serotonin AL'de KİME sinyal gönderiyor? (hücre sınıfı) ---")
target_class = al_ser_targets.groupby('cell_class')['syn_count'].sum().sort_values(ascending=False)
for cls, syns in target_class.head(15).items():
    pct = syns / target_class.sum() * 100
    bar = "█" * int(pct / 2)
    print(f"  {str(cls):<25} {syns:>8,} sinaps  ({pct:>5.1f}%)  {bar}")

print("\n--- Hedef nöronların NT tipi ---")
target_nt = al_ser_targets.groupby('top_nt')['syn_count'].sum().sort_values(ascending=False)
for nt, syns in target_nt.items():
    pct = syns / target_nt.sum() * 100
    bar = "█" * int(pct / 2)
    print(f"  {str(nt):<20} {syns:>8,} sinaps  ({pct:>5.1f}%)  {bar}")

# === AL'deki serotonerjik nöronlar ===
print("\n" + "=" * 80)
print("AL'DEKİ SEROTONERJİK NÖRONLAR KİM?")
print("=" * 80)

al_ser_neurons = ser_neurons[ser_neurons['cell_class'].isin(['olfactory', 'ALLN', 'ALPN'])]
print(f"\nKoku ile ilgili serotonerjik nöron: {len(al_ser_neurons)}")
print("\n--- Sınıf ---")
for cls, count in al_ser_neurons['cell_class'].value_counts().items():
    print(f"  {cls:<25} {count:>5}")
print("\n--- Flow ---")
for f, count in al_ser_neurons['flow'].value_counts().items():
    print(f"  {str(f):<25} {count:>5}")

# === Serotonin ödül/ceza devresine karışıyor mu? ===
print("\n" + "=" * 80)
print("SEROTONİN ÖDÜL/CEZA DEVRESİNE KARIŞIYOR MU?")
print("=" * 80)

pam_ids = set(ann[ann['cell_type'].str.startswith('PAM', na=False)]['root_id'])
ppl1_ids = set(ann[ann['cell_type'].str.startswith('PPL1', na=False)]['root_id'])
kenyon_ids = set(ann[ann['cell_class'] == 'Kenyon_Cell']['root_id'])

ser_to_pam = conn[(conn['pre_pt_root_id'].isin(ser_ids)) & (conn['post_pt_root_id'].isin(pam_ids))]
ser_to_ppl1 = conn[(conn['pre_pt_root_id'].isin(ser_ids)) & (conn['post_pt_root_id'].isin(ppl1_ids))]
ser_to_kenyon = conn[(conn['pre_pt_root_id'].isin(ser_ids)) & (conn['post_pt_root_id'].isin(kenyon_ids))]
pam_to_ser = conn[(conn['pre_pt_root_id'].isin(pam_ids)) & (conn['post_pt_root_id'].isin(ser_ids))]
ppl1_to_ser = conn[(conn['pre_pt_root_id'].isin(ppl1_ids)) & (conn['post_pt_root_id'].isin(ser_ids))]

print(f"\n  SER → PAM (ödül):    {ser_to_pam['syn_count'].sum():>6,} sinaps")
print(f"  SER → PPL1 (ceza):   {ser_to_ppl1['syn_count'].sum():>6,} sinaps")
print(f"  SER → Kenyon Cell:   {ser_to_kenyon['syn_count'].sum():>6,} sinaps")
print(f"  PAM → SER:           {pam_to_ser['syn_count'].sum():>6,} sinaps")
print(f"  PPL1 → SER:          {ppl1_to_ser['syn_count'].sum():>6,} sinaps")

ser_al_total = al_ser_conn['syn_count'].sum()
ser_reward_total = ser_to_pam['syn_count'].sum() + ser_to_ppl1['syn_count'].sum() + ser_to_kenyon['syn_count'].sum()
print(f"\n  Serotonin AL'deki sinaps:              {ser_al_total:>8,}")
print(f"  Serotonin ödül/ceza devresindeki sinaps: {ser_reward_total:>8,}")
print(f"  Oran (AL / ödül-ceza):                   {ser_al_total/ser_reward_total:.1f}x")

# === Koku işleme devresi detayı ===
print("\n" + "=" * 80)
print("SEROTONİN KOKU İŞLEME YOLUNU NASIL MODÜLE EDİYOR?")
print("=" * 80)

orn = ann[ann['cell_class'] == 'olfactory']
alpn = ann[ann['cell_class'] == 'ALPN']
alln = ann[ann['cell_class'] == 'ALLN']
orn_ids = set(orn['root_id'])
alpn_ids = set(alpn['root_id'])
alln_ids = set(alln['root_id'])

print(f"\n  Koku reseptörleri (ORN):     {len(orn_ids):,} nöron")
print(f"  Projeksiyon nöronları (ALPN): {len(alpn_ids):,} nöron")
print(f"  Yerel nöronlar (ALLN):       {len(alln_ids):,} nöron")

ser_alln = alln[alln['top_nt'] == 'serotonin']
ser_orn = orn[orn['top_nt'] == 'serotonin']
print(f"\n  Serotonerjik ALLN: {len(ser_alln)} / {len(alln)} ({len(ser_alln)/len(alln)*100:.1f}%)")
print(f"  Serotonerjik ORN:  {len(ser_orn)} / {len(orn)} ({len(ser_orn)/len(orn)*100:.1f}%)")

ser_alln_ids = set(ser_alln['root_id'])
ser_alln_to_orn = conn[(conn['pre_pt_root_id'].isin(ser_alln_ids)) & (conn['post_pt_root_id'].isin(orn_ids))]
ser_alln_to_alpn = conn[(conn['pre_pt_root_id'].isin(ser_alln_ids)) & (conn['post_pt_root_id'].isin(alpn_ids))]
ser_alln_to_alln = conn[(conn['pre_pt_root_id'].isin(ser_alln_ids)) & (conn['post_pt_root_id'].isin(alln_ids))]

print(f"\n  Serotonerjik yerel nöron hedefleri:")
print(f"    → ORN (reseptörlere geri bildirim):  {ser_alln_to_orn['syn_count'].sum():>6,} sinaps")
print(f"    → ALPN (projeksiyon nöronlarına):     {ser_alln_to_alpn['syn_count'].sum():>6,} sinaps")
print(f"    → ALLN (diğer yerel nöronlara):      {ser_alln_to_alln['syn_count'].sum():>6,} sinaps")

ser_orn_ids = set(ser_orn['root_id'])
ser_orn_to_alpn = conn[(conn['pre_pt_root_id'].isin(ser_orn_ids)) & (conn['post_pt_root_id'].isin(alpn_ids))]
ser_orn_to_alln = conn[(conn['pre_pt_root_id'].isin(ser_orn_ids)) & (conn['post_pt_root_id'].isin(alln_ids))]

print(f"\n  Serotonerjik koku reseptörü hedefleri:")
print(f"    → ALPN (projeksiyon nöronlarına):     {ser_orn_to_alpn['syn_count'].sum():>6,} sinaps")
print(f"    → ALLN (yerel nöronlara):             {ser_orn_to_alln['syn_count'].sum():>6,} sinaps")
