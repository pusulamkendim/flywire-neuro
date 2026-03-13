"""
FlyWire Analiz 03 - Dopamin ve Serotonin Odaklı Analiz
Bu nadir ama kritik sinyalleri gönderen nöronlar kim, nereye bağlı?
"""
import pandas as pd

conn = pd.read_feather("data/proofread_connections_783.feather")
ann = pd.read_csv("data/neuron_annotations.tsv", sep="\t", low_memory=False)

# === DOPAMİNERJİK NÖRONLAR ===
da_neurons = ann[ann['top_nt'] == 'dopamine']

print("=" * 80)
print("DOPAMİNERJİK NÖRONLAR - KİMLER?")
print("=" * 80)
print(f"\nToplam dopaminerjik nöron: {len(da_neurons):,} / {len(ann):,} ({len(da_neurons)/len(ann)*100:.1f}%)")
print(f"Ortalama güven skoru: {da_neurons['top_nt_conf'].mean():.3f}")

print("\n--- Süper Sınıf Dağılımı ---")
for cls, count in da_neurons['super_class'].value_counts().items():
    pct = count / len(da_neurons) * 100
    bar = "█" * int(pct / 2)
    print(f"  {cls:<25} {count:>5}  ({pct:>5.1f}%)  {bar}")

print("\n--- Hücre Sınıfı (İlk 15) ---")
for cls, count in da_neurons['cell_class'].value_counts().head(15).items():
    print(f"  {cls:<25} {count:>5}")

print("\n--- Hücre Tipi (İlk 15) ---")
for cls, count in da_neurons['cell_type'].value_counts().head(15).items():
    print(f"  {cls:<25} {count:>5}")

# === DOPAMİN SİNYALİ NEREYE GİDİYOR? ===
print("\n" + "=" * 80)
print("DOPAMİN SİNYALİ NEREYE GİDİYOR?")
print("=" * 80)

da_connections = conn[conn['da_avg'] > 0.5]
print(f"\nDopamin-baskın bağlantı sayısı: {len(da_connections):,} / {len(conn):,}")
print(f"Toplam sinaps: {da_connections['syn_count'].sum():,}")

print("\n--- Hedef Beyin Bölgeleri (sinaps sayısına göre) ---")
target_regions = da_connections.groupby('neuropil')['syn_count'].sum().sort_values(ascending=False)
total_da_syn = target_regions.sum()
for region, syns in target_regions.head(20).items():
    pct = syns / total_da_syn * 100
    bar = "█" * int(pct / 2)
    print(f"  {region:<15} {syns:>10,} sinaps  ({pct:>5.1f}%)  {bar}")

# === SEROTONERJİK NÖRONLAR ===
ser_neurons = ann[ann['top_nt'] == 'serotonin']

print("\n" + "=" * 80)
print("SEROTONERJİK NÖRONLAR - KİMLER?")
print("=" * 80)
print(f"\nToplam serotonerjik nöron: {len(ser_neurons):,} / {len(ann):,} ({len(ser_neurons)/len(ann)*100:.1f}%)")
print(f"Ortalama güven skoru: {ser_neurons['top_nt_conf'].mean():.3f}")

print("\n--- Süper Sınıf Dağılımı ---")
for cls, count in ser_neurons['super_class'].value_counts().items():
    pct = count / len(ser_neurons) * 100
    bar = "█" * int(pct / 2)
    print(f"  {cls:<25} {count:>5}  ({pct:>5.1f}%)  {bar}")

print("\n--- Hücre Sınıfı (İlk 15) ---")
for cls, count in ser_neurons['cell_class'].value_counts().head(15).items():
    print(f"  {cls:<25} {count:>5}")

# === SEROTONİN SİNYALİ NEREYE GİDİYOR? ===
print("\n" + "=" * 80)
print("SEROTONİN SİNYALİ NEREYE GİDİYOR?")
print("=" * 80)

ser_connections = conn[conn['ser_avg'] > 0.5]
print(f"\nSerotonin-baskın bağlantı sayısı: {len(ser_connections):,} / {len(conn):,}")
print(f"Toplam sinaps: {ser_connections['syn_count'].sum():,}")

print("\n--- Hedef Beyin Bölgeleri (sinaps sayısına göre) ---")
target_ser = ser_connections.groupby('neuropil')['syn_count'].sum().sort_values(ascending=False)
total_ser_syn = target_ser.sum()
for region, syns in target_ser.head(20).items():
    pct = syns / total_ser_syn * 100
    bar = "█" * int(pct / 2)
    print(f"  {region:<15} {syns:>10,} sinaps  ({pct:>5.1f}%)  {bar}")

# === KARŞILAŞTIRMA ===
print("\n" + "=" * 80)
print("DOPAMİN vs SEROTONİN - BÖLGE KARŞILAŞTIRMASI")
print("=" * 80)

da_by_region = da_connections.groupby('neuropil')['syn_count'].sum()
ser_by_region = ser_connections.groupby('neuropil')['syn_count'].sum()

all_regions = set(da_by_region.index) | set(ser_by_region.index)
comparison = []
for r in all_regions:
    da_s = da_by_region.get(r, 0)
    ser_s = ser_by_region.get(r, 0)
    total = da_s + ser_s
    if total > 1000:
        comparison.append((r, da_s, ser_s, da_s / total if total > 0 else 0))

comparison.sort(key=lambda x: x[3], reverse=True)

print(f"\n{'Bölge':<15} {'Dopamin':>10} {'Serotonin':>10} {'DA/SER Oranı':>12}  Kimde baskın?")
print("-" * 70)
for r, da_s, ser_s, ratio in comparison[:25]:
    label = "◄ DOPAMİN" if ratio > 0.6 else ("SEROTONİN ►" if ratio < 0.4 else "DENGELİ")
    print(f"  {r:<15} {da_s:>10,} {ser_s:>10,} {ratio:>11.1%}  {label}")
