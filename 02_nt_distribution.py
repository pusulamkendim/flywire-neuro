"""
FlyWire Analiz 02 - Beyin Bölgelerinde Nörotransmitter Dağılımı
Her beyin bölgesinde hangi nörotransmitter baskın?
"""
import pandas as pd

df = pd.read_feather("data/proofread_connections_783.feather")

nt_cols = ['gaba_avg', 'ach_avg', 'glut_avg', 'da_avg', 'ser_avg', 'oct_avg']
nt_labels = ['GABA', 'Asetilkolin', 'Glutamat', 'Dopamin', 'Serotonin', 'Oktopamin']

print("=" * 80)
print("BEYİN BÖLGELERİNDE NÖROTRANSMITTER DAĞILIMI (sinaps ağırlıklı)")
print("=" * 80)

regions = df.groupby('neuropil').apply(
    lambda g: pd.Series({
        col: (g[col] * g['syn_count']).sum() / g['syn_count'].sum()
        for col in nt_cols
    } | {'toplam_sinaps': g['syn_count'].sum(), 'baglanti_sayisi': len(g)}),
    include_groups=False
).sort_values('toplam_sinaps', ascending=False)

print(f"\nToplam {len(regions)} beyin bölgesi bulundu.\n")

print(f"{'Bölge':<15} {'Sinaps':>10} {'Baskın NT':<15} {'Oran':>6}  Dağılım")
print("-" * 80)

for region, row in regions.head(20).iterrows():
    vals = [row[c] for c in nt_cols]
    max_idx = vals.index(max(vals))
    dominant = nt_labels[max_idx]
    dominant_pct = max(vals)

    bar = ""
    symbols = ['G', 'A', 'L', 'D', 'S', 'O']
    for i, v in enumerate(vals):
        count = int(v * 20)
        if count > 0:
            bar += symbols[i] * count

    print(f"{region:<15} {int(row['toplam_sinaps']):>10,} {dominant:<15} {dominant_pct:>5.1%}  {bar}")

print("\n  G=GABA  A=Asetilkolin  L=Glutamat  D=Dopamin  S=Serotonin  O=Oktopamin")

print("\n" + "=" * 80)
print("TÜM BEYİN GENELİNDE NÖROTRANSMITTER DAĞILIMI")
print("=" * 80)

total_synapses = df['syn_count'].sum()
for i, col in enumerate(nt_cols):
    weighted = (df[col] * df['syn_count']).sum() / total_synapses
    bar = "█" * int(weighted * 50)
    print(f"  {nt_labels[i]:<15} {weighted:>5.1%}  {bar}")

print(f"\n  Toplam sinaps: {total_synapses:,}")
