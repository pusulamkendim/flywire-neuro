"""
FlyWire Analiz 05 - Dopamin-Serotonin Etkileşim Analizi
Bu iki sistem birbirine sinyal gönderiyor mu? Aralarında köprü nöronlar var mı?
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

conn = pd.read_feather("data/proofread_connections_783.feather")
ann = pd.read_csv("data/neuron_annotations.tsv", sep="\t", low_memory=False)

# Nöron setlerini oluştur
da_ids = set(ann[ann['top_nt'] == 'dopamine']['root_id'])
ser_ids = set(ann[ann['top_nt'] == 'serotonin']['root_id'])
all_ids = set(ann['root_id'])

print("=" * 80)
print("DOPAMİN ↔ SEROTONİN ETKİLEŞİM ANALİZİ")
print("=" * 80)
print(f"\nDopaminerjik nöron: {len(da_ids):,}")
print(f"Serotonerjik nöron: {len(ser_ids):,}")

# =====================================================================
# 1. Doğrudan bağlantılar: DA → SER ve SER → DA
# =====================================================================
print("\n" + "=" * 80)
print("1. DOĞRUDAN BAĞLANTILAR")
print("=" * 80)

# DA → SER: Dopamin nöronları serotonin nöronlarına sinyal gönderiyor mu?
da_to_ser = conn[
    (conn['pre_pt_root_id'].isin(da_ids)) &
    (conn['post_pt_root_id'].isin(ser_ids))
]

# SER → DA: Serotonin nöronları dopamin nöronlarına sinyal gönderiyor mu?
ser_to_da = conn[
    (conn['pre_pt_root_id'].isin(ser_ids)) &
    (conn['post_pt_root_id'].isin(da_ids))
]

# DA → DA (kendi içi)
da_to_da = conn[
    (conn['pre_pt_root_id'].isin(da_ids)) &
    (conn['post_pt_root_id'].isin(da_ids))
]

# SER → SER (kendi içi)
ser_to_ser = conn[
    (conn['pre_pt_root_id'].isin(ser_ids)) &
    (conn['post_pt_root_id'].isin(ser_ids))
]

print(f"\n  DA → SER :  {len(da_to_ser):>6,} bağlantı,  {da_to_ser['syn_count'].sum():>8,} sinaps")
print(f"  SER → DA :  {len(ser_to_da):>6,} bağlantı,  {ser_to_da['syn_count'].sum():>8,} sinaps")
print(f"  DA → DA  :  {len(da_to_da):>6,} bağlantı,  {da_to_da['syn_count'].sum():>8,} sinaps")
print(f"  SER → SER:  {len(ser_to_ser):>6,} bağlantı,  {ser_to_ser['syn_count'].sum():>8,} sinaps")

# Karşılaştırma için: rastgele beklenen değer
total_neurons = len(all_ids)
da_frac = len(da_ids) / total_neurons
ser_frac = len(ser_ids) / total_neurons
expected_da_ser = len(conn) * da_frac * ser_frac
expected_ser_da = len(conn) * ser_frac * da_frac

print(f"\n  Rastgele beklenen DA→SER bağlantı: {expected_da_ser:,.0f}")
print(f"  Gerçek DA→SER bağlantı:            {len(da_to_ser):,}")
print(f"  Oran (gerçek/beklenen):             {len(da_to_ser)/expected_da_ser:.2f}x")

print(f"\n  Rastgele beklenen SER→DA bağlantı: {expected_ser_da:,.0f}")
print(f"  Gerçek SER→DA bağlantı:            {len(ser_to_da):,}")
print(f"  Oran (gerçek/beklenen):             {len(ser_to_da)/expected_ser_da:.2f}x")

# =====================================================================
# 2. Hangi bölgelerde buluşuyorlar?
# =====================================================================
print("\n" + "=" * 80)
print("2. HANGİ BÖLGELERDe BULUŞUYORLAR?")
print("=" * 80)

print("\n--- DA → SER bağlantıları (bölge bazında) ---")
da_ser_regions = da_to_ser.groupby('neuropil')['syn_count'].sum().sort_values(ascending=False)
for region, syns in da_ser_regions.head(15).items():
    pct = syns / da_to_ser['syn_count'].sum() * 100
    bar = "█" * int(pct / 2)
    print(f"  {region:<15} {syns:>6,} sinaps  ({pct:>5.1f}%)  {bar}")

print("\n--- SER → DA bağlantıları (bölge bazında) ---")
ser_da_regions = ser_to_da.groupby('neuropil')['syn_count'].sum().sort_values(ascending=False)
for region, syns in ser_da_regions.head(15).items():
    pct = syns / ser_to_da['syn_count'].sum() * 100
    bar = "█" * int(pct / 2)
    print(f"  {region:<15} {syns:>6,} sinaps  ({pct:>5.1f}%)  {bar}")

# =====================================================================
# 3. Köprü nöronlar: DA ve SER'den aynı anda sinyal alan nöronlar
# =====================================================================
print("\n" + "=" * 80)
print("3. KÖPRÜ NÖRONLAR - HEM DA HEM SER'DEN SİNYAL ALANLAR")
print("=" * 80)

# DA'dan sinyal alan nöronlar
da_targets = set(conn[conn['pre_pt_root_id'].isin(da_ids)]['post_pt_root_id'])
# SER'den sinyal alan nöronlar
ser_targets = set(conn[conn['pre_pt_root_id'].isin(ser_ids)]['post_pt_root_id'])
# Her ikisinden de sinyal alanlar
bridge_neurons = da_targets & ser_targets

print(f"\n  DA'dan sinyal alan nöron:      {len(da_targets):>6,}")
print(f"  SER'den sinyal alan nöron:     {len(ser_targets):>6,}")
print(f"  Her ikisinden de alan (köprü): {len(bridge_neurons):>6,}")

# Köprü nöronların profili
bridge_ann = ann[ann['root_id'].isin(bridge_neurons)]

print(f"\n--- Köprü Nöronların Süper Sınıfı ---")
for cls, count in bridge_ann['super_class'].value_counts().head(10).items():
    pct = count / len(bridge_ann) * 100
    print(f"  {cls:<25} {count:>5}  ({pct:>5.1f}%)")

print(f"\n--- Köprü Nöronların Hücre Sınıfı (İlk 15) ---")
for cls, count in bridge_ann['cell_class'].value_counts().head(15).items():
    pct = count / len(bridge_ann) * 100
    print(f"  {cls:<25} {count:>5}  ({pct:>5.1f}%)")

print(f"\n--- Köprü Nöronların Nörotransmitter Tipi ---")
for nt, count in bridge_ann['top_nt'].value_counts().items():
    pct = count / len(bridge_ann) * 100
    bar = "█" * int(pct / 2)
    print(f"  {nt:<20} {count:>5}  ({pct:>5.1f}%)  {bar}")

# =====================================================================
# 4. En güçlü DA↔SER bağlantıları (en çok sinaps)
# =====================================================================
print("\n" + "=" * 80)
print("4. EN GÜÇLÜ DOĞRUDAN BAĞLANTILAR")
print("=" * 80)

print("\n--- DA → SER (en güçlü 10 bağlantı) ---")
top_da_ser = da_to_ser.nlargest(10, 'syn_count')
for _, row in top_da_ser.iterrows():
    pre_info = ann[ann['root_id'] == row['pre_pt_root_id']]
    post_info = ann[ann['root_id'] == row['post_pt_root_id']]
    pre_type = pre_info['cell_type'].values[0] if len(pre_info) > 0 else '?'
    post_type = post_info['cell_type'].values[0] if len(post_info) > 0 else '?'
    pre_class = pre_info['cell_class'].values[0] if len(pre_info) > 0 else '?'
    post_class = post_info['cell_class'].values[0] if len(post_info) > 0 else '?'
    print(f"  {str(pre_type):<20} ({str(pre_class):<15}) → {str(post_type):<20} ({str(post_class):<15})  {row['syn_count']:>3} sinaps  @{row['neuropil']}")

print("\n--- SER → DA (en güçlü 10 bağlantı) ---")
top_ser_da = ser_to_da.nlargest(10, 'syn_count')
for _, row in top_ser_da.iterrows():
    pre_info = ann[ann['root_id'] == row['pre_pt_root_id']]
    post_info = ann[ann['root_id'] == row['post_pt_root_id']]
    pre_type = pre_info['cell_type'].values[0] if len(pre_info) > 0 else '?'
    post_type = post_info['cell_type'].values[0] if len(post_info) > 0 else '?'
    pre_class = pre_info['cell_class'].values[0] if len(pre_info) > 0 else '?'
    post_class = post_info['cell_class'].values[0] if len(post_info) > 0 else '?'
    print(f"  {str(pre_type):<20} ({str(pre_class):<15}) → {str(post_type):<20} ({str(post_class):<15})  {row['syn_count']:>3} sinaps  @{row['neuropil']}")

# =====================================================================
# GRAFİKLER
# =====================================================================

# Grafik 1: Etkileşim şeması (bağlantı sayıları)
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Sol: Bağlantı matrisi
labels = ['Dopamin', 'Serotonin', 'Diğer']
matrix = np.array([
    [da_to_da['syn_count'].sum(), da_to_ser['syn_count'].sum(),
     conn[conn['pre_pt_root_id'].isin(da_ids)]['syn_count'].sum() - da_to_da['syn_count'].sum() - da_to_ser['syn_count'].sum()],
    [ser_to_da['syn_count'].sum(), ser_to_ser['syn_count'].sum(),
     conn[conn['pre_pt_root_id'].isin(ser_ids)]['syn_count'].sum() - ser_to_da['syn_count'].sum() - ser_to_ser['syn_count'].sum()],
    [0, 0, 0]  # placeholder
])

# Sadece DA ve SER arası
small_matrix = np.array([
    [da_to_da['syn_count'].sum(), da_to_ser['syn_count'].sum()],
    [ser_to_da['syn_count'].sum(), ser_to_ser['syn_count'].sum()]
])
small_labels = ['Dopamin', 'Serotonin']

im = axes[0].imshow(small_matrix, cmap='YlOrRd')
axes[0].set_xticks(range(2))
axes[0].set_xticklabels(['→ Dopamin', '→ Serotonin'], fontsize=11)
axes[0].set_yticks(range(2))
axes[0].set_yticklabels(['Dopamin →', 'Serotonin →'], fontsize=11)
axes[0].set_title('Sinaps Sayısı Matrisi\n(satır=gönderen, sütun=alan)', fontsize=13, fontweight='bold')

for i in range(2):
    for j in range(2):
        axes[0].text(j, i, f'{small_matrix[i,j]:,.0f}',
                     ha='center', va='center', fontsize=14, fontweight='bold',
                     color='white' if small_matrix[i,j] > 5000 else 'black')

plt.colorbar(im, ax=axes[0], shrink=0.8)

# Sağ: Buluşma bölgeleri karşılaştırması
top_regions = list(da_ser_regions.head(10).index)
da_ser_vals = [da_ser_regions.get(r, 0) for r in top_regions]
ser_da_vals = [ser_da_regions.get(r, 0) for r in top_regions]

x = np.arange(len(top_regions))
w = 0.35
axes[1].bar(x - w/2, da_ser_vals, w, label='DA → SER', color='#f39c12', alpha=0.85)
axes[1].bar(x + w/2, ser_da_vals, w, label='SER → DA', color='#9b59b6', alpha=0.85)
axes[1].set_xticks(x)
axes[1].set_xticklabels(top_regions, rotation=45, ha='right', fontsize=10)
axes[1].set_ylabel('Sinaps Sayısı', fontsize=11)
axes[1].set_title('DA ↔ SER Buluşma Bölgeleri', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/07_da_ser_interaction.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nKaydedildi: results/07_da_ser_interaction.png")

# Grafik 2: Köprü nöronların NT dağılımı
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Köprü nöronların NT dağılımı
bridge_nt = bridge_ann['top_nt'].value_counts()
colors_nt = {'acetylcholine': '#2ecc71', 'GABA': '#e74c3c', 'glutamate': '#3498db',
             'dopamine': '#f39c12', 'serotonin': '#9b59b6', 'octopamine': '#e67e22'}
bar_colors = [colors_nt.get(nt, 'gray') for nt in bridge_nt.index]

ax1.barh(range(len(bridge_nt)), bridge_nt.values, color=bar_colors)
ax1.set_yticks(range(len(bridge_nt)))
ax1.set_yticklabels(bridge_nt.index, fontsize=11)
ax1.set_xlabel('Nöron Sayısı', fontsize=11)
ax1.set_title(f'Köprü Nöronların NT Tipi\n(n={len(bridge_ann):,})', fontsize=13, fontweight='bold')
ax1.invert_yaxis()
for i, v in enumerate(bridge_nt.values):
    ax1.text(v + 50, i, f'{v:,}', va='center', fontsize=10)

# Köprü nöronların süper sınıf dağılımı
bridge_super = bridge_ann['super_class'].value_counts().head(8)
ax2.barh(range(len(bridge_super)), bridge_super.values, color='#1abc9c', alpha=0.85)
ax2.set_yticks(range(len(bridge_super)))
ax2.set_yticklabels(bridge_super.index, fontsize=11)
ax2.set_xlabel('Nöron Sayısı', fontsize=11)
ax2.set_title(f'Köprü Nöronların Süper Sınıfı', fontsize=13, fontweight='bold')
ax2.invert_yaxis()
for i, v in enumerate(bridge_super.values):
    ax2.text(v + 50, i, f'{v:,}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('results/08_bridge_neurons.png', dpi=150, bbox_inches='tight')
plt.close()
print("Kaydedildi: results/08_bridge_neurons.png")

# Grafik 3: Venn benzeri özet diagram
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# DA kutusu
da_box = plt.Rectangle((0.5, 6), 3.5, 3.5, fill=True, facecolor='#f39c12', alpha=0.2, edgecolor='#f39c12', linewidth=2)
ax.add_patch(da_box)
ax.text(2.25, 9, 'DOPAMİN', ha='center', fontsize=16, fontweight='bold', color='#e67e22')
ax.text(2.25, 8.3, f'{len(da_ids):,} nöron', ha='center', fontsize=12)
ax.text(2.25, 7.6, 'Merkezi beyin (%96)', ha='center', fontsize=10)
ax.text(2.25, 7.0, 'Kenyon Cell, PAM', ha='center', fontsize=10)
ax.text(2.25, 6.4, '→ Mushroom Body', ha='center', fontsize=10, style='italic')

# SER kutusu
ser_box = plt.Rectangle((6, 6), 3.5, 3.5, fill=True, facecolor='#9b59b6', alpha=0.2, edgecolor='#9b59b6', linewidth=2)
ax.add_patch(ser_box)
ax.text(7.75, 9, 'SEROTONİN', ha='center', fontsize=16, fontweight='bold', color='#8e44ad')
ax.text(7.75, 8.3, f'{len(ser_ids):,} nöron', ha='center', fontsize=12)
ax.text(7.75, 7.6, 'Duyusal (%68)', ha='center', fontsize=10)
ax.text(7.75, 7.0, 'Koku, Görme', ha='center', fontsize=10)
ax.text(7.75, 6.4, '→ Antennal Lobe', ha='center', fontsize=10, style='italic')

# Oklar
ax.annotate('', xy=(5.8, 8), xytext=(4.2, 8),
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2.5))
ax.text(5, 8.4, f'{da_to_ser["syn_count"].sum():,}\nsinaps', ha='center', fontsize=10, color='#e74c3c')

ax.annotate('', xy=(4.2, 7), xytext=(5.8, 7),
            arrowprops=dict(arrowstyle='->', color='#3498db', lw=2.5))
ax.text(5, 6.4, f'{ser_to_da["syn_count"].sum():,}\nsinaps', ha='center', fontsize=10, color='#3498db')

# Köprü kutusu
bridge_box = plt.Rectangle((2.5, 1), 5, 4, fill=True, facecolor='#1abc9c', alpha=0.15, edgecolor='#1abc9c', linewidth=2)
ax.add_patch(bridge_box)
ax.text(5, 4.5, f'KÖPRÜ NÖRONLAR', ha='center', fontsize=14, fontweight='bold', color='#16a085')
ax.text(5, 3.8, f'{len(bridge_neurons):,} nöron', ha='center', fontsize=13)
ax.text(5, 3.1, 'Hem DA hem SER\'den sinyal alıyor', ha='center', fontsize=10)
ax.text(5, 2.4, f'Çoğunluğu: Asetilkolin ({bridge_nt.iloc[0]:,})', ha='center', fontsize=10)
ax.text(5, 1.8, f'Ana konum: Optic ({bridge_super.iloc[0]:,}) + Central ({bridge_super.iloc[1]:,})', ha='center', fontsize=10)
ax.text(5, 1.2, 'İki sistemi birleştiren entegrasyon katmanı', ha='center', fontsize=10, style='italic', color='#16a085')

# Köprüye oklar
ax.annotate('', xy=(3.5, 5), xytext=(2.25, 6),
            arrowprops=dict(arrowstyle='->', color='#f39c12', lw=2, alpha=0.7))
ax.annotate('', xy=(6.5, 5), xytext=(7.75, 6),
            arrowprops=dict(arrowstyle='->', color='#9b59b6', lw=2, alpha=0.7))

ax.set_title('Dopamin ↔ Serotonin Etkileşim Şeması', fontsize=16, fontweight='bold', pad=20)

plt.savefig('results/09_interaction_diagram.png', dpi=150, bbox_inches='tight')
plt.close()
print("Kaydedildi: results/09_interaction_diagram.png")

print("\n=== TAMAMLANDI ===")
