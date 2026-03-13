"""
FlyWire Analiz 04 - Nörotransmitter Görselleştirme
Beyin bölgelerinde dopamin/serotonin dağılımı grafikleri.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

conn = pd.read_feather("data/proofread_connections_783.feather")
ann = pd.read_csv("data/neuron_annotations.tsv", sep="\t", low_memory=False)

nt_cols = ['ach_avg', 'gaba_avg', 'glut_avg', 'da_avg', 'ser_avg', 'oct_avg']
nt_labels = ['Acetylcholine', 'GABA', 'Glutamate', 'Dopamine', 'Serotonin', 'Octopamine']
nt_colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6', '#e67e22']

# =====================================================================
# GRAFİK 1: Tüm beyinde nörotransmitter pasta grafiği
# =====================================================================
total_synapses = conn['syn_count'].sum()
nt_weighted = []
for col in nt_cols:
    nt_weighted.append((conn[col] * conn['syn_count']).sum() / total_synapses)

fig, ax = plt.subplots(figsize=(8, 8))
wedges, texts, autotexts = ax.pie(
    nt_weighted, labels=nt_labels, colors=nt_colors,
    autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12}
)
for t in autotexts:
    t.set_fontsize(11)
    t.set_fontweight('bold')
ax.set_title('Drosophila Brain — Neurotransmitter Distribution\n(54.5M synapses, weighted)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/01_pie_overall.png', dpi=150, bbox_inches='tight')
plt.close()
print("Kaydedildi: results/01_pie_overall.png")

# =====================================================================
# GRAFİK 2: En büyük 15 beyin bölgesinde NT dağılımı (yığılmış bar)
# =====================================================================
regions = conn.groupby('neuropil').apply(
    lambda g: pd.Series({
        col: (g[col] * g['syn_count']).sum() / g['syn_count'].sum()
        for col in nt_cols
    } | {'toplam_sinaps': g['syn_count'].sum()}),
    include_groups=False
).sort_values('toplam_sinaps', ascending=False)

top15 = regions.head(15)

fig, ax = plt.subplots(figsize=(14, 7))
bottom = np.zeros(len(top15))
for i, col in enumerate(nt_cols):
    vals = top15[col].values
    ax.bar(range(len(top15)), vals, bottom=bottom, label=nt_labels[i], color=nt_colors[i])
    bottom += vals

ax.set_xticks(range(len(top15)))
ax.set_xticklabels(top15.index, rotation=45, ha='right', fontsize=11)
ax.set_ylabel('Proportion', fontsize=12)
ax.set_title('Neurotransmitter Distribution in Top 15 Brain Regions', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.set_ylim(0, 1.05)

# Sinaps sayılarını üstte göster
for i, (region, row) in enumerate(top15.iterrows()):
    ax.text(i, 1.01, f'{int(row["toplam_sinaps"]/1e6):.1f}M', ha='center', fontsize=8, color='gray')

plt.tight_layout()
plt.savefig('results/02_bar_regions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Kaydedildi: results/02_bar_regions.png")

# =====================================================================
# GRAFİK 3: Dopamin vs Serotonin hedef bölgeleri (karşılaştırmalı bar)
# =====================================================================
da_conn = conn[conn['da_avg'] > 0.5]
ser_conn = conn[conn['ser_avg'] > 0.5]

da_targets = da_conn.groupby('neuropil')['syn_count'].sum().sort_values(ascending=False)
ser_targets = ser_conn.groupby('neuropil')['syn_count'].sum().sort_values(ascending=False)

# Her iki sistemde de anlamlı olan bölgeleri al
all_regions = set(da_targets.head(15).index) | set(ser_targets.head(15).index)
comp_data = []
for r in all_regions:
    comp_data.append({
        'region': r,
        'dopamin': da_targets.get(r, 0),
        'serotonin': ser_targets.get(r, 0)
    })
comp_df = pd.DataFrame(comp_data)
comp_df['total'] = comp_df['dopamin'] + comp_df['serotonin']
comp_df = comp_df.sort_values('total', ascending=True).tail(20)

fig, ax = plt.subplots(figsize=(10, 10))
y_pos = range(len(comp_df))
bar_height = 0.35

ax.barh([y - bar_height/2 for y in y_pos], comp_df['dopamin'], bar_height,
        label='Dopamin', color='#f39c12', alpha=0.9)
ax.barh([y + bar_height/2 for y in y_pos], comp_df['serotonin'], bar_height,
        label='Serotonin', color='#9b59b6', alpha=0.9)

ax.set_yticks(y_pos)
ax.set_yticklabels(comp_df['region'], fontsize=11)
ax.set_xlabel('Synapse Count', fontsize=12)
ax.set_title('Dopamine vs Serotonin — Target Brain Regions', fontsize=14, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('results/03_da_vs_ser_targets.png', dpi=150, bbox_inches='tight')
plt.close()
print("Kaydedildi: results/03_da_vs_ser_targets.png")

# =====================================================================
# GRAFİK 4: Dopaminerjik nöron tipleri (hücre sınıfı)
# =====================================================================
da_neurons = ann[ann['top_nt'] == 'dopamine']
ser_neurons = ann[ann['top_nt'] == 'serotonin']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Dopamin - süper sınıf
da_super = da_neurons['super_class'].value_counts()
ax1.pie(da_super.values, labels=da_super.index, autopct='%1.1f%%',
        startangle=90, textprops={'fontsize': 9},
        colors=plt.cm.Oranges(np.linspace(0.3, 0.9, len(da_super))))
ax1.set_title(f'Dopaminergic Neurons\n(n={len(da_neurons):,})', fontsize=13, fontweight='bold')

# Serotonin - süper sınıf
ser_super = ser_neurons['super_class'].value_counts()
ax2.pie(ser_super.values, labels=ser_super.index, autopct='%1.1f%%',
        startangle=90, textprops={'fontsize': 9},
        colors=plt.cm.Purples(np.linspace(0.3, 0.9, len(ser_super))))
ax2.set_title(f'Serotonergic Neurons\n(n={len(ser_neurons):,})', fontsize=13, fontweight='bold')

plt.suptitle('Neuron Super Class Distribution', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/04_da_ser_neuron_types.png', dpi=150, bbox_inches='tight')
plt.close()
print("Kaydedildi: results/04_da_ser_neuron_types.png")

# =====================================================================
# GRAFİK 5: Dopamin hücre tipleri (top 15 bar chart)
# =====================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Dopamin hücre sınıfları
da_class = da_neurons['cell_class'].value_counts().head(10)
ax1.barh(range(len(da_class)), da_class.values, color='#f39c12', alpha=0.85)
ax1.set_yticks(range(len(da_class)))
ax1.set_yticklabels(da_class.index, fontsize=11)
ax1.set_xlabel('Neuron Count', fontsize=11)
ax1.set_title('Dopamine — Cell Classes', fontsize=13, fontweight='bold')
ax1.invert_yaxis()
for i, v in enumerate(da_class.values):
    ax1.text(v + 20, i, str(v), va='center', fontsize=10)

# Serotonin hücre sınıfları
ser_class = ser_neurons['cell_class'].value_counts().head(10)
ax2.barh(range(len(ser_class)), ser_class.values, color='#9b59b6', alpha=0.85)
ax2.set_yticks(range(len(ser_class)))
ax2.set_yticklabels(ser_class.index, fontsize=11)
ax2.set_xlabel('Neuron Count', fontsize=11)
ax2.set_title('Serotonin — Cell Classes', fontsize=13, fontweight='bold')
ax2.invert_yaxis()
for i, v in enumerate(ser_class.values):
    ax2.text(v + 5, i, str(v), va='center', fontsize=10)

plt.tight_layout()
plt.savefig('results/05_da_ser_cell_classes.png', dpi=150, bbox_inches='tight')
plt.close()
print("Kaydedildi: results/05_da_ser_cell_classes.png")

# =====================================================================
# GRAFİK 6: Beyin bölgelerinde DA vs SER baskınlık haritası (heatmap)
# =====================================================================
da_by_region = da_conn.groupby('neuropil')['syn_count'].sum()
ser_by_region = ser_conn.groupby('neuropil')['syn_count'].sum()

all_r = set(da_by_region.index) | set(ser_by_region.index)
heatmap_data = []
for r in all_r:
    da_s = da_by_region.get(r, 0)
    ser_s = ser_by_region.get(r, 0)
    total = da_s + ser_s
    if total > 500:
        ratio = da_s / total if total > 0 else 0.5
        heatmap_data.append({'region': r, 'da_ratio': ratio, 'total': total})

hm_df = pd.DataFrame(heatmap_data).sort_values('da_ratio', ascending=True)

fig, ax = plt.subplots(figsize=(12, 14))
colors = plt.cm.RdYlGn_r(hm_df['da_ratio'].values)

bars = ax.barh(range(len(hm_df)), hm_df['da_ratio'], color=colors, edgecolor='white', linewidth=0.5)
ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, linewidth=1)
ax.set_yticks(range(len(hm_df)))
ax.set_yticklabels(hm_df['region'], fontsize=8)
ax.set_xlabel('Dopamine Ratio (0=Serotonin, 1=Dopamine)', fontsize=11)
ax.set_title('Dopamine vs Serotonin Dominance Across Brain Regions', fontsize=14, fontweight='bold')
ax.set_xlim(0, 1)

# Etiketler
ax.text(0.15, len(hm_df) + 0.5, '◄ SEROTONIN', fontsize=12, color='#9b59b6', fontweight='bold')
ax.text(0.75, len(hm_df) + 0.5, 'DOPAMINE ►', fontsize=12, color='#f39c12', fontweight='bold')

plt.tight_layout()
plt.savefig('results/06_da_ser_dominance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Kaydedildi: results/06_da_ser_dominance.png")

print("\n=== TAMAMLANDI: 6 grafik results/ klasörüne kaydedildi ===")
