"""
FlyWire Analiz 06 - Ödül vs Ceza Dopamin Devreleri
PAM (ödül) vs PPL1 (ceza) nöronlarının karşılaştırmalı analizi.
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

conn = pd.read_feather("data/proofread_connections_783.feather")
ann = pd.read_csv("data/neuron_annotations.tsv", sep="\t", low_memory=False)

# Nöron setleri
pam = ann[ann['cell_type'].str.startswith('PAM', na=False)]
ppl1 = ann[ann['cell_type'].str.startswith('PPL1', na=False)]
kenyon_ids = set(ann[ann['cell_class'] == 'Kenyon_Cell']['root_id'])
pam_ids = set(pam['root_id'])
ppl1_ids = set(ppl1['root_id'])

print("=" * 80)
print("DOPAMİN NÖRONLARİ: ÖDÜL vs CEZA")
print("=" * 80)

# DAN sınıfı
dan = ann[ann['cell_class'] == 'DAN']
print(f"\nToplam DAN nöronu: {len(dan)}")
print("\n--- Hücre Tipleri ---")
for ct, count in dan['cell_type'].value_counts().items():
    print(f"  {str(ct):<25} {count:>4}")

# PAM detayları
print(f"\n{'='*80}")
print("PAM NÖRONLARİ (ÖDÜL SİSTEMİ) — 'Bu iyiydi, tekrarla!'")
print(f"{'='*80}")
print(f"Toplam: {len(pam)}")
print("\n--- PAM alt tipleri ---")
for ct, count in pam['cell_type'].value_counts().items():
    print(f"  {str(ct):<25} {count:>4}")

pam_conn = conn[conn['pre_pt_root_id'].isin(pam_ids)]
print(f"\n--- PAM hedef bölgeleri ---")
pam_targets = pam_conn.groupby('neuropil')['syn_count'].sum().sort_values(ascending=False)
for region, syns in pam_targets.head(10).items():
    pct = syns / pam_targets.sum() * 100
    bar = "█" * int(pct / 2)
    print(f"  {region:<15} {syns:>8,} sinaps  ({pct:>5.1f}%)  {bar}")

# PPL1 detayları
print(f"\n{'='*80}")
print("PPL1 NÖRONLARİ (CEZA SİSTEMİ) — 'Bu kötüydü, kaç!'")
print(f"{'='*80}")
print(f"Toplam: {len(ppl1)}")
print("\n--- PPL1 alt tipleri ---")
for ct, count in ppl1['cell_type'].value_counts().items():
    print(f"  {str(ct):<25} {count:>4}")

ppl1_conn = conn[conn['pre_pt_root_id'].isin(ppl1_ids)]
print(f"\n--- PPL1 hedef bölgeleri ---")
ppl1_targets = ppl1_conn.groupby('neuropil')['syn_count'].sum().sort_values(ascending=False)
for region, syns in ppl1_targets.head(10).items():
    pct = syns / ppl1_targets.sum() * 100
    bar = "█" * int(pct / 2)
    print(f"  {region:<15} {syns:>8,} sinaps  ({pct:>5.1f}%)  {bar}")

# MB karşılaştırma
print(f"\n{'='*80}")
print("MUSHROOM BODY'DE PAM vs PPL1")
print(f"{'='*80}")
mb_regions = sorted(set(
    [r for r in pam_targets.index if r.startswith('MB_')] +
    [r for r in ppl1_targets.index if r.startswith('MB_')]
))
print(f"\n{'MB Bölgesi':<15} {'PAM (ödül)':>12} {'PPL1 (ceza)':>12} {'Baskın':>10}")
print("-" * 55)
for r in mb_regions:
    p = pam_targets.get(r, 0)
    pp = ppl1_targets.get(r, 0)
    dominant = "ÖDÜL" if p > pp else "CEZA"
    print(f"  {r:<15} {p:>10,} {pp:>10,}  {dominant}")

# Ortak Kenyon hedefleri
print(f"\n{'='*80}")
print("PAM vs PPL1: AYNI KENYON HÜCRELERİNE Mİ BAĞLANIYOR?")
print(f"{'='*80}")
pam_to_kenyon = conn[(conn['pre_pt_root_id'].isin(pam_ids)) & (conn['post_pt_root_id'].isin(kenyon_ids))]
ppl1_to_kenyon = conn[(conn['pre_pt_root_id'].isin(ppl1_ids)) & (conn['post_pt_root_id'].isin(kenyon_ids))]
pam_kenyon_targets = set(pam_to_kenyon['post_pt_root_id'])
ppl1_kenyon_targets = set(ppl1_to_kenyon['post_pt_root_id'])
shared_kenyon = pam_kenyon_targets & ppl1_kenyon_targets

print(f"\n  PAM → Kenyon Cell:   {len(pam_kenyon_targets):,} farklı Kenyon hücresi")
print(f"  PPL1 → Kenyon Cell:  {len(ppl1_kenyon_targets):,} farklı Kenyon hücresi")
print(f"  Ortak hedef:         {len(shared_kenyon):,} HER İKİSİNDEN de sinyal alıyor")
print(f"  Toplam Kenyon Cell:  {len(kenyon_ids):,}")
print(f"  Ortak / Toplam:      {len(shared_kenyon)/len(kenyon_ids)*100:.1f}%")

# =====================================================================
# GRAFİKLER
# =====================================================================

# Grafik 1: PAM vs PPL1 MB bölgelerinde
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Sol: MB bölgeleri karşılaştırma
mb_pam = [pam_targets.get(r, 0) for r in mb_regions]
mb_ppl1 = [ppl1_targets.get(r, 0) for r in mb_regions]

x = np.arange(len(mb_regions))
w = 0.35
axes[0].bar(x - w/2, mb_pam, w, label='PAM (ödül)', color='#2ecc71', alpha=0.85)
axes[0].bar(x + w/2, mb_ppl1, w, label='PPL1 (ceza)', color='#e74c3c', alpha=0.85)
axes[0].set_xticks(x)
axes[0].set_xticklabels(mb_regions, rotation=45, ha='right', fontsize=10)
axes[0].set_ylabel('Sinaps Sayısı', fontsize=11)
axes[0].set_title('Mushroom Body: Ödül vs Ceza Sinyali', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(axis='y', alpha=0.3)

# Sağ: PAM alt tipleri
pam_types = pam['cell_type'].value_counts()
axes[1].barh(range(len(pam_types)), pam_types.values, color='#2ecc71', alpha=0.85)
axes[1].set_yticks(range(len(pam_types)))
axes[1].set_yticklabels(pam_types.index, fontsize=10)
axes[1].set_xlabel('Nöron Sayısı', fontsize=11)
axes[1].set_title(f'PAM Alt Tipleri (n={len(pam)})', fontsize=13, fontweight='bold')
axes[1].invert_yaxis()
for i, v in enumerate(pam_types.values):
    axes[1].text(v + 0.5, i, str(v), va='center', fontsize=10)

plt.tight_layout()
plt.savefig('results/10_reward_vs_punishment.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nKaydedildi: results/10_reward_vs_punishment.png")

# Grafik 2: Kenyon Cell bağlantı şeması
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# PAM kutusu
pam_box = plt.Rectangle((0.3, 7), 3, 2.5, fill=True, facecolor='#2ecc71', alpha=0.2, edgecolor='#2ecc71', linewidth=2)
ax.add_patch(pam_box)
ax.text(1.8, 9, 'PAM (ÖDÜL)', ha='center', fontsize=15, fontweight='bold', color='#27ae60')
ax.text(1.8, 8.3, f'{len(pam)} nöron, 15 alt tip', ha='center', fontsize=10)
ax.text(1.8, 7.6, '→ MB Medial Lobe', ha='center', fontsize=10, style='italic')
ax.text(1.8, 7.2, f'{pam_conn["syn_count"].sum():,} sinaps', ha='center', fontsize=10)

# PPL1 kutusu
ppl1_box = plt.Rectangle((6.7, 7), 3, 2.5, fill=True, facecolor='#e74c3c', alpha=0.2, edgecolor='#e74c3c', linewidth=2)
ax.add_patch(ppl1_box)
ax.text(8.2, 9, 'PPL1 (CEZA)', ha='center', fontsize=15, fontweight='bold', color='#c0392b')
ax.text(8.2, 8.3, f'{len(ppl1)} nöron, 8 alt tip', ha='center', fontsize=10)
ax.text(8.2, 7.6, '→ MB Vertical Lobe', ha='center', fontsize=10, style='italic')
ax.text(8.2, 7.2, f'{ppl1_conn["syn_count"].sum():,} sinaps', ha='center', fontsize=10)

# Kenyon Cell kutusu
kc_box = plt.Rectangle((2, 2.5), 6, 3.5, fill=True, facecolor='#f39c12', alpha=0.15, edgecolor='#f39c12', linewidth=2)
ax.add_patch(kc_box)
ax.text(5, 5.5, 'KENYON CELLS', ha='center', fontsize=16, fontweight='bold', color='#e67e22')
ax.text(5, 4.8, f'{len(kenyon_ids):,} nöron (hafıza hücreleri)', ha='center', fontsize=11)
ax.text(5, 4.1, f'{len(shared_kenyon):,} tanesi (%{len(shared_kenyon)/len(kenyon_ids)*100:.0f}) HER İKİSİNDEN sinyal alıyor', ha='center', fontsize=11, fontweight='bold')
ax.text(5, 3.3, '"PAM daha güçlü → yaklaş"', ha='center', fontsize=10, color='#27ae60')
ax.text(5, 2.8, '"PPL1 daha güçlü → kaç"', ha='center', fontsize=10, color='#c0392b')

# Oklar
ax.annotate('', xy=(3.5, 6), xytext=(1.8, 7),
            arrowprops=dict(arrowstyle='->', color='#27ae60', lw=3))
ax.annotate('', xy=(6.5, 6), xytext=(8.2, 7),
            arrowprops=dict(arrowstyle='->', color='#c0392b', lw=2))

# Karar oku
ax.annotate('', xy=(5, 1.5), xytext=(5, 2.5),
            arrowprops=dict(arrowstyle='->', color='#e67e22', lw=2.5))
ax.text(5, 1, 'KARAR: Yaklaş mı? Kaç mı?', ha='center', fontsize=13, fontweight='bold', color='#e67e22')

ax.set_title('Dopamin Ödül-Ceza Devresi', fontsize=16, fontweight='bold', pad=20)

plt.savefig('results/11_reward_punishment_diagram.png', dpi=150, bbox_inches='tight')
plt.close()
print("Kaydedildi: results/11_reward_punishment_diagram.png")

print("\n=== TAMAMLANDI ===")
