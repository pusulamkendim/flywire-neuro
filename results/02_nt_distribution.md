# Analiz 02 - Beyin Bölgelerinde Nörotransmitter Dağılımı

Script: `02_nt_distribution.py`

## Tüm Beyin Genelinde Nörotransmitter Dağılımı

| Nörotransmitter | Oran | Açıklama |
|-----------------|------|----------|
| Asetilkolin | %48.4 | Neredeyse yarısı "yap!" sinyali. Beyin çoğunlukla uyarıcı. |
| GABA | %23.1 | Dörtte biri "yapma!" sinyali. Fren sistemi. |
| Glutamat | %19.2 | Üçüncü büyük oyuncu. |
| Dopamin | %4.4 | Az ama kritik. Ödül/öğrenme sinyali. |
| Oktopamin | %2.7 | Nadir ama önemli düzenleyici. |
| Serotonin | %2.1 | Nadir ama önemli düzenleyici. |

Toplam sinaps: 54,492,922

## Bölge Bazında Dağılım (En Büyük 20 Bölge)

Tüm büyük bölgelerde Asetilkolin baskın.

| Bölge | Sinaps | Baskın NT | Oran |
|-------|--------|-----------|------|
| ME_R | 8,821,540 | Asetilkolin | 44.3% |
| ME_L | 7,712,403 | Asetilkolin | 42.5% |
| LO_R | 3,877,311 | Asetilkolin | 53.0% |
| LO_L | 3,792,886 | Asetilkolin | 53.8% |
| GNG | 2,713,213 | Asetilkolin | 46.9% |
| AVLP_R | 2,000,857 | Asetilkolin | 54.6% |
| LOP_R | 1,901,109 | Asetilkolin | 39.0% |
| AVLP_L | 1,510,021 | Asetilkolin | 53.9% |
| LOP_L | 1,189,689 | Asetilkolin | 40.0% |
| SMP_R | 898,333 | Asetilkolin | 48.9% |
| FB | 897,766 | Asetilkolin | 36.7% |
| PVLP_L | 895,615 | Asetilkolin | 47.2% |
| SMP_L | 816,580 | Asetilkolin | 48.6% |
| AL_L | 760,542 | Asetilkolin | 43.8% |
| PLP_L | 746,323 | Asetilkolin | 55.8% |
| SPS_R | 731,877 | Asetilkolin | 50.0% |
| AL_R | 727,451 | Asetilkolin | 44.8% |
| SLP_R | 723,028 | Asetilkolin | 52.8% |
| PVLP_R | 704,599 | Asetilkolin | 49.8% |
| PLP_R | 664,786 | Asetilkolin | 54.2% |

## Öne Çıkan Gözlemler

- Tüm büyük bölgelerde Asetilkolin baskın
- **AL (Antennal Lobe)** — Koku merkezi: serotonin ve dopamin diğer bölgelere göre daha yüksek
- **FB (Fan-shaped Body)** — Navigasyon merkezi: glutamat ve dopamin oranı yüksek, Asetilkolin oranı en düşük (%36.7)
- 79 beyin bölgesi tespit edildi
