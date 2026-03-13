# Analiz 03 - Dopamin ve Serotonin Odaklı Analiz

Script: `03_dopamine_serotonin.py`

---

## Dopaminerjik Nöronlar — "Ödül Öğretmenleri"

- **5,909 nöron** (%4.2) — az ama çok kritik
- Ortalama güven skoru: 0.643

### Süper Sınıf Dağılımı

| Sınıf | Sayı | Oran |
|-------|------|------|
| central (merkezi beyin) | 5,705 | %96.5 |
| sensory (duyusal) | 129 | %2.2 |
| optic (görme) | 26 | %0.4 |
| diğer | 49 | %0.9 |

**%96.5'i merkezi beyinde** — düşünme/karar merkezinde yoğunlaşmış.

### Hücre Sınıfları

| Sınıf | Sayı | Açıklama |
|-------|------|----------|
| Kenyon_Cell | 5,172 | Mantar cisimciği (mushroom body) — öğrenme/hafıza merkezi |
| DAN | 330 | Dopamine Neurons — klasik dopamin nöronları |
| mechanosensory | 87 | Mekanik duyusal |
| CX | 61 | Central Complex — navigasyon |

**%87'si Kenyon Hücresi** — öğrenme/hafıza merkezinin temel hücreleri.

### En Yaygın Hücre Tipleri

| Tip | Sayı | Açıklama |
|-----|------|----------|
| KCg-m | 2,186 | Kenyon Cell gamma - medial |
| KCab | 1,643 | Kenyon Cell alpha/beta |
| KCapbp-m | 338 | Kenyon Cell alpha'/beta' prime - medial |
| KCapbp-ap2 | 298 | |
| KCg-d | 295 | Kenyon Cell gamma - dorsal |
| PAM08 | 45 | PAM dopamin nöronu — ödül sinyali |
| PAM01 | 41 | PAM dopamin nöronu |
| PAM04 | 32 | PAM dopamin nöronu |

### Dopamin Sinyali Nereye Gidiyor?

Dopamin-baskın bağlantı: 328,940 / 16,847,997 — Toplam 461,520 sinaps

| Hedef Bölge | Sinaps | Oran | Açıklama |
|-------------|--------|------|----------|
| MB_ML_L | 34,669 | %7.5 | Mushroom Body Medial Lobe (sol) — hafıza |
| MB_ML_R | 27,841 | %6.0 | Mushroom Body Medial Lobe (sağ) — hafıza |
| FB | 26,106 | %5.7 | Fan-shaped Body — navigasyon |
| GNG | 26,020 | %5.6 | Gnathal Ganglia — tat/motor kontrol |
| AVLP_R | 24,003 | %5.2 | Anterior Ventrolateral Protocerebrum |
| ME_L | 22,875 | %5.0 | Medulla (sol) — görme |
| LO_L | 19,254 | %4.2 | Lobula (sol) — görme |

**Dopamin en çok hafıza merkezine (Mushroom Body) sinyal gönderiyor.**

---

## Serotonerjik Nöronlar — "Duyusal Düzenleyiciler"

- **2,282 nöron** (%1.6) — dopaminden daha nadir
- Ortalama güven skoru: 0.520

### Süper Sınıf Dağılımı

| Sınıf | Sayı | Oran |
|-------|------|------|
| sensory (duyusal) | 1,553 | %68.1 |
| central (merkezi) | 340 | %14.9 |
| ascending (yükselen) | 116 | %5.1 |
| optic (görme) | 94 | %4.1 |
| diğer | 179 | %7.8 |

**%68'i duyusal nöron** — dopaminden çok farklı! Serotonin dışarıdan gelen sinyalleri düzenliyor.

### Hücre Sınıfları

| Sınıf | Sayı | Açıklama |
|-------|------|----------|
| olfactory | 650 | Koku duyusal nöronları |
| visual | 638 | Görme duyusal nöronları |
| AN | 171 | Ascending Neurons |
| CX | 147 | Central Complex |
| mechanosensory | 141 | Dokunma/titreşim |
| gustatory | 60 | Tat nöronları |

**Koku (%29) ve görme (%28) duyusal nöronlarında yoğun.**

### Serotonin Sinyali Nereye Gidiyor?

Serotonin-baskın bağlantı: 189,743 / 16,847,997 — Toplam 350,304 sinaps

| Hedef Bölge | Sinaps | Oran | Açıklama |
|-------------|--------|------|----------|
| AL_L | 53,378 | %15.2 | Antennal Lobe (sol) — koku merkezi |
| GNG | 52,577 | %15.0 | Gnathal Ganglia — tat/motor |
| AL_R | 44,476 | %12.7 | Antennal Lobe (sağ) — koku merkezi |
| FB | 27,676 | %7.9 | Fan-shaped Body — navigasyon |
| PRW | 14,861 | %4.2 | Prow — motor kontrol |
| SMP_R | 14,504 | %4.1 | Superior Medial Protocerebrum |
| SMP_L | 14,241 | %4.1 | Superior Medial Protocerebrum |

**Serotonin en çok koku merkezine (Antennal Lobe, %28) sinyal gönderiyor.**

---

## Dopamin vs Serotonin — Bölge Karşılaştırması

Her bölgede dopamin/serotonin sinaps oranı (sadece toplam >1000 sinaps olan bölgeler):

| Bölge | Dopamin | Serotonin | DA Oranı | Baskın |
|-------|---------|-----------|----------|--------|
| MB_ML_L | 34,669 | 795 | %97.8 | DOPAMİN |
| MB_VL_R | 8,118 | 351 | %95.9 | DOPAMİN |
| MB_VL_L | 6,425 | 329 | %95.1 | DOPAMİN |
| MB_ML_R | 27,841 | 2,623 | %91.4 | DOPAMİN |
| LO_L | 19,254 | 1,926 | %90.9 | DOPAMİN |
| LO_R | 13,685 | 1,950 | %87.5 | DOPAMİN |
| CRE_L | 12,940 | 1,885 | %87.3 | DOPAMİN |
| MB_PED_R | 4,041 | 660 | %86.0 | DOPAMİN |
| EB | 2,808 | 594 | %82.5 | DOPAMİN |

**Neredeyse hiçbir bölgede serotonin dopamini geçmiyor** — dopamin daha yaygın bir düzenleyici.

---

## Özet Bulgular

| | Dopamin | Serotonin |
|---|---------|-----------|
| **Nöron sayısı** | 5,909 (%4.2) | 2,282 (%1.6) |
| **Ana konum** | Merkezi beyin (%96.5) | Duyusal nöronlar (%68.1) |
| **Anahtar hücreler** | Kenyon Cells, PAM nöronları | Koku ve görme duyusal nöronları |
| **Birincil hedef** | Mushroom Body (hafıza) | Antennal Lobe (koku) |
| **İşlev** | "Bunu hatırla!" — ödül/öğrenme | "Bunu iyi kokla/tat!" — duyusal düzenleme |
