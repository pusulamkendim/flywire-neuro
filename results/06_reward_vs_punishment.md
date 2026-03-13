# Analiz 06 - Ödül vs Ceza Dopamin Devreleri

Script: `06_reward_vs_punishment.py`

---

## PAM vs PPL1: Sayısal Karşılaştırma

| | PAM (ödül) | PPL1 (ceza) | Oran |
|---|-----------|-------------|------|
| Nöron sayısı | 307 | 16 | 19:1 |
| Alt tip | 15 | 8 | 2:1 |
| Toplam sinaps | ~59K | ~31K | 2:1 |
| Birincil hedef | MB Medial Lobe | MB Vertical Lobe | Farklı loblar |

## Mushroom Body Bölge Dağılımı

| MB Bölgesi | PAM (ödül) | PPL1 (ceza) | Baskın |
|------------|-----------|-------------|--------|
| MB_ML_L | 22,170 | 4,125 | ÖDÜL |
| MB_ML_R | 19,284 | 2,259 | ÖDÜL |
| MB_VL_R | 646 | 5,503 | CEZA |
| MB_VL_L | 617 | 4,163 | CEZA |
| MB_PED_L | 658 | 2,753 | CEZA |
| MB_PED_R | 1,044 | 1,709 | CEZA |

## Ortak Kenyon Cell Hedefleri

- PAM → 5,121 Kenyon hücresi
- PPL1 → 4,752 Kenyon hücresi
- **Ortak: 4,708 (%90.9)** — neredeyse tüm hafıza hücreleri her iki sinyali de alıyor

## Neden PPL1 Daha Az Nöron Ama Etkili?

PPL1 16 nöronla 307 PAM nöronunun etkisine yakın kuvvette:

1. **Her PPL1 nöronu çok daha fazla sinaps kuruyor**: 31K sinaps / 16 nöron = ~1,940 sinaps/nöron vs PAM'ın 59K / 307 = ~192 sinaps/nöron. PPL1 nöron başına **10x daha güçlü**.

2. **Evrimsel mantık — tehlike sinyali hızlı ve kesin olmalı**:
   - Ödülü kaçırmak → yemek bulamazsın, ama ölmezsin
   - Tehlikeyi kaçırmak → ölürsün
   - Bu yüzden az ama çok güçlü ceza nöronları, çok ama zayıf ödül nöronlarından daha hayatta kalma avantajı sağlıyor

3. **Farklı kodlama stratejisi**:
   - PAM (ödül): Çok nöron, ince ayar — "bu koku biraz iyi, şu koku çok iyi, bu orta"
   - PPL1 (ceza): Az nöron, güçlü sinyal — "TEHLİKE! KAÇ!" (nüans gereksiz)

## Üretilen Grafikler

- `10_reward_vs_punishment.png` — MB bölgelerinde PAM vs PPL1 + PAM alt tipleri
- `11_reward_punishment_diagram.png` — Ödül-ceza devresi özet şeması
