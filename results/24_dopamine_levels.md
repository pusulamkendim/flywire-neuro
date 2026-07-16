# Analysis 24 — PAM Dopamine Levels (150 brain steps = 22.5ms)

## Event Timeline with PAM Activity

| Zaman (ms) | Olay | PAM Level | Hunger | P9 (Hz) |
|------------|------|-----------|--------|---------|
| 0 | **Simulasyon baslar** | 0.0000 | 0.80 | 96 |
| 150 | tracking | 0.0025 | 0.80 | 96 |
| 300 | tracking | 0.0023 | 0.81 | 97 |
| 600 | tracking | 0.0026 | 0.81 | 97 |
| 900 | tracking (grooming oncesi) | 0.0023 | 0.82 | 98 |
| **1005** | **GROOMING BASLAR** | — | 0.82 | 99 |
| 1050 | grooming | 0.0025 | 0.82 | 99 |
| 1350 | grooming | 0.0025 | 0.83 | 99 |
| 1650 | grooming | 0.0027 | 0.83 | 100 |
| **1800** | **GROOMING BITER** | 0.0026 | 0.84 | 100 |
| 1950 | tracking (grooming sonrasi) | 0.0026 | 0.84 | 101 |
| 2550 | tracking | 0.0026 | 0.85 | 102 |
| 3150 | tracking (tasting oncesi) | 0.0029 | 0.86 | 104 |
| 3300 | tasting | 0.0030 | 0.87 | 104 |
| 3450 | tasting | 0.0031 | 0.87 | 104 |
| **3510** | **FEEDING BASLAR** | — | 0.87 | 104 |
| 3600 | feeding | 0.0030 | 0.86 | 103 |
| 3900 | feeding | 0.0029 | 0.82 | 98 |
| 4350 | feeding | 0.0032 | 0.76 | 91 |
| 4800 | feeding | 0.0030 | 0.70 | 84 |
| 5250 | feeding | 0.0029 | 0.64 | 77 |
| 5700 | feeding | 0.0030 | 0.58 | 70 |
| 6150 | feeding | 0.0030 | 0.52 | 63 |
| 6600 | feeding | 0.0028 | 0.46 | 55 |
| 7050 | feeding | 0.0029 | 0.40 | 48 |
| 7500 | feeding | 0.0032 | 0.34 | 41 |
| 7950 | feeding | 0.0029 | 0.28 | 34 |
| 8400 | feeding | 0.0032 | 0.22 | 27 |
| 8850 | feeding | 0.0028 | 0.16 | 20 |
| 9300 | feeding (doyum oncesi) | 0.0029 | 0.10 | 13 |
| **9345** | **DOYUM — Sinek durur** | — | 0.099 | — |

## Olay Bazli Ozet

### Grooming Oncesi vs Sonrasi
| | PAM | Degisim |
|--|-----|---------|
| Oncesi (900ms) | 0.0023 | — |
| Sirasinda (ort.) | 0.0026 | +13% |
| Sonrasi (1950ms) | 0.0026 | +13% |

### Feeding Oncesi vs Sirasinda
| | PAM | Degisim |
|--|-----|---------|
| Oncesi (3150ms) | 0.0029 | — |
| Feeding baslangic (3600ms) | 0.0030 | +3% |
| Feeding orta (5700ms) | 0.0030 | +3% |
| Feeding son (9300ms) | 0.0029 | 0% |

## Yorum

PAM dopamin seviyesi tum simulasyon boyunca **0.0023–0.0032 arasinda** — neredeyse sabit.
Feeding sirasinda belirgin bir artis **yok**.

**Neden:**
- Sugar GRN → PAM yolu connectome'da var (BFS: 2 sinaps, 307/307 PAM ulasılabilir)
- Ancak 150 brain step = 22.5ms beyin zamani, sinaptik agirliklar dusuk
- Ara noronlar yeterince sarj olamiyor, PAM'lara sinyal ulasmiyor
- Olculen ~0.003 deger spontan (baseline) aktivite, sugar-driven degil

**Cozum:** Brain step'i 500'e cikarmak (75ms beyin zamani) sinyalin 2 sinapsi gecmesine yeterli sure tanir.
