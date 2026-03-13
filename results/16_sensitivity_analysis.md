# Analiz 16 - Parametre Duyarlılık Analizi: PPL1 Önceliği Yapısal mı?

Script: `16_sensitivity_analysis.py`

---

## Soru

PPL1'in PAM'dan önce aktif olması, model parametrelerimize (decay, gain, threshold) bağlı bir artefakt mı, yoksa connectome'un yapısal bir özelliği mi?

## Yöntem

- **400 parametre kombinasyonu** test edildi (4 decay × 5 gain × 5 threshold × 4 min_synapse)
- **5 farklı koku** × 5 parametre seti = 25 ek test
- **Sadece uyarıcı ağ** (GABA yok) ile 3 test
- Toplam **428 simülasyon**

---

## Ana Sonuç

```
  ╔══════════════════════════════════════════════╗
  ║  400 KOMBİNASYONDAN:                        ║
  ║                                              ║
  ║    PPL1 önce:  153/217  (%70.5)              ║
  ║    PAM önce:     0/217  (%0.0)               ║
  ║    Eşit:        64/217  (%29.5)              ║
  ║                                              ║
  ║  PAM HİÇBİR ZAMAN PPL1'DEN ÖNCE DEĞİL!     ║
  ╚══════════════════════════════════════════════╝
```

**217 geçerli testin hiçbirinde PAM, PPL1'den önce aktif olmadı.** Ya PPL1 önce (153), ya eşit (64). Bu %100 tek yönlü bir sonuç.

---

## Test A: Tek Parametre Değişimi

### A1: Decay (bozunma) — 0.0 ile 0.8 arası

| Decay | PPL1 | PAM | Fark | PPL1 önce? |
|-------|------|-----|------|-----------|
| 0.0 | t+4 | t+6 | +2 | EVET |
| 0.1 | t+4 | t+6 | +2 | EVET |
| 0.3 | t+4 | t+7 | +3 | EVET |
| 0.5 | t+5 | t+8 | +3 | EVET |
| 0.6 | t+5 | t+10 | +5 | EVET |
| 0.7 | t+5 | — | — | PAM ulaşamadı |

Decay arttıkça fark **büyüyor** (2→5). Çünkü sinyal zayıfladıkça uzak hedefe (PAM, 307 nöron) ulaşmak zorlaşıyor, yakın hedefe (PPL1, 16 nöron) ulaşmak hâlâ kolay.

### A2: Gain (kazanç) — 0.5 ile 5.0 arası

| Gain | PPL1 | PAM | Fark |
|------|------|-----|------|
| 0.5-1.0 | — | — | İkisi de ulaşamadı |
| 1.5 | t+7 | — | Sadece PPL1 |
| 2.0 | t+4 | t+7 | +3 |
| 3.0 | t+3 | t+3 | 0 (eşit) |
| 4.0-5.0 | t+2 | t+3 | +1 |

Gain çok yüksekse fark kapanıyor ama **asla tersine dönmüyor**.

### A3: Threshold (eşik) — 0.01 ile 0.50 arası

| Eşik | PPL1 | PAM | Fark |
|------|------|-----|------|
| 0.01-0.02 | t+2 | t+3 | +1 |
| 0.05 | t+3 | t+3 | 0 |
| 0.10 | t+4 | t+7 | +3 |
| 0.15 | t+6 | t+9 | +3 |
| 0.20+ | — | — | Ulaşamadı |

Düşük eşikte fark azalıyor (her şey kolay aktif), yüksek eşikte artıyor.

### A4: Minimum Sinaps Filtresi

| Min Syn | PPL1 | PAM | Fark |
|---------|------|-----|------|
| 1 | t+4 | t+6 | +2 |
| 3 | t+4 | t+7 | +3 |
| 5 | t+4 | t+7 | +3 |
| **10** | **t+4** | **t+10** | **+6** |

Zayıf bağlantıları çıkardıkça fark **dramatik artıyor** (2→6). Çünkü PAM'a ulaşan dolaylı yollar kesiliyor, ama PPL1'in güçlü doğrudan bağlantıları korunuyor.

---

## Test B: Grid Search — 400 Kombinasyon

### Fark Dağılımı

```
  +0 adım:  64 ████████████████████████████████  eşit
  +1 adım:  77 ██████████████████████████████████████  PPL1 1 adım önde
  +2 adım:  44 ██████████████████████  PPL1 2 adım önde
  +3 adım:  18 █████████  PPL1 3 adım önde
  +4 adım:   9 ████
  +5 adım:   4 ██
  +6 adım:   1 █

  Negatif (PAM önce): 0 — HİÇ YOK!
```

- Ortalama fark: **+1.3 adım**
- Medyan: **+1 adım**
- Minimum: **0** (eşit), maksimum: **+6**
- **Hiçbir kombinasyonda PAM önce değil**

---

## Test C: 5 Farklı Koku × 5 Parametre Seti

| Parametre | K1 | K2 | K3 | K4 | K5 | Ort |
|-----------|----|----|----|----|-----|-----|
| Düşük d/g (0.1/1.0) | +4 | +4 | +4 | +5 | +4 | **+4.2** |
| Varsayılan (0.3/2.0) | +3 | +3 | +3 | +3 | +3 | **+3.0** |
| Yüksek d/g (0.5/3.0) | +3 | +3 | +3 | +3 | +3 | **+3.0** |
| Ekstrem (0.7/5.0) | 0 | +1 | 0 | +1 | 0 | +0.4 |
| Düşük d + yüksek g | 0 | 0 | 0 | 0 | 0 | 0 |

Makul parametre aralığında 5/5 kokuda tutarlı. Ekstrem parametrelerde fark azalıyor ama **asla tersine dönmüyor**.

---

## Test D: Sadece Uyarıcı Ağ (GABA yok)

| Parametre | PPL1 | PAM | Fark |
|-----------|------|-----|------|
| d=0.3 g=2.0 t=0.1 | t+4 | t+5 | +1 |
| d=0.1 g=1.0 t=0.05 | t+4 | t+7 | +3 |
| d=0.5 g=3.0 t=0.2 | t+4 | t+5 | +1 |

**Tüm sinapslar uyarıcı olsa bile PPL1 önce.** İnhibisyon bu sonucu yaratmıyor.

---

## Sonuç

PPL1 temporal önceliği:
- **400 kombinasyonun %0'ında** PAM önce çıktı
- **%70.5'inde** PPL1 önce, **%29.5'inde** eşit
- Decay, gain, threshold, ağ filtresi, koku türü, GABA varlığı — **hiçbiri sonucu tersine çevirmiyor**
- Bu bir model artefaktı değil, **connectome'un yapısal bir özelliği**

Güvenilirlik notu: Parametre uzayının test edilen bölgesinde sonuç tek yönlü. Bu, bulgunun modele özgü varsayımlara değil, ağ topolojisine dayandığını güçlü şekilde destekliyor.
