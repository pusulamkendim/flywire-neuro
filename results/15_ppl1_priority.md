# Analiz 15 - PPL1 Temporal Öncelik: "Ceza Neden Ödülden Önce?"

Script: `15_ppl1_priority.py`

---

## Hipotez

PPL1 (ceza) sistemi, PAM (ödül) sisteminden yapısal olarak daha hızlı aktive oluyor. Bu evrimsel bir avantaj: tehlikeyi önce algıla, faydayı sonra değerlendir.

---

## Test 1: Farklı Koku Yoğunlukları — Her Zaman Önce

| ORN % | PPL1 ilk | PAM ilk | Fark | PPL1 son | PAM son |
|-------|---------|---------|------|----------|---------|
| %5 | t+7 | t+10 | **+3** | 16/16 | 23/307 |
| %10 | t+6 | t+8 | **+2** | 16/16 | 75/307 |
| %20 | t+5 | t+7 | **+2** | 16/16 | 87/307 |
| %30 | t+4 | t+7 | **+3** | 16/16 | 94/307 |
| %50 | t+3 | t+6 | **+3** | 16/16 | 97/307 |
| %100 | t+3 | t+6 | **+3** | 16/16 | 100/307 |

**Bulgu:** Koku ne kadar güçlü olursa olsun, PPL1 her zaman 2-3 adım önce. Ve PPL1 her zaman 16/16 (%100) aktif olurken, PAM en güçlü kokuda bile sadece 100/307 (%33).

Ek bulgu: PPL1 **doygunluğa** çok hızlı ulaşıyor (16 nöronun tamamı). PAM ise asla tam aktif olmuyor. Bu "her zaman tetikte" vs "kademeli değerlendirme" farkını gösteriyor.

---

## Test 2: 10 Farklı Koku — %100 Tutarlı

| | PPL1 ilk | PAM ilk | Fark |
|---|---------|---------|------|
| Koku 1-8, 10 | t+4 | t+7 | **+3** |
| Koku 9 | t+3 | t+7 | **+4** |

- Ortalama PPL1: **t+3.9**
- Ortalama PAM: **t+7.0**
- Ortalama fark: **+3.1 adım**
- **10/10 kokuda PPL1 önce** — %100 tutarlılık

Bu rastgele değil, yapısal bir özellik.

---

## Test 3: Yapısal Yol Analizi (BFS)

| Yol | En Kısa | Ortalama | Ulaşılan |
|-----|---------|----------|----------|
| ORN → PPL1 | 2 sinaps | 2.4 sinaps | **16/16 (%100)** |
| ORN → PAM | 2 sinaps | 2.6 sinaps | 50/307 (%16) |

**Sürpriz:** Minimum yol uzunluğu aynı (2 sinaps)! Ama kritik fark:
- PPL1: 16 nöronun **tamamına** 2-3 sinapsta ulaşılıyor
- PAM: 307 nöronun sadece **50'sine** (%16) aynı derinlikte ulaşılıyor

PPL1 daha hızlı değil çünkü daha yakın — daha hızlı çünkü **daha az nöron var ve hepsi ulaşılabilir**. 16 nöronun hepsini aktive etmek, 307 nöronun çoğunluğunu aktive etmekten çok daha kolay.

---

## Test 4: Farklı Duyusal Girişler

| Duyu | Başlangıç | PPL1 ilk | PAM ilk | Fark |
|------|-----------|---------|---------|------|
| **Koku** | 684 ORN | t+5 | t+7 | **+2** |
| **Tat** | 408 gust. | t+3 | t+3 | **0** |
| **Mekanosensör** | 801 | t+6 | — | **PPL1 tek** |
| **Görme** | 422 foto. | t+12 | — | **PPL1 tek** |

**Kritik bulgular:**
- **Koku:** PPL1 2 adım önce — klasik bulgu
- **Tat:** PPL1 ve PAM **aynı anda** (t+3) — tatta paralel değerlendirme (önceki simülasyonla tutarlı)
- **Mekanosensör:** PAM hiç aktif olmuyor! Sadece PPL1 → dokunma = tehlike odaklı
- **Görme:** PAM hiç aktif olmuyor, PPL1 bile geç (t+12) → görme sistemi ödül/ceza devresine çok uzak

---

## Test 5: Ara Nöron Analizi — Kim Besliyor?

### PPL1'e giriş (76,864 sinaps)

| Kaynak | Sinaps | Oran |
|--------|--------|------|
| **Kenyon Cell** | **39,222** | **%85.4** |
| MBON | 3,195 | %7.0 |
| MBIN | 1,153 | %2.5 |
| CX | 903 | %2.0 |
| ALPN | 648 | %1.4 |

### PAM'a giriş (127,021 sinaps)

| Kaynak | Sinaps | Oran |
|--------|--------|------|
| **Kenyon Cell** | **85,527** | **%88.1** |
| MBON | 5,480 | %5.6 |
| MBIN | 2,060 | %2.1 |
| DAN | 1,675 | %1.7 |
| CX | 1,134 | %1.2 |

**İkisinin de ana girdisi Kenyon Cell** (~%85-88). Ama kritik fark:
- PPL1 nöron başına **4,804 girdi sinaps** (76,864 / 16)
- PAM nöron başına **414 girdi sinaps** (127,021 / 307)

**PPL1 nöron başına 11.6x daha fazla girdi alıyor!** Bu nedenle aktivasyon eşiğine daha hızlı ulaşıyor.

### Duyusal nöronlardan doğrudan giriş: SIFIR

Hiçbir duyusal nöron (koku, tat, görme, mekanosensör) PPL1 veya PAM'a doğrudan bağlı değil. Sinyal her zaman en az 2 ara nöron üzerinden geçiyor.

### NT profili farkı

| NT | PPL1'e | PAM'a |
|----|--------|-------|
| Dopamin | %52.8 | %69.8 |
| ACh | %28.9 | %16.1 |
| Glutamat | %12.6 | %10.0 |
| GABA | %5.5 | %4.0 |

PPL1 daha fazla ACh (%29 vs %16) alıyor — yani daha fazla "doğrudan uyarıcı" girdi. PAM ise daha fazla dopamin alıyor (%70 vs %53) — yani PAM'ın aktivasyonu daha çok **diğer dopamin nöronlarına** bağımlı (dolaylı).

---

## Test 6: GABA Olmadan Ne Olur?

| Adım | Normal PPL1 | Normal PAM | GABA yok PPL1 | GABA yok PAM |
|------|-------------|------------|---------------|--------------|
| t+4 | 2/16 | 0/307 | 6/16 | 0/307 |
| t+5 | 6/16 | 0/307 | 9/16 | 0/307 |
| t+6 | 10/16 | 0/307 | 14/16 | 2/307 |
| t+7 | 13/16 | 3/307 | **16/16** | 21/307 |
| t+8 | 16/16 | 20/307 | 16/16 | 52/307 |

- **Normal:** PPL1 t+4, PAM t+7, fark **3 adım**
- **GABA yok:** PPL1 t+4, PAM t+6, fark **2 adım**

**GABA olmadan da PPL1 hâlâ önce!** Fark 3'ten 2'ye düştü — yani GABA PAM'ı biraz geciktiriyor (1 adım) ama asıl neden GABA değil. PPL1'in önceliği **yapısal**, inhibisyon kaynaklı değil.

Ek bulgu: GABA olmadan PAM çok daha hızlı büyüyor (t+8: 52 vs 20) — GABA normalde PAM'ı kısmen bastırıyor.

---

## Test 7: PPL1 vs PAM Nöron Başına Girdi

### PPL1 (8 alt tip, 16 nöron)

| Alt tip | Nöron | Girdi/nöron |
|---------|-------|-------------|
| PPL101 | 2 | **10,382** |
| PPL103 | 2 | 6,613 |
| PPL107 | 2 | 6,124 |
| PPL106 | 2 | 4,639 |
| PPL105 | 2 | 4,186 |
| PPL104 | 2 | 2,498 |
| PPL102 | 2 | 2,460 |
| PPL108 | 2 | 1,529 |

### PAM (ilk 10 alt tip, 307 nöron)

| Alt tip | Nöron | Girdi/nöron |
|---------|-------|-------------|
| PAM11 | 16 | 551 |
| PAM06 | 30 | 539 |
| PAM04 | 32 | 519 |
| PAM05 | 21 | 496 |
| PAM08 | 45 | 367 |
| PAM07 | 18 | 314 |

**PPL101 nöron başına 10,382 girdi sinaps — PAM'ın en güçlüsünün 19x katı!**

---

## Neden PPL1 Önce? — 5 Yapısal Neden

```
                    PPL1 (CEZA)                     PAM (ÖDÜL)
                    ───────────                     ──────────
  Nöron sayısı:     16                              307
  Girdi/nöron:      4,804                           414  (11.6x az)
  ACh girdisi:      %29 (doğrudan uyarı)            %16
  DA bağımlılığı:   %53 (kısmen bağımsız)           %70 (başka DA'ya bağımlı)
  Doygunluk:        Her zaman 16/16 (%100)          Asla %100 değil
```

### 1. Sayı Avantajı
16 nöronun tamamını aktive etmek, 307'nin çoğunluğunu aktive etmekten çok daha kolay ve hızlı.

### 2. Girdi Yoğunluğu
Her PPL1 nöronu 11.6x daha fazla girdi sinaps alıyor → aktivasyon eşiğine daha hızlı ulaşıyor.

### 3. ACh Doğrudan Uyarı
PPL1 daha fazla ACh (%29 vs %16) alıyor. ACh en hızlı uyarıcı — doğrudan aktive eder. PAM ise daha çok dopamin alıyor (%70) — başka dopamin nöronlarının önce aktif olmasını bekliyor.

### 4. GABA Bağımsız
GABA devre dışı bırakıldığında bile PPL1 önce. Öncelik yapısal, inhibisyon kaynaklı değil.

### 5. Doygunluk Hızı
PPL1 her senaryoda %100'e ulaşıyor. PAM asla tam aktif olmuyor. PPL1 "hep veya hiç" çalışıyor — ya tehlike var ya yok. PAM kademeli çalışıyor — "ne kadar faydalı?"

---

## Evrimsel Yorumlar

### Duyusal Modalite Farkları

```
  KOKU:    PPL1 +3 önce → "uzaktan gelen tehlikeyi değerlendir"
  TAT:     PPL1 = PAM    → "ağızdaki şey hakkında paralel karar ver"
  DOKUNMA: Sadece PPL1    → "dokunma = potansiyel tehlike"
  GÖRME:   Çok geç        → "görsel bilgi ödül/cezaya uzak"
```

Bu farklar evrimsel mantıkla uyumlu:
- **Koku** uzaktan gelir → tehlikeyi önce değerlendir, karar vermeye zaman var
- **Tat** doğrudan temas → paralel karar ver (ye/tükür), zaman yok
- **Dokunma** beklenmeyen temas → neredeyse sadece tehlike sinyali
- **Görme** dolaylı → ödül/ceza devresiyle zayıf bağlantı

### "Hayatta Kalma Öncelikli" Mimari

```
  Yanlış Pozitif (gereksiz kaçış):  Maliyeti düşük (enerji kaybı)
  Yanlış Negatif (tehlikeyi kaçırma): Maliyeti ölümcül

  → Doğal seçilim PPL1'i optimize etmiş:
    - Az nöron (16) ama çok güçlü (11.6x girdi)
    - Her zaman %100 doygunluk
    - Yapısal öncelik (GABA'dan bağımsız)
    - Tüm duyulardan önce aktif
```
