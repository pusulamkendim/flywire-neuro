# Analiz 12 - Sinyal Yayılma Simülasyonu: "Koku Geldi, Beyin Ne Yapıyor?"

Script: `12_signal_propagation.py`

---

## Model

Basit "spreading activation" (yayılan aktivasyon) modeli:
- 684 koku reseptörü (ORN) aktive edilir (%30'u — bir koku alt kümesi)
- Her adımda aktif nöronlar bağlantıları üzerinden sinyal gönderir
- Sinyal gücü = kaynak aktivasyon × sinaps ağırlığı (normalize)
- **ACh/Glut/DA/SER/OCT → uyarıcı (+)**, **GABA → inhibitör (-)**
- Eşik: aktivasyon > 0.1 → aktif
- 5.1 milyon güçlü bağlantı (≥3 sinaps) kullanıldı

---

## Adım Adım: Beyinde Neler Oluyor?

```
  Adım   Aktif  |  ORN   ALPN  ALLN    KC  MBON   PAM  PPL1   Mot  Desc
  ─────────────────────────────────────────────────────────────────────────
  ⚡ 0      684  |  684     0     0     0     0     0     0     0     0
    1    1,197  |  690   310   184     0     0     0     0     0     0
    2    1,862  |  690   352   238    22     0     0     0     0     7
    3    2,258  |  690   362   246    27     6     0     0     0    22
    4    2,693  |  678   366   249    30    19     0     2     0    34
    5    3,283  |  653   366   251    32    26     0     6     0    49
    6    3,997  |  535   367   253    32    33     0    10     0    91
    7    4,709  |   75   365   252    32    42     3    13     2   161
    8    6,214  |   17   354   248    32    54    20    16     8   268
```

---

## Zaman Çizelgesi — Film Gibi

```
  ⚡ KOKU GELDİ
  │
  │  684 koku reseptörü (ORN) algıladı
  │  Aktif bölge: AL (Antennal Lobe) — sadece koku merkezi
  │
  ▼ t+1  İLK TEPKİ (milisaniyeler)
  │
  │  → ALLN (184 yerel nöron) devrede — kokuyu filtrele, senkronize et
  │  → ALPN (310 projeksiyon nöronu) devrede — temiz sinyali ileriye gönder
  │  Serotonin zaten aktif (%10) — "volume düğmesi açık, kokuyu kalibre et"
  │
  ▼ t+2  BİLGİ YAYILIYOR
  │
  │  → İlk Kenyon Cell'ler aktif (22) — hafıza sistemi uyanıyor
  │  → LH (Lateral Horn) parlıyor — doğuştan gelen tepkiler
  │  → İlk Descending nöronlar (7) — motor sisteme ilk sızıntı
  │  → GABA devrede (%1.4) — fren sistemi hazırlanıyor
  │
  ▼ t+3  KARAR SÜRECİ BAŞLIYOR
  │
  │  → İlk MBON'lar aktif (6) — "bu koku hakkında ne biliyoruz?"
  │  → Descending artıyor (22) — vücut hazırlanıyor
  │  → LH aktivasyonu güçleniyor — doğuştan "iyi koku / kötü koku" tepkisi
  │
  ▼ t+4  CEZA SİSTEMİ UYANIYOR
  │
  │  → PPL1 aktif (2/16) — "bu tehlikeli mi?"
  │  → MBON artıyor (19) — hafızadan bilgi geliyor
  │  → ORN'ler azalmaya başlıyor (678) — GABA kokuyu bastırıyor
  │
  ▼ t+5-6  SİNYAL DERİNLEŞİYOR
  │
  │  → PPL1 artıyor (10/16) — ceza sistemi tam devrede
  │  → Descending 91'e çıktı — motor komutlar güçleniyor
  │  → SMP, AVLP aktif — çok duyusal entegrasyon bölgeleri
  │  → ORN'ler düşüyor (535) — GABA koku girişini bastırıyor (adaptasyon)
  │
  ▼ t+7  ÖDÜL SİSTEMİ GEÇ GELİYOR
  │
  │  → PAM ilk kez aktif (3/307) — ödül sistemi en son devrede
  │  → Motor nöronlar aktif (2) — vücut hareket etmeye başlıyor
  │  → ORN'ler çöküyor (75) — koku adaptasyonu tamamlanıyor
  │
  ▼ t+8  TAM TEPKİ
  │
  │  → PPL1 %100 aktif (16/16) — tüm ceza nöronları devrede!
  │  → PAM artıyor (20/307) — ödül sistemi yavaş yavaş
  │  → MBON %56 aktif — hafıza çıkışı güçlü
  │  → 268 Descending + 8 Motor — davranış çıkışı
  │  → 6,214 toplam aktif nöron
```

---

## Kritik Bulgular

### 1. PPL1 (ceza) PAM'dan (ödül) çok önce geliyor

| Sistem | İlk aktif | Tam aktif |
|--------|-----------|-----------|
| **PPL1 (ceza)** | **t+4** | **t+8 (%100!)** |
| PAM (ödül) | t+7 | t+8 (%6.5) |

Ceza sistemi 3 adım önce devrede ve %100'e ulaşıyor. Ödül sistemi geç ve zayıf (%6.5).
**Evrimsel mantık: Önce "tehlikeli mi?" sor, sonra "faydalı mı?" sor.** Yanlış pozitif (gereksiz kaçış) ucuz, yanlış negatif (tehlikeyi kaçırma) ölümcül.

### 2. GABA Koku Adaptasyonunu Sağlıyor

ORN'ler: 684 → 690 → 690 → 690 → 678 → 653 → 535 → **75 → 17**

GABA inhibisyonu zamanla koku reseptörlerini bastırıyor. Bu "adaptasyon": aynı kokuya sürekli maruz kalınca duyarlılık düşüyor. Parfüm sıktığında bir süre sonra kokmadığını sanırsın — aynı mekanizma.

### 3. Serotonin En Baştan Devrede

| Adım | Serotonin % | Diğerleri |
|------|-------------|-----------|
| 0 | **%8.7** | ACh %0.6, geri kalan ~%0 |
| 1 | **%10.2** | ACh %0.9 |
| 2 | **%10.7** | GABA %1.4, Glut %0.9 |

Serotonin koku reseptörlerinin %28'inde var — koku geldiği anda otomatik devrede. "Volume düğmesi" baştan açık.

### 4. Uyarıcı/İnhibitör Denge Sabit Kalıyor

| Adım | Uyarıcı | İnhibitör | Oran |
|------|---------|-----------|------|
| 1 | 1,095 | 101 | 10.8:1 |
| 3 | 1,908 | 349 | 5.5:1 |
| 5 | 2,770 | 512 | 5.4:1 |
| 8 | 5,258 | 956 | 5.5:1 |

İlk şoktan sonra (t+1: 10.8:1) denge hızla 5.5:1'e oturuyor ve sabit kalıyor. Beyin **her 5-6 uyarıcı nöron için 1 frenleyici nöron** çalıştırıyor — bu oran evrimsel olarak optimize edilmiş.

### 5. Sinyal Akış Sırası

```
ORN → ALLN + ALPN → LH + Kenyon → MBON → PPL1 → PAM → Motor
 ⚡        t+1          t+2        t+3     t+4    t+7    t+7

     KOKU          FİLTRELE       HATIRLA    CEZA   ÖDÜL  HAREKET
     ALGILAMA      TEMİZLE        KARŞILAŞTIR  ↑      ↑     ET
                                              ÖNCE  SONRA
```

---

## Özet

Bu simülasyon, bir kokunun beyinde nasıl işlendiğini gösteriyor:

1. **Koku gelir** → ORN'ler algılar (t=0)
2. **Filtrelenir** → ALLN + ALPN temizler, serotonin kalibre eder (t+1)
3. **Hafızaya sorulur** → Kenyon Cell + MBON: "bunu biliyoruz mu?" (t+2-3)
4. **Önce tehlike kontrol** → PPL1 hemen devrede (t+4), PAM geç gelir (t+7)
5. **Adaptasyon** → GABA koku girişini zamanla bastırır (t+4-8)
6. **Hareket** → Descending + Motor nöronlar davranışı başlatır (t+7-8)
7. **Denge** → Uyarıcı/inhibitör oranı 5.5:1'de sabitlenir

Tüm süreç gerçek sinekte ~100-200 milisaniyede gerçekleşir.
