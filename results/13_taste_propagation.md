# Analiz 13 - Tat Sinyal Yayılma Simülasyonu: "Sinek Bir Şey Tattı"

Script: `13_taste_propagation.py`

---

## Tat Nöronları Profili

- **408 tat nöronu** (kokuya göre 5.6x az)
- %77 ACh, %15 serotonin, %7 glutamat
- Ana hedef: **GNG (%60)** ve **PRW (%33)** — alt beyin bölgeleri
- Doğrudan hedef sınıflar: AN (%47), gustatory (%33), mAL (%14)

Tat sistemi koku sisteminden çok farklı: **alt beyinde kalıyor**, daha ilkel, daha doğrudan.

---

## Adım Adım: Sinek Bir Şey Tattı

```
  Adım   Aktif  |  Gust   ORN  ALPN  ALLN    KC  MBON   PAM  PPL1   Mot  Desc
  ────────────────────────────────────────────────────────────────────────────────
  ⚡ 0      408  |  408     0     0     0     0     0     0     0     0     0
    1      956  |  408     0    10     2     0     0     0     0     1    27
    2    1,552  |  396     0    10     2     0     2     0     0    28   104
    3    1,874  |  369     0    11     2     0     8     2     5    46   166
    5    2,616  |  248     0    14     3     0    21     3    10    72   287
    8    4,730  |   84     0    19     7     0    39    19    14    95   484
   10    7,105  |   50     0    19    25     0    63    50    16    96   619
```

---

## Zaman Çizelgesi

```
  ⚡ TAT ALGILANDI
  │
  │  408 tat nöronu aktif (GNG + PRW bölgesi — alt beyin)
  │
  ▼ t+1  HEMEN MOTOR TEPKİ!
  │
  │  → 27 Descending + 1 Motor — vücut hemen tepki veriyor!
  │  → Koku ile fark: koku t+1'de sadece ALPN/ALLN aktif, motor t+7'de
  │  → GABA hemen devrede (%0.9) — fren baştan hazır
  │
  ▼ t+2  MOTOR PATLAMA
  │
  │  → 104 Descending + 28 Motor (%25!) — motor nöronların çeyreği aktif
  │  → İlk MBON'lar (2) — hafıza sistemi uyanıyor
  │  → Kenyon Cell hâlâ sıfır — tat hafıza merkezini atla, doğrudan tepki ver
  │
  ▼ t+3  CEZA + ÖDÜL AYNI ANDA
  │
  │  → PPL1 aktif (5/16 = %31) — "bu zehirli mi?"
  │  → PAM aktif (2/307) — "bu besleyici mi?"
  │  → Koku ile fark: kokuda PPL1 t+4, PAM t+7 — tatta ikisi aynı anda!
  │  → 46 Motor (%42) — neredeyse yarısı aktif
  │
  ▼ t+5  GÜÇLÜ TEPKİ
  │
  │  → PPL1 %63, Motor %66, MBON %22
  │  → Tat adaptasyonu: gustatory 408 → 248 (GABA bastırıyor)
  │
  ▼ t+8  TAM TEPKİ
  │
  │  → Motor %86, PPL1 %88, Descending %37
  │  → Oktopamin %12.5 — adrenalin devrede (koku simülasyonunda %5.6)
  │
  ▼ t+10  SONUÇ
  │
  │  → PPL1 %100, Motor %87, MBON %66
  │  → PAM %16 (ödül sistemi tatta daha aktif)
  │  → Kenyon Cell hâlâ SIFIR — tat hafıza merkezine ulaşmıyor!
  │  → Koku sistemi (ORN) hiç aktif olmadı
```

---

## Kritik Bulgular

### 1. Tat Çok Daha Hızlı Motor Tepki Üretiyor

| | Koku | Tat |
|---|---|---|
| Motor ilk aktif | **t+7** | **t+1** |
| Descending ilk aktif | t+2 (7) | t+1 (**27**) |
| Motor %25 | t+8 | **t+2** |
| Motor %87 | — | **t+10** |

**Tat, motor sisteme 6 adım önce ulaşıyor!** Mantıklı: zehirli bir şey yediğinde düşünmeden tükürmek hayat kurtarır. Koku uzaktan gelir, düşünmeye zaman var. Tat ağızdadır, acil tepki gerekir.

### 2. Kenyon Cell'e SIFIR Sinyal — Tat Hafıza Oluşturmuyor!

| | Koku | Tat |
|---|---|---|
| Kenyon Cell (t+2) | 22 | **0** |
| Kenyon Cell (t+10) | 32 | **0** |

Tat sinyali Mushroom Body'ye (hafıza merkezi) hiç ulaşmıyor! Bu şu anlama geliyor:
- **Koku**: "Bu kokuyu hatırla, bir dahaki sefere ne yapacağını bil" → öğrenme
- **Tat**: "Düşünme, hemen tepki ver" → refleks

Sinek koku ile öğreniyor (Pavlov), ama tat ile refleks yapıyor.

### 3. PPL1 ve PAM Aynı Anda Geliyor

| | Koku | Tat |
|---|---|---|
| PPL1 (ceza) ilk | **t+4** | **t+3** |
| PAM (ödül) ilk | **t+7** | **t+3** |
| Fark | 3 adım | **Aynı anda!** |

Kokuda "önce tehlike mi kontrol et" stratejisi var. Tatta ise ceza ve ödül aynı anda değerlendiriliyor. Çünkü tat doğrudan "ye veya tükür" kararı gerektirir — ikisini paralel çalıştırmak daha verimli.

### 4. Oktopamin Tatta Çok Daha Aktif

| Adım | Koku | Tat |
|------|------|-----|
| t+5 | %1.9 | **%3.7** |
| t+8 | %5.6 | **%12.5** |
| t+10 | — | **%17.1** |

Oktopamin (adrenalin) tatta 3x daha aktif. Tat daha güçlü bir "savaş-kaç" tepkisi tetikliyor.

### 5. Uyarıcı/İnhibitör Dengesi Farklı

| | Koku | Tat |
|---|---|---|
| Denge oranı | **5.5:1** | **3.5-5.1:1** |

Tat işleme daha fazla GABA (fren) kullanıyor. Özellikle başlangıçta (3.5:1 vs 5.5:1) — motor tepkiyi kontrol altında tutmak için daha güçlü fren gerekiyor.

---

## Koku vs Tat Karşılaştırması — Büyük Resim

```
                KOKU                                    TAT
          (uzaktan algılama)                    (doğrudan temas)

    ORN (2,282) ─→ AL                    Gustatory (408) ─→ GNG
         │                                        │
    ┌────┤                                   ┌────┤
    ▼    ▼                                   ▼    ▼
  ALLN  ALPN                              Motor  Descending
  filtrele  temizle                       HEMEN   HEMEN
    │    │                                TEPKİ   TEPKİ
    └────┤                                   │
         ▼                                   │
    Kenyon Cell ← HAFIZA                     ├──→ MBON (t+2)
         │                                   │
    MBON (t+3)                               ├──→ PPL1 + PAM (t+3)
         │                                   │
    PPL1 (t+4) → TEHLİKE                    │  (paralel değerlendirme)
         │                                   │
    PAM (t+7) → ÖDÜL                         └──→ Kenyon Cell: SIFIR!
         │                                        (hafıza yok)
    Motor (t+7) → HAREKET
```

| Özellik | Koku | Tat |
|---------|------|-----|
| **Reseptör** | 2,282 | 408 |
| **İşleme merkezi** | AL (üst beyin) | GNG (alt beyin) |
| **Motor tepki** | t+7 (yavaş) | t+1 (anında) |
| **Hafıza oluşumu** | Var (Kenyon Cell) | Yok! |
| **Ceza/ödül sırası** | Önce ceza, sonra ödül | Aynı anda |
| **Oktopamin** | Düşük | 3x daha yüksek |
| **Strateji** | "Düşün, hatırla, karar ver" | "Hemen tepki ver" |
| **Analoji** | Şüpheli paketi koklama | Ağza bir şey koyma |

---

## Özet

**Koku = Analitik Sistem**: Uzaktan algıla → filtrele → hafızayla karşılaştır → önce tehlike mi bak → sonra faydalı mı → en son hareket et. Düşünceli, hafızaya dayalı.

**Tat = Refleks Sistem**: Ağızda bir şey var → HEMEN motor tepki → paralelde ceza/ödül → hafıza oluşturmaya bile gerek yok. İlkel, hızlı, hayat kurtarıcı.

İnsan beyninde de benzer: acı bir şeyi yediğinde düşünmeden tükürürsün (refleks), ama uzaktan kötü bir koku alırsan düşünür, analiz eder, sonra karar verirsin.
