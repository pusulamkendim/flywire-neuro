# Analiz 07 - Serotoninin Koku İşlemedeki Rolü

Script: `07_serotonin_role.py`

---

## Ana Bulgu: Serotonin Ödül/Ceza Devresine Karışmıyor

| Serotonin nereye sinyal gönderiyor? | Sinaps | Oran |
|-------------------------------------|--------|------|
| **Antennal Lobe (koku işleme)** | **250,927** | **%99.8** |
| PAM + PPL1 + Kenyon (ödül/ceza) | 521 | %0.2 |
| **Fark** | **481x** | |

Serotonin ve dopamin devreleri arasında neredeyse sıfır etkileşim var:
- SER → PAM: 156 sinaps (gürültü seviyesi)
- SER → PPL1: 82 sinaps
- SER → Kenyon: 283 sinaps
- PAM → SER: 50 sinaps
- PPL1 → SER: 31 sinaps

---

## Serotonin AL'de Ne Yapıyor?

### Koku İşleme Devresi

```
KOKU GELDİ
    │
    ▼
ORN (Koku Reseptörleri) ───→ ALPN (Projeksiyon Nöronları) ──→ Mushroom Body
  2,282 nöron                  685 nöron                       (dopamin devresine)
  %28'i serotonerjik
    │          ▲                    ▲
    │          │                    │
    │    ┌─────┘                    │
    ▼    │                          │
ALLN (Yerel Nöronlar) ─────────────┘
  429 nöron
  %9'u serotonerjik
```

### Serotonin AL'de Kime Sinyal Gönderiyor?

| Hedef | Sinaps | Oran | İşlev |
|-------|--------|------|-------|
| ALLN (yerel nöronlar) | 119,324 | %47.8 | Kendi ağını senkronize et |
| ALPN (projeksiyon) | 93,133 | %37.3 | Çıkış sinyalini ayarla |
| ORN (reseptörler) | 31,402 | %12.6 | Girişe geri bildirim |

### Serotonerjik Yerel Nöron (ALLN) Hedefleri

| Hedef | Sinaps | Ne Yapıyor? |
|-------|--------|-------------|
| → ALLN (kendi arası) | 91,929 | Yerel ağı senkronize et |
| → ALPN (çıkışa) | 66,529 | Çıkış kalitesini ayarla |
| → ORN (girişe geri bildirim) | 27,169 | Reseptör hassasiyetini kalibre et |

### Serotonerjik Koku Reseptörü (ORN) Hedefleri

| Hedef | Sinaps |
|-------|--------|
| → ALPN | 26,155 |
| → ALLN | 24,366 |

---

## AL'deki Serotonerjik Nöronlar

| Sınıf | Sayı | Oran | Açıklama |
|-------|------|------|----------|
| ORN (koku reseptörü) | 650 | %28.5 | Gelen koku bilgisini serotonin ile modüle ediyor |
| ALLN (yerel nöron) | 40 | %9.3 | Yerel düzenleme ağı |

---

## Serotoninin 3 Görevi

### 1. Reseptör Kalibrasyonu (→ ORN: 27K sinaps)
"Koku çok zayıf → hassasiyeti aç" veya "Koku çok güçlü → hassasiyeti kıs"
Sineğin kokunun gücüne göre "yaklaş" veya "bu kadar yakın olma" tepkisi vermesini sağlıyor.

### 2. Çıkış Kalite Kontrolü (→ ALPN: 66K sinaps)
Projeksiyon nöronlarına gönderilen koku bilgisinin kalitesini düzenliyor.
Önemli kokuyu güçlendir, arka plan gürültüsünü bastır → Mushroom Body'ye temiz sinyal gitsin.

### 3. Ağ Senkronizasyonu (→ ALLN: 92K sinaps)
Yerel nöronlar arası koordinasyon — birlikte çalışarak tutarlı bir "koku resmi" oluştur.

---

## Büyük Resim: İş Bölümü

```
SEROTONİN                              DOPAMİN
"Kokuyu düzgün algıla"               "Bu kokuyu değerlendir"
     │                                     │
     │  (481x daha güçlü burada)           │
     ▼                                     ▼
Antennal Lobe                       Mushroom Body
  - Reseptör kalibrasyonu              - PAM: "Bu iyiydi!"
  - Çıkış kalite kontrolü             - PPL1: "Bundan kaç!"
  - Ağ senkronizasyonu                - Kenyon: Karar ver
     │                                     │
     └──── temiz koku bilgisi ─────────────┘
               (ALPN üzerinden)
```

Serotonin diyor ki: "Ben sana temiz, doğru bir koku sinyali veriyorum. Ödül mü ceza mı, o senin işin dopamin."

İki sistem arasında net iş bölümü:
- **Serotonin** = Algılama kalitesi (volume düğmesi)
- **Dopamin** = Değer atama (iyi/kötü etiketi)
- Birbirlerine neredeyse hiç dokunmuyorlar (481x fark)
