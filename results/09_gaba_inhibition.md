# Analiz 09 - GABA: Beyindeki Fren Sistemi

Script: `09_gaba_inhibition.py`

---

## GABA Nöronları — Profil

- **19,170 nöron** (%13.8) — her 7 nörondan biri GABA'cı
- Güven skoru: 0.663

### Süper Sınıf Dağılımı

| Sınıf | Sayı | Oran |
|-------|------|------|
| optic (görme) | 12,654 | %66.0 |
| central (merkezi) | 4,669 | %24.4 |
| sensory (duyusal) | 938 | %4.9 |
| ascending | 312 | %1.6 |
| visual_centrifugal | 297 | %1.5 |
| descending | 148 | %0.8 |

**%66'sı optik (görme) sistemi** — GABA'nın en büyük işi görme sinyallerini filtrelemek.

### Bilgi Akışı (Flow)

| Flow | Oran | Açıklama |
|------|------|----------|
| intrinsic (iç) | %92.6 | İç işleme — GABA ağırlıklı olarak "içeride" çalışıyor |
| afferent (giriş) | %6.6 | Dış girdi |
| efferent (çıkış) | %0.8 | Motor çıkış |

**%92.6 intrinsic** — GABA dışarıya sinyal göndermekten çok, içerideki sinyalleri düzenliyor (fren).

### Öne Çıkan Hücre Tipleri

| Tip | Sayı | Açıklama |
|-----|------|----------|
| Mi4 | 1,530 | Medulla Intrinsic — hareket algılama |
| C3 | 1,507 | Medulla Columnar — yön seçiciliği |
| C2 | 1,124 | Medulla Columnar — kenar/hareket |
| R1-6 | 530 | Fotoreseptörler |
| L1 | 519 | Lamina — ilk görme işleme katmanı |
| Dm10 | 494 | Medulla — geniş alan inhibisyonu |
| Mi13 | 445 | Medulla Intrinsic |

İlk 7 hücre tipinin hepsi görme ile ilgili!

---

## GABA Sinyali Nereye Gidiyor?

GABA-baskın bağlantı: 3,197,309 / Toplam sinaps: 11,249,262

| Hedef Bölge | Sinaps | Oran | Açıklama |
|-------------|--------|------|----------|
| ME_R (Medulla sağ) | 2,223,669 | %19.8 | Görme işleme |
| ME_L (Medulla sol) | 1,945,740 | %17.3 | Görme işleme |
| LO_R (Lobula sağ) | 761,293 | %6.8 | Görme — hareket |
| LO_L (Lobula sol) | 713,458 | %6.3 | Görme — hareket |
| GNG | 625,973 | %5.6 | Alt beyin — motor koordinasyon |
| AVLP_R | 472,398 | %4.2 | Çok duyusal entegrasyon |
| EB | 197,333 | %1.8 | Merkezi kompleks — navigasyon |

**Sinapsların %50'si görme bölgelerinde (ME + LO).** GABA da oktopamin gibi görme odaklı, ama görevi tam tersi: oktopamin "aç, keskinleştir", GABA "filtrele, bastır".

---

## GABA Kimi Susturuyor?

### Hedef NT Tipi (Nöron Başına Normalize)

| Hedef | Sinaps | Hedef Nöron Sayısı | Sinaps/Nöron |
|-------|--------|-------------------|-------------|
| **GABA → GABA** | 2,478,568 | 19,170 | **129.3** |
| **GABA → Oktopamin** | 24,164 | 216 | **111.9** |
| GABA → ACh | 7,786,960 | 86,188 | 90.3 |
| GABA → Glutamat | 2,021,874 | 24,875 | 81.3 |
| GABA → Serotonin | 81,356 | 2,282 | 35.7 |
| GABA → Dopamin | 206,846 | 5,909 | 35.0 |

**Sürpriz: GABA en çok kendini susturuyor!** (129.3 sinaps/nöron) — Bu "disinhibisyon" mekanizması: frenin frenini bırakarak dolaylı olarak hızlandırma.

**İkinci sürpriz: Oktopamine karşı yoğun inhibisyon** (111.9/nöron) — Ama önceki analizde OCT→GABA sıfır çıkmıştı! Yani GABA oktopamini susturabilir ama oktopamin GABA'ya dokunmaz. Tek yönlü kontrol.

**DA ve SER'e düşük inhibisyon** (35/nöron) — GABA bu iki sisteme görece az müdahale ediyor.

---

## Kritik Devrelerdeki GABA Etkisi

| Hedef | Sinaps | Yorum |
|-------|--------|-------|
| **GABA → Kenyon Cell** | **125,232** | Hafıza hücrelerine güçlü fren |
| GABA → ALPN (koku proj.) | 139,935 | Koku çıkışını filtrele |
| GABA → ORN (koku reseptör) | 97,890 | Koku girişini filtrele |
| GABA → ALLN (koku yerel) | 96,370 | Yerel ağı düzenle |
| GABA → MBON (çıkış) | 28,539 | Karar çıkışını kontrol et |
| GABA → PAM (ödül) | 5,094 | Ödül sinyalini frenle |
| GABA → PPL1 (ceza) | 4,257 | Ceza sinyalini frenle |

**Kenyon Cell'lere 125K sinaps** — Hafıza hücreleri üzerinde güçlü inhibisyon. Aldıkları toplam sinapsın **%13.3'ü GABA'dan**. Bu "gereksiz hafıza oluşumunu engelle" mekanizması — her şeyi hatırlamak yerine sadece önemli olanları hatırla.

**MBON'ların aldığının %7.5'i GABA** — Karar çıkışını da kontrol ediyor.

---

## GABA vs ACh Dengesi (Fren vs Gaz)

GABA (inhibitör/fren) ve ACh (uyarıcı/gaz) dengesinin bölge bazında karşılaştırması:

| Bölge | GABA | ACh | GABA % | Durum |
|-------|------|-----|--------|-------|
| **EB (Elipsoid Body)** | 197,333 | 56,934 | **%77.6** | GABA krallığı |
| AMMC_L (mekanosensör) | 44,663 | 33,552 | %57.1 | GABA baskın |
| SAD | 181,994 | 221,807 | %45.1 | Dengeli |
| ME_L (Medulla) | 1,945,740 | 3,578,343 | %35.2 | ACh baskın |
| GNG | 625,973 | 1,277,161 | %32.9 | ACh baskın |

**EB (Elipsoid Body) %77.6 GABA** — Navigasyon merkezi neredeyse tamamen inhibisyon ile çalışıyor! Sinek yön değiştirirken, "gitme" sinyalleri bastırılarak "git" yönü belirleniyor. Tüm yönleri inhibe et, sadece doğru olanı serbest bırak.

---

## GABA Kendini İnhibe Ediyor mu?

| Metrik | Değer |
|--------|-------|
| GABA → GABA | 2,478,568 sinaps |
| GABA → Tüm | 12,600,959 sinaps |
| **Kendi kendini inhibisyon** | **%19.7** |

Her 5 GABA sinapsından biri başka bir GABA nöronuna gidiyor. Bu "**disinhibisyon**" (çift negatif = pozitif):
- GABA nöronu A, bir hedefi susturur (inhibisyon)
- GABA nöronu B, A'yı susturur → hedef serbest kalır (disinhibisyon)
- Sonuç: dolaylı olarak uyarma efekti, ama zamanlaması kontrollü

---

## Motor Sistem Bağlantısı

| NT | Motor+Descending Sinaps | Sinaps/Nöron |
|-----|------------------------|-------------|
| **GABA** | **645,252** | **33.7** |
| Oktopamin | 3,375 | 15.6 |
| Serotonin | 18,393 | 8.1 |
| Dopamin | 9,347 | 1.6 |

**GABA nöron başına motor sisteme en çok sinyal gönderen NT!** (33.7/nöron) Oktopaminden bile 2x daha fazla.

---

## Özet: GABA'nın Rolü

```
                    BEYİN KONTROL SİSTEMİ

    UYARICI (GAZ)                    İNHİBİTÖR (FREN)
    ─────────────                    ─────────────────
    ACh (genel uyarı)                GABA (genel fren)
    Oktopamin (acil gaz)             │
    Dopamin (ödül/ceza)              ├── Görme filtreleme (%50)
                                     ├── Kendi kendini frenle (%20)
                                     ├── Navigasyon kontrolü (EB %78)
                                     └── Hafıza kapısı (Kenyon %13)
```

### GABA = Beyindeki Akıllı Fren Sistemi

1. **19,170 nöron (%13.8)** — her 7 nörondan biri frenleyici
2. **%92.6 intrinsic** — dışarı çıkmıyor, içeride düzenliyor
3. **%66 optik** — görme sinyallerini filtreliyor
4. **EB'de %78 baskın** — navigasyonu "tüm yönleri bastır, doğru olanı serbest bırak" ile kontrol ediyor
5. **Nöron başına en yüksek motor bağlantı** (33.7) — davranış frenlemesinde en aktif
6. **%20 kendi kendini susturuyor** — disinhibisyon ile dolaylı uyarma

### Dört Monoamin + GABA Karşılaştırması

| | Dopamin | Serotonin | Oktopamin | GABA |
|---|---------|-----------|-----------|------|
| **Nöron** | 5,909 | 2,282 | 216 | 19,170 |
| **Ana bölge** | Mushroom Body | Antennal Lobe | Medulla/Lobula | Medulla/EB |
| **İşlev** | Ödül/ceza | Duyusal kalibrasyon | Görüş keskinleştirme | Filtreleme/fren |
| **Motor/nöron** | 1.6 | 8.1 | 15.6 | **33.7** |
| **Analoji** | "İyi/kötü etiketi" | "Volume düğmesi" | "Adrenalin" | **"Akıllı fren"** |
