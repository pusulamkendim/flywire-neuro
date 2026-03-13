# Analiz 11 - Glutamat: Çift Taraflı Ajan

Script: `11_glutamate.py`

---

## Glutamat Nöronları — Profil

- **24,875 nöron** (%17.9) — üçüncü en büyük NT grubu
- Güven skoru: 0.611 (en düşük güven — sınıflandırması zor)

### Süper Sınıf Dağılımı

| Sınıf | Sayı | Oran |
|-------|------|------|
| optic (görme) | 13,291 | %53.4 |
| central (merkezi) | 7,067 | %28.4 |
| sensory (duyusal) | 3,005 | %12.1 |
| visual_projection | 1,019 | %4.1 |

**%53 optik, %28 central** — Glutamat hem görme hem merkezi işlemede güçlü.

### Bilgi Akışı

| Flow | Oran |
|------|------|
| intrinsic (iç) | %86.3 |
| afferent (giriş) | %12.9 |
| efferent (çıkış) | %0.8 |

### Öne Çıkan Hücre Tipleri

| Tip | Sayı | Açıklama |
|-----|------|----------|
| R1-6 | 1,675 | Fotoreseptörler |
| Mi9 | 1,258 | Medulla Intrinsic — parlaklık |
| TmY5a | 1,166 | Transmedullary — hareket |
| L1 | 1,054 | Lamina monopolar — kontrast |
| R7 | 730 | UV fotoreseptör |

---

## Glutamat Sinyali Nereye Gidiyor?

Glutamat-baskın bağlantı: 2,782,108 / Toplam sinaps: 7,825,277

| Hedef Bölge | Sinaps | Oran |
|-------------|--------|------|
| ME_R | 1,129,331 | %14.4 |
| ME_L | 998,332 | %12.8 |
| LO_R | 563,450 | %7.2 |
| LO_L | 512,475 | %6.5 |
| LOP_R | 372,802 | %4.8 |
| GNG | 351,837 | %4.5 |
| **FB (Fan-shaped Body)** | **298,547** | **%3.8** |
| SMP_R | 266,526 | %3.4 |

Görme bölgeleri ağırlıklı (%41), ama dikkat çeken: **FB (Fan-shaped Body)** ve **SMP** bölgelerinde yüksek yoğunluk.

---

## Glutamat Baskınlığı — Nerede Öne Çıkıyor?

| Bölge | Glut | ACh | GABA | Glut % | Not |
|-------|------|-----|------|--------|-----|
| **PB (Protocerebral Bridge)** | 25,474 | 26,174 | 838 | **%48.5** | Neredeyse ACh ile eşit! |
| **FB (Fan-shaped Body)** | 298,547 | 335,938 | 33,169 | **%44.7** | Merkezi komplekste güçlü |
| SMP_L | 253,436 | 430,965 | 64,000 | %33.9 | Üst medial protokerebrum |
| SMP_R | 266,526 | 473,567 | 84,690 | %32.3 | |
| SLP | 183,553 | 395,672 | 95,870 | %27.2 | Üst lateral protokerebrum |
| LOP_R | 372,802 | 805,342 | 335,727 | %24.6 | Lobula plate |

**PB ve FB: Merkezi komplekste glutamat neredeyse ACh kadar güçlü!**

Merkezi kompleks (CX) = navigasyon + karar + motor planlama merkezi. Burada:
- GABA → yön seçimi (EB'de %78)
- Glutamat → motor planlama (FB'de %45, PB'de %49)
- ACh → genel bilgi aktarımı

---

## Kritik Devrelerdeki Glutamat

| Hedef | Sinaps | Yorum |
|-------|--------|-------|
| **Glut → ALPN (koku proj.)** | **75,295** | Koku çıkışını modüle et |
| **Glut → MBON (çıkış)** | **31,562** | Karar çıkışında güçlü |
| Glut → PAM (ödül) | 12,710 | Ödül devresine sinyal |
| Glut → PPL1 (ceza) | 9,703 | Ceza devresine sinyal |
| Glut → Kenyon Cell | 7,290 | Hafıza hücrelerine düşük |
| Glut → ORN (koku) | 1,443 | Koku girişine minimal |

**MBON'lara 31.5K sinaps** — Glutamat karar çıkışında (MBON) dopaminden sonra en etkili NT.

---

## Glutamaterjik MBON'lar — Özel Durum

MBON'ların (Mushroom Body çıkış nöronları) NT dağılımı:

| NT | Sayı | Oran |
|----|------|------|
| ACh | 52 | %54.2 |
| **Glutamat** | **25** | **%26.0** |
| GABA | 19 | %19.8 |

**MBON'ların %26'sı glutamaterjik** — ve bunlar özel bir rol oynuyor:

| MBON Tipi | → PAM (ödül) | → PPL1 (ceza) |
|-----------|-------------|---------------|
| **Glut-MBON** | **2,789** | 870 |
| ACh-MBON | 1,780 | 1,638 |
| GABA-MBON | 911 | 687 |

**Glutamaterjik MBON'lar PAM'a (ödül) en çok sinyal gönderen grup!** Böceklerde glutamat MBON üzerinden inhibitör etki yapabilir — yani bu MBON'lar ödül nöronlarını "bastırıyor" olabilir. Bu "olumsuz geri bildirim" döngüsü: "Yeterince ödül aldın, dur."

---

## Motor Sistem

| NT | Motor+Desc Sinaps | Sinaps/Nöron |
|----|-------------------|-------------|
| GABA | 645,252 | 33.7 |
| **Glutamat** | **438,225** | **17.6** |
| ACh | 1,343,688 | 15.6 |
| Oktopamin | 3,375 | 15.6 |
| Serotonin | 18,393 | 8.1 |
| Dopamin | 9,347 | 1.6 |

Nöron başına motor çıkışta glutamat ikinci sırada (17.6/nöron).

---

## Özet: Glutamatın Rolü

```
             MERKEZI KOMPLEKS (Navigasyon Merkezi)

   EB (%78 GABA)         FB (%45 Glut)         PB (%49 Glut)
   "Yönleri bastır"      "Motor planla"        "Baş yönü hesapla"
        │                     │                      │
     İNHİBİSYON           PLANLAMA              HESAPLAMA
     (GABA)               (Glutamat)            (Glutamat)
```

### Glutamat = Beyindeki Motor Planlamacı ve Çift Taraflı Ajan

1. **24,875 nöron (%18)** — üçüncü büyük grup
2. **FB ve PB'de ~%45-49** — navigasyon merkezinde ACh kadar güçlü
3. **MBON üzerinden ödül freni** — Glut-MBON'lar PAM'a en çok sinyal gönderir (inhibitör → "yeter, dur")
4. **Çift rolü var** — böceklerde glutamat hem uyarıcı hem inhibitör olabilir (reseptöre bağlı)
5. **Motor planlama odaklı** (17.6/nöron) — nöron başına motor çıkışta ikinci

Analoji: Glutamat bir **GPS navigasyon sistemi**. Nereye gideceğini planlar (FB), baş yönünü hesaplar (PB), ve "hedefe vardın, dur" sinyali verir (MBON→PAM freni).

---

## 6 NT Karşılaştırma Tablosu (Tümü)

| | ACh | GABA | Glutamat | Dopamin | Serotonin | Oktopamin |
|---|-----|------|----------|---------|-----------|-----------|
| **Nöron** | 86,188 | 19,170 | 24,875 | 5,909 | 2,282 | 216 |
| **Oran** | %62 | %14 | %18 | %4 | %2 | %0.2 |
| **Ana bölge** | Her yer / MB | EB / Görme | FB / PB | Mushroom Body | Antennal Lobe | Medulla |
| **İşlev** | Genel uyarı | Filtreleme/fren | Motor planlama | Ödül/ceza | Duyusal kalibrasyon | Görüş keskinleştirme |
| **Motor/nöron** | 15.6 | 33.7 | 17.6 | 1.6 | 8.1 | 15.6 |
| **Analoji** | Elektrik şebekesi | Akıllı fren | GPS navigasyon | İyi/kötü etiketi | Volume düğmesi | Adrenalin |
