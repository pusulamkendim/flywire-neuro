# Analiz 08 - Oktopamin: Böceklerin Adrenalini

Script: `08_octopamine.py`

---

## Oktopamin Nöronları — Profil

- **216 nöron** (%0.2) — en nadir NT, beyindeki nöronların binde ikisi
- Güven skoru: 0.482 (diğerlerine göre düşük — bu nöronlar zor tanınıyor)

### Süper Sınıf Dağılımı

| Sınıf | Sayı | Oran |
|-------|------|------|
| sensory (duyusal) | 117 | %54.2 |
| optic (görme) | 20 | %9.3 |
| central (merkezi) | 18 | %8.3 |
| visual_centrifugal | 14 | %6.5 |
| endocrine (hormonal) | 13 | %6.0 |
| ascending | 11 | %5.1 |
| descending | 10 | %4.6 |
| motor | 3 | %1.4 |

**%54'ü duyusal** (serotonine benzer), ama profili çok farklı.

### Bilgi Akışı (Flow)

| Flow | Oran | Açıklama |
|------|------|----------|
| afferent (giriş) | %60 | Dışarıdan gelen sinyal |
| intrinsic (iç) | %29 | İç işleme |
| efferent (çıkış) | %11 | Motor/davranış çıkışı |

**%11 efferent** — dopamin (%0) ve serotonine (%0) kıyasla çok yüksek. Oktopamin doğrudan davranış çıkışına bağlı.

### Öne Çıkan Hücre Tipleri

| Tip | Sayı | Açıklama |
|-----|------|----------|
| **R1-6** | 89 | Fotoreseptörler — gözün temel ışık algılayıcıları |
| R7 | 8 | UV fotoreseptör |
| R8 | 7 | Renk fotoreseptörü |
| LC12 | 8 | Lobula Columnar — hareket algılama |
| T1 | 7 | Medulla nöronu — görme işleme |
| **OA-VUMa** serisi | ~10 | Klasik oktopaminerjik nöronlar (VUM = Ventral Unpaired Median) |
| **OA-AL2i** serisi | 8 | Antennal Lobe oktopamin nöronları |
| DH44 | 4 | Nöropeptid üreten endokrin nöronlar |

**%48'i fotoreseptör (R1-6, R7, R8)** — oktopamin görme sisteminde çok yoğun!

---

## Oktopamin Sinyali Nereye Gidiyor?

Oktopamin-baskın bağlantı: 143,947 / 16.8M — Toplam 224,734 sinaps

| Hedef Bölge | Sinaps | Oran | Açıklama |
|-------------|--------|------|----------|
| **ME_R (Medulla sağ)** | 61,406 | %27.3 | Görme işleme |
| **ME_L (Medulla sol)** | 39,692 | %17.7 | Görme işleme |
| LO_R (Lobula sağ) | 17,035 | %7.6 | Görme — hareket |
| LO_L (Lobula sol) | 12,502 | %5.6 | Görme — hareket |
| PVLP_R | 9,954 | %4.4 | Görsel entegrasyon |
| LOP_R | 7,506 | %3.3 | Görme işleme |
| AVLP_R | 7,077 | %3.1 | Çok duyusal entegrasyon |

**Sinapsların %58'i görme bölgelerinde (ME + LO).** Oktopamin bir "görme modülatörü".

---

## Bölge Baskınlığı: Oktopaminin Krallığı

Dopamin, Serotonin ve Oktopamin arasında bölge baskınlığı:

| Bölge | DA | SER | OCT | OCT % | Not |
|-------|-----|-----|-----|-------|-----|
| **ME_R** | 15,637 | 8,385 | **61,406** | **%72** | Oktopamin krallığı |
| **LOP_R** | 3,244 | 629 | **7,506** | **%66** | Görme |
| **LOP_L** | 1,977 | 351 | **3,291** | **%59** | Görme |
| **PVLP_R** | 6,347 | 1,250 | **9,954** | **%57** | Görsel entegrasyon |
| **ME_L** | 22,875 | 7,808 | **39,692** | **%56** | Oktopamin baskın |
| LO_R | 13,685 | 1,950 | 17,035 | %52 | Neredeyse eşit DA ile |

**Görme bölgelerinde oktopamin açık ara baskın monoamin!**

Karşılaştırma:
- **Mushroom Body** → Dopamin baskın (hafıza)
- **Antennal Lobe** → Serotonin baskın (koku)
- **Medulla/Lobula** → **Oktopamin baskın (görme)**

---

## Diğer Sistemlerle Etkileşim

| Yön | Bağlantı | Sinaps |
|-----|----------|--------|
| OCT → ACh | 38,842 | 57,389 |
| ACh → OCT | 35,097 | 76,230 |
| OCT → DA | 1,612 | 2,352 |
| DA → OCT | 1,053 | 1,434 |
| OCT → SER | 447 | 1,123 |
| SER → OCT | 695 | 1,443 |
| **OCT → GABA** | **0** | **0** |
| **GABA → OCT** | **0** | **0** |

**Kritik bulgu: Oktopamin ve GABA arasında SIFIR bağlantı!**
GABA inhibitör (frenleyici) sistem. Oktopamin uyarıcı (savaş-kaç) sistem. İkisinin birbirine dokunmaması mantıklı — fren ve gaz aynı anda çalışmaz.

### Ödül/Ceza Devresine Etkisi

| Hedef | Sinaps |
|-------|--------|
| OCT → PAM (ödül) | 53 |
| OCT → PPL1 (ceza) | 16 |
| OCT → Kenyon Cell | 793 |
| OCT → ALPN (koku projeksiyon) | 1,341 |
| OCT → ALLN (koku yerel) | 2,432 |

Serotonin gibi, oktopamin de ödül/ceza devresine neredeyse hiç dokunmuyor.
Ama koku işleme devresine (ALPN + ALLN: 3,773 sinaps) hafif bir etkisi var.

---

## Motor Sistem Bağlantısı

| NT | → Motor | → Descending | Sinaps/Nöron |
|-----|---------|-------------|--------------|
| Oktopamin | 103 | 3,272 | **15.6** |
| Serotonin | 1,912 | 16,481 | 8.1 |
| Dopamin | 334 | 9,013 | 1.6 |

**Oktopamin nöron başına motor sisteme 10x dopaminden, 2x serotoninden daha fazla sinyal gönderiyor.**

---

## Özet: Oktopaminin Rolü

```
                   HER NT'NİN KRALLIĞI

  KOKU              GÖRME              HAFIZA
  (Antennal Lobe)   (Medulla/Lobula)   (Mushroom Body)
       │                  │                  │
   SEROTONİN          OKTOPAMİN          DOPAMİN
   "kokuyu             "görüntüyü         "bunu
    kalibre et"         keskinleştir"      değerlendir"
       │                  │                  │
   2,282 nöron         216 nöron          5,909 nöron
```

### Oktopamin = Görme Sistemi Adrenalini

1. **%48'i fotoreseptör** — doğrudan gözün içinde
2. **%58 sinaps görme bölgesinde** — Medulla ve Lobula'ya odaklı
3. **GABA ile sıfır etkileşim** — fren sistemiyle çalışmıyor, sadece gaz
4. **Nöron başına en yüksek motor çıkış** — davranışa en doğrudan bağlı NT
5. **Ödül/ceza devresine dokunmuyor** — sadece "gör ve tepki ver"

Sinek tehlike gördüğünde: Oktopamin görme sistemini keskinleştiriyor ("daha net gör!"), motor sisteme sinyal gönderiyor ("kanat çırp!") — tıpkı insanda adrenalin pupilleri genişletip kasları harekete geçirdiği gibi.

### Üç Monoamin Sistemi Karşılaştırması

| | Dopamin | Serotonin | Oktopamin |
|---|---------|-----------|-----------|
| **Nöron** | 5,909 | 2,282 | 216 |
| **Ana bölge** | Mushroom Body | Antennal Lobe | Medulla/Lobula |
| **Ana duyu** | — (iç değerlendirme) | Koku | Görme |
| **İşlev** | Hafıza etiketleme | Duyusal kalibrasyon | Görüş keskinleştirme |
| **Motor bağlantı/nöron** | 1.6 | 8.1 | **15.6** |
| **GABA etkileşim** | Var | Var | **Sıfır** |
| **Analoji** | "Bu iyiydi/kötüydü" | "Volume düğmesi" | "Adrenalin" |
