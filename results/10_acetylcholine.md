# Analiz 10 - Asetilkolin (ACh): Beyindeki Ana Uyarıcı

Script: `10_acetylcholine.py`

---

## ACh Nöronları — Profil

- **86,188 nöron** (%61.9) — her 5 nörondan 3'ü ACh kullanıyor
- Güven skoru: 0.700 (en yüksek güven)

### Süper Sınıf Dağılımı

| Sınıf | Sayı | Oran |
|-------|------|------|
| optic (görme) | 51,403 | %59.6 |
| central (merkezi) | 14,573 | %16.9 |
| sensory (duyusal) | 10,683 | %12.4 |
| visual_projection | 6,884 | %8.0 |
| ascending | 1,135 | %1.3 |
| descending | 873 | %1.0 |

**%60'ı görme sisteminde** — ACh görmenin ana dili.

### Bilgi Akışı

| Flow | Oran |
|------|------|
| intrinsic (iç) | %84.7 |
| afferent (giriş) | %14.2 |
| efferent (çıkış) | %1.1 |

### Öne Çıkan Hücre Tipleri

| Tip | Sayı | Açıklama |
|-----|------|----------|
| R1-6 | 5,342 | Fotoreseptörler — gözün temel algılayıcıları |
| T2a | 1,782 | Medulla — hareket algılama |
| Tm3 | 1,756 | Transmedullary — renk/parlaklık |
| T4c/T4d | 3,258 | Hareket yönü seçiciliği |
| L2 | 1,668 | Lamina — ilk görme işleme |

---

## ACh Sinyali Nereye Gidiyor?

ACh-baskın bağlantı: 8,488,348 / Toplam sinaps: 28,180,519

| Hedef Bölge | Sinaps | Oran |
|-------------|--------|------|
| ME_R (Medulla sağ) | 4,309,080 | %15.3 |
| ME_L (Medulla sol) | 3,578,343 | %12.7 |
| LO_R (Lobula sağ) | 2,305,862 | %8.2 |
| LO_L (Lobula sol) | 2,279,458 | %8.1 |
| GNG | 1,277,161 | %4.5 |
| AVLP_R | 1,183,681 | %4.2 |

**Sinapsların %44'ü görme bölgelerinde (ME + LO).** ACh her yerde ama görme sistemi ağırlıklı.

---

## ACh Baskınlığı — Mushroom Body Hakimiyeti

ACh'nin diğer büyük NT'lerden (Glut + GABA) en baskın olduğu bölgeler:

| Bölge | ACh | Glut | GABA | ACh % |
|-------|-----|------|------|-------|
| **MB_VL_R** | 93,326 | 2,059 | 7,699 | **%90.5** |
| **MB_ML_R** | 192,779 | 3,918 | 16,761 | **%90.3** |
| **MB_ML_L** | 247,247 | 9,555 | 24,435 | **%87.9** |
| **MB_CA_L** | 222,928 | 2,655 | 31,422 | **%86.7** |
| **MB_VL_L** | 72,765 | 5,582 | 6,700 | **%85.6** |
| AL_R | 287,739 | 19,163 | 88,615 | %72.8 |
| AL_L | 298,504 | 17,698 | 96,648 | %72.3 |

**Mushroom Body'de ACh %85-90 baskın!** Hafıza ve karar merkezinin ana iletişim dili ACh. Dopamin orada "etiketi" yapıştırırken, ACh "bilgiyi" taşıyor.

---

## Kritik Devrelerdeki ACh

| Hedef | Sinaps | Yorum |
|-------|--------|-------|
| **ACh → ALLN (koku yerel)** | **444,420** | Koku ağını uyar |
| **ACh → Kenyon Cell** | **360,160** | Hafıza hücrelerine bilgi taşı |
| **ACh → ALPN (koku proj.)** | **281,184** | Koku çıkışını uyar |
| ACh → MBON (çıkış) | 44,597 | Karar çıkışı |
| ACh → ORN (koku) | 29,605 | Koku reseptörlerine geri bildirim |
| ACh → PPL1 (ceza) | 22,228 | Ceza sinyalini uyar |
| ACh → PAM (ödül) | 20,394 | Ödül sinyalini uyar |

ACh hem Kenyon Cell'lere bilgi taşıyor (360K) hem de koku işleme devresini uyarıyor (444K + 281K).

---

## Diğer Sistemlerle Etkileşim

| Yön | Sinaps | Yorum |
|-----|--------|-------|
| ACh → ACh | 15,316,470 | Kendi kendine en büyük — zincirleme uyarma |
| ACh → GABA | 8,177,507 | Freni de uyarıyor (denge mekanizması) |
| ACh → Glut | 5,921,492 | Diğer uyarıcıya sinyal |
| ACh → DA | 724,826 | Dopamini uyar |
| ACh → SER | 240,509 | Serotonini uyar |
| ACh → OCT | 76,230 | Oktopamini uyar |

**ACh → GABA: 8.2M sinaps** — ACh her yerde herkesi uyarıyor, ama aynı zamanda fren sistemini (GABA) de çalıştırıyor. Gaz verirken aynı anda freni de hazırlıyor → aşırı uyarılmayı önleyen denge.

---

## Motor Sistem

| NT | Motor+Desc Sinaps | Sinaps/Nöron |
|----|-------------------|-------------|
| GABA | 645,252 | 33.7 |
| Glutamat | 438,225 | 17.6 |
| **ACh** | **1,343,688** | **15.6** |
| Oktopamin | 3,375 | 15.6 |
| Serotonin | 18,393 | 8.1 |
| Dopamin | 9,347 | 1.6 |

Toplam motor sinaps sayısında ACh açık ara birinci (1.3M), ama nöron başına oktopamin ve GABA ile aynı seviyede.

---

## Özet: ACh'nin Rolü

**ACh = Beyindeki Elektrik Şebekesi**

1. **86,188 nöron (%62)** — beynin varsayılan iletişim dili
2. **Mushroom Body'de %90** — hafıza ve karar bilgisini taşıyan ana hat
3. **Herkesi uyarıyor** — DA, SER, OCT, GABA hepsine sinyal gönderiyor
4. **GABA'yı da uyarıyor** — gaz verirken freni hazırlıyor (denge)
5. **Görme odaklı** (%60 optik) — görme bilgisinin taşınmasında ana rol

Analoji: ACh bir şehirdeki **elektrik şebekesi**. Her binaya (nörona) temel enerjiyi sağlıyor. Diğer NT'ler özel sistemler (su, doğalgaz, internet), ama elektrik olmadan hiçbiri çalışmaz.
