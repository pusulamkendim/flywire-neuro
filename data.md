# ELI5: Meyve Sineği Beyni — 5 Yaşına Anlatır Gibi

## Bu proje ne?

Bir meyve sineğinin (*Drosophila melanogaster*) **tüm beynini** haritaladık — her bir kablo, her bağlantı, her kimyasal sinyal. Bunu **139.255 işlemcili** (nöron) ve aralarında **50 milyon kablolu** (sinaps) minicik bir bilgisayarın komple devre şeması gibi düşün.

---

## Beyin hücreleri nasıl konuşuyor?

Bir şehrin telefon ağını hayal et:

```
Nöron A  ──(7 arama)──►  Nöron B
              mahalle: AVLP_R
              dil: %65 GABA, %27 Glutamat, %2 Asetilkolin...
```

- **Nöron A** = arayan kişi (sinyal gönderen)
- **Nöron B** = aranan kişi (sinyal alan)
- **Sinaps sayısı (7)** = bağlantı ne kadar güçlü (fazla sinaps = daha yüksek sesle konuşma)
- **Beyin bölgesi** = hangi mahallede yaşıyorlar
- **Nörotransmitter** = hangi "dil" konuşuyorlar

Verimizdeki her satır = *"A nöronu, B nöronuna, şu bölgede, şu kimyasalla sinyal gönderiyor."*

---

## 6 Kimyasal Dil

Beyin hücreleri sadece elektrik kullanmaz — kimyasallarla konuşur. Sinek beyninde tam 6 tane var:

| Kimyasal | Ne diyor? | Günlük hayat karşılığı |
|----------|-----------|----------------------|
| **Asetilkolin (ACh)** | "Yap!" | Emir veren patron — en yaygın (%48) |
| **GABA** | "Dur!" | Fren pedalı — aşırı tepkiyi önler (%23) |
| **Glutamat** | "Yap!" veya "Dur!" | Joker — duruma göre değişir (%19) |
| **Dopamin** | "Bu iyiydi, hatırla!" | Öğretmen — ödül ve öğrenme (%4.4) |
| **Serotonin** | "Sakin ol, dikkat et" | Ruh hali ayarlayıcı — duyuları ince ayar yapar (%2.1) |
| **Oktopamin** | "Tehlike! Uyan!" | Adrenalin — savaş ya da kaç (%2.7) |

---

## Ne bulduk?

### 1. Her beyin bölgesi kendi dilini konuşuyor

Bir şehirde farklı mahallelerin farklı diller konuşması gibi, farklı beyin bölgeleri farklı kimyasalları tercih ediyor:

- **Mantar Cisimciği** (hafıza merkezi) → %90 Asetilkolin — "Bunu hatırla!"
- **Elipsoid Cisim** (navigasyon) → %78 GABA — "Gürültüyü filtrele!"
- **Antenal Lob** (koku merkezi) → Serotonin yoğun — "Dikkatli kokla!"
- **Medulla/Lobula** (görme) → Oktopamin yoğun — "Tetikte kal!"

### 2. Koku vs tat: tamamen farklı iki strateji

Sineğin bir şey koklaması ile tatması arasında ne olduğunu simüle ettik:

| | Koku | Tat |
|---|---|---|
| **Harekete geçme hızı** | 7 adımda kaslara ulaşır | 1 adım — anında refleks! |
| **Hafıza devreye giriyor mu?** | Evet — geçmiş deneyimleri kontrol eder | Hayır — hafızayı tamamen atlıyor |
| **Strateji** | "Bi düşüneyim..." | "Hemen tükür!" |

**Neden?** Koku uzaktan gelir — değerlendirmeye zaman var. Tat ise şey zaten ağzında — hemen harekete geçmen lazım.

### 3. Büyük keşif: Tehlike ödülden önce gelir

Bu, daha önce kimsenin raporlamadığı **ana bulgumuz**.

Sinek beyninde iki alarm sistemi var:
- **PPL1** (16 nöron) = "Bu TEHLİKELİ!" (ceza)
- **PAM** (307 nöron) = "Bu GÜZEL!" (ödül)

Bir koku geldiğinde **PPL1 her zaman önce ateşleniyor** — PAM'dan 2-3 adım önce. Her zaman. Bunu 400 farklı şekilde test ettik ve **bir kez bile** ödül cezayı geçmedi.

```
Koku gelir → ... → PPL1 ateşlenir (4. adım) → ... → PAM ateşlenir (7. adım)
                    "TEHLİKE!"                        "hmm, güzelmiş"
```

**Bu neden önemli?**

Evrim açısından düşün:
- Tehlike sinyalini kaçırmak = **ölürsün**
- Ödül sinyalini kaçırmak = **bir atıştırmalığı kaçırırsın**

Bu yüzden beyin tehditleri önce kontrol edecek şekilde kablolanmış. Yılan gördüğünde, onun sadece bir çubuk olabileceğini fark etmeden önce nasıl zıplıyorsan — aynı mantık.

### 4. PPL1 neden daha hızlı? Beş neden:

| | PPL1 (tehlike) | PAM (ödül) |
|---|---|---|
| **Nöron sayısı** | 16 (küçük, kompakt) | 307 (büyük, dağınık) |
| **Nöron başına girdi** | 4.804 sinaps | 414 sinaps |
| **Nasıl aktive oluyor** | Doğrudan uyarı (hızlı) | Başka nöronlara bağımlı (yavaş) |
| **GABA'ya ihtiyacı var mı?** | Hayır — GABA olmadan da çalışır | Hayır — ama GABA onu daha çok yavaşlatır |
| **Doygunluk** | Her zaman %100 (ya hep ya hiç) | Asla %100'e ulaşmaz (kademeli) |

PPL1 bir **duman dedektörü** gibi — küçük, her zaman açık, anında tetiklenir.
PAM bir **yemek eleştirmeni** gibi — büyük ekip, değerlendirmesi zaman alır.

### 5. Sinek öğrenebiliyor

Gerçek beyin kablolaması üzerinde öğrenme simülasyonu yaptık:
- **Koku A + şeker** (10 deneme) → sinek yaklaşmayı öğreniyor
- **Koku B + şok** (10 deneme) → sinek kaçmayı öğreniyor
- **Koku C** (hiçbir şeyle eşleşmemiş) → sinek umursamıyor — yanlış bağlantı yok!
- **Şekeri kes** (15 deneme) → sinek yavaşça unutuyor (%81 söndürme)

Bu işe yarıyor çünkü her koku hafıza hücrelerinin sadece ~%10'unu aktive ediyor (seyrek kodlama), böylece farklı kokular birbirine karışmıyor.

---

## Özet

Bir haşhaş tohumu büyüklüğündeki sinek beyni, güzelce organize edilmiş bir sistem evrimleştirmiş:
- **Önce tehlikeyi kontrol et** (PPL1, PAM'dan önce)
- **Tada anında tepki ver** (kaslara 1 adım)
- **Kokuyu dikkatli değerlendir** (7 adım, hafıza devrede)
- **Deneyimlerden öğren** (ödül ve ceza gerçek sinapsları değiştiriyor)

Bunların hepsi beynin fiziksel yapısına gömülü — yazılım tercihi değil, mimari.

---

*Veri: FlyWire v783 connectome (Dorkenwald et al., 2024). Kod: [github.com/pusulamkendim/flywire-neuro](https://github.com/pusulamkendim/flywire-neuro)*
