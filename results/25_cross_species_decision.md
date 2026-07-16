# Analysis 25: Türler Arası Karar Devresi Karşılaştırması
# Sinek ↔ Fare ↔ İnsan: 600 Milyon Yıllık Korunmuş Mimari

**Veri Kaynakları:**
- FlyWire v783 connectome — 139,255 nöron, 50M+ sinaps (Dorkenwald et al., 2024)
- MICrONS — 54,887 nöron, 523M sinaps, fare görsel korteks (MICrONS Consortium, 2025)
- Allen Mouse Brain Connectivity Atlas — tüm fare beyni projeksiyon haritası
- Hippocampome.org v2.0 — 122 hipokampal nöron tipi (Wheeler et al., 2024)
- İnsan nörobilim literatürü — Sharot, Solomon, Kahneman ve diğerleri

---

## 1. Genel Karşılaştırma

Sinek beyni 139 bin, fare korteks parçası 55 bin, toplam fare beyni ~70 milyon,
insan beyni 86 milyar nöron içeriyor. Ama karar verme devrelerinin temel
prensipleri şaşırtıcı derecede benzer.

| Metrik | Sinek (FlyWire) | Fare Korteks (MICrONS) | Fare Hipokampüs | İnsan |
|--------|----------------|----------------------|-----------------|-------|
| İnhibitör nöron oranı | ~%20 (GABA) | %10.6 | ~%15 | ~%20 |
| İnhibitör alt tip | 1 (GABA) | 4+ (BC, MC, BPC, NGC) | 8+ (PV, SST, VIP, CCK...) | 20+ |
| Self-inhibisyon (I→I) | %19.7 | %9.5 | bilinmiyor | bilinmiyor |
| Ana excitatory NT | Asetilkolin | Glutamat | Glutamat | Glutamat |
| Dopamin nöron sayısı | 323 | — (korteks dışı) | — (VTA'dan gelir) | ~400,000 |

**Yorum:** İnhibitör nöron oranı %10-20 arasında değişiyor — tür fark etmeksizin
excitatory nöronlar her zaman çoğunluk. Ama inhibisyon oranı düşük olmasına rağmen,
etkisi orantısız büyük. Sinekte GABA'yı kaldırınca motor nöron aktivitesi %4,850
artıyor. Fare korteksinde GABA bozulması şizofreni benzeri tablolara yol açıyor.

---

## 2. Altı Korunan Prensip

### Prensip 1: "Önce Güvenlik Kontrolü" (Negativity Bias)

Beyin yeni bir uyarana karşı önce kaçınma, sonra yaklaşma tepkisi veriyor.

| | Sinek | Fare | İnsan |
|---|---|---|---|
| **Mekanizma** | Kaçınma MBON'ları yaklaşmadan önce aktive oluyor (t+3-6) | Amigdala ~120ms'de tepki veriyor, korteks henüz analiz yapmadan | Amigdala yüz ifadesini 33ms'de tespit ediyor, bilinçli farkındalık ~500ms |
| **Devre** | ORN → MBON (avoidance) | Thalamus → Amygdala (düşük yol) | Thalamus → Amygdala (LeDoux yolu) |
| **Kanıt** | FlyWire simülasyon, valans t+3'te negatif | Allen: Amyg bağlantıları çok hızlı | Rozin & Royzman 2001, "bad is stronger than good" |

**Neden korunmuş:** Tehditten kaçınmayı kaçırmak = ölüm. Ödülü kaçırmak = sadece
fırsat kaybı. Doğal seçilim "önce güvenlik" stratejisini koruyor.

### Prensip 2: İki Hızlı Karar Sistemi

Her üç türde de hızlı-doğuştan ve yavaş-öğrenilmiş iki paralel yol var.

| | Sinek | Fare | İnsan |
|---|---|---|---|
| **Hızlı yol** | Lateral Horn (t+2) | Amygdala | Sistem 1 (Kahneman) |
| **Ne yapar** | "Genel olarak tehlikeli mi?" | "Korkmalı mıyım?" | Sezgisel, otomatik |
| **Yavaş yol** | Mushroom Body (t+3) | Hippocampus + PFC | Sistem 2 (Kahneman) |
| **Ne yapar** | "Geçen sefer ne olmuştu?" | "Deneyimime göre ne yapmalıyım?" | Analitik, bilinçli |
| **Entegrasyon** | 34 inen nöron (her iki yoldan input) | Striatum (amygdala + PFC) | PFC (duygu + mantık) |

**Veri desteği:**
- Sinek: LH 487 sinapsla 74 inen nörona, MB 3,126 sinapsla 160 inen nörona bağlanıyor.
  Hızlı yol zayıf ama erken, yavaş yol güçlü ama geç. (FlyWire Analysis 19)
- Fare: Allen verisinde Amygdala→Striatum (270.78) ve PFC→Striatum (282.01) neredeyse
  eşit — iki yol striatum'da buluşuyor.

### Prensip 3: Opponent Processing (Karşıt Süreç)

Her valans sinyali karşıtını modüle ediyor — ödül ve ceza ayrı kanallar değil,
birbirine bağlı bir denge sistemi.

| | Sinek | Fare | İnsan |
|---|---|---|---|
| **Mekanizma** | PPL1 boost → paradoksal yaklaşma | D1/D2 reseptörleri zıt etki | Solomon opponent process |
| **Devre** | MBON avoidance → PAM feedback (3.2x) | VTA-DA → CA1 (D1=LTP, D2=LTD) | Dopamin ↔ CRF/noradrenalin dengesi |
| **Sonuç** | Ceza sinyali güçlendirilince karşıt valans ortaya çıkıyor | Aynı dopamin nöronu D1 üzerinden güçlendirme, D2 üzerinden zayıflatma yapıyor | Acıdan sonra öfori, zevkten sonra çekilme |

**Veri desteği:**
- Sinek: PPL1 3x boost → valans +0.115 (yaklaşma). PAM 3x boost → valans -0.316
  (kaçınma). Paradoksal ama opponent processing ile açıklanıyor. (FlyWire Analysis 19)
- Fare: Aynı CA1 piramidal nöronunda D1 reseptörü sinapsı güçlendirirken D2
  zayıflatıyor — moleküler seviyede opponent processing.

### Prensip 4: GABA ile Karar Keskinleştirme

İnhibisyon olmadan karar sinyali gürültüde kaybolur.

| | Sinek | Fare Korteks | Fare Hipokampüs | İnsan |
|---|---|---|---|---|
| **İnhibisyon olmadan** | Motor nöron %4,850 artış | — | — | Şizofreni benzeri tablo |
| **GABA ne yapar** | Tüm MBON'lar yerine sadece doğru olanlar ateşleniyor | BC: hızlı veto, MC: input filtre, NGC: global sessizleştirme | PV+: gamma osilasyon (dikkat), SST+: dendritik kapı | PFC GABA↓ → karar bozukluğu |
| **Self-inhibisyon** | %19.7 (GABA→GABA) | BC %8.7, MC %9.4, BPC %33.7, NGC %13.3 | Bilinmiyor | Bilinmiyor |

**MICrONS'tan yeni bulgu:** BPC (Bipolar cell) %33.7 self-inhibisyon oranıyla
disinhibisyon yapıyor — diğer inhibitörleri inhibe ederek net excitasyon sağlıyor.
Bu, öğrenme sırasında inhibisyonu geçici olarak kaldıran bir "kapı açma" mekanizması.

**İnhibitör alt tipler — türler arası eşleştirme:**

| MICrONS (fare korteks) | Hipokampüs | Sinek | Fonksiyon |
|------------------------|-----------|-------|-----------|
| BC (Basket, PV+) | PV+ Basket | GABA MBON (kısmen) | Hızlı veto: "dur, ateşleme" |
| MC (Martinotti, SST+) | SST+ O-LM | — | Input filtresi: "bu sinyali görmezden gel" |
| BPC (Bipolar, VIP+) | VIP+ IS | — | Disinhibisyon: "öğrenme kapısını aç" |
| NGC (Neurogliaform) | Neurogliaform | — | Global fren: "her şeyi yavaşlat" |

### Prensip 5: Yapısal İyimserlik Bias'ı

Karar verdikten sonra sistem ödül arayışına dönmeye yapısal olarak eğilimli.

| | Sinek | Fare | İnsan |
|---|---|---|---|
| **Geri bildirim** | MBON → PAM 1.7x (PPL1'e göre) | PFC → VTA 1.8x (SNc'ye göre) | L-DOPA → iyimserlik artışı |
| **Kaçınma sonrası** | Avoidance MBON bile PAM'a 3.2x feedback | Amygdala → VTA ödül feedback | Kötü haber sonrası inanç güncelleme azalır |
| **Mekanizma** | Yapısal kablolama (sinaps sayısı) | Projeksiyon asimetrisi | Dopaminerjik asimetrik güncelleme |

**Veri desteği:**
- Sinek: 5,480 MBON→PAM sinapsı vs 3,195 MBON→PPL1. Kaçınma MBON'ları bile
  2,789 sinapsla PAM'a, sadece 870 sinapsla PPL1'e bağlanıyor. (FlyWire)
- Fare: PL (prefrontal) → VTA: 23.39 vs PL → SNc: 13.17 energy. Karar
  merkezi ödül merkezine daha güçlü bağlı. (Allen Atlas)
- İnsan: Sharot et al. 2012 — L-DOPA (dopamin artışı) kötü haberlere karşı
  inanç güncellemeyi azaltıyor. Nüfusun %80'inde gözlenen fenomen.

### Prensip 6: Yaklaşma Baskınlığı

Varsayılan davranış yaklaşma; kaçınma bunu baskılayarak çalışıyor.

| | Sinek | Fare | İnsan |
|---|---|---|---|
| **Yaklaşma nöronları** | 52/96 MBON (%54) | E→E %78.6 (varsayılan excitasyon) | Sol PFC aktivasyonu = yaklaşma |
| **Susturma etkisi** | Yaklaşma MBON off → -%22, Kaçınma off → -%5 | — | Sol PFC hasarı → depresyon |
| **Kaçınma nasıl** | GABA + PPL1 ile yaklaşmayı override | I→E inhibisyonla excitasyonu baskıla | Sağ PFC / amygdala ile override |

**Yorum:** Kaçınma ayrı bir sistem değil, yaklaşma sisteminin üzerine
bindirilen bir "fren" mekanizması. Fren bozulursa (GABA↓) → mania/dürtüsellik.
Fren aşırı güçlüyse (GABA↑) → depresyon/kaçınma.

---

## 3. Dopamin Sistemi Detaylı Karşılaştırma

### Dopaminin ana hedefleri

| Hedef Bölge | VTA (ödül) | SNc (ceza/motor) | Sinek Karşılığı |
|-------------|-----------|-----------------|-----------------|
| N. Accumbens (motivasyon) | **231.12** | 15.69 | PAM → γ lobu (yeni bellek) |
| Dorsal Striatum (alışkanlık) | 115.13 | **100.33** | PPL1 → α/β lobu |
| Dentate Gyrus (yeni bellek) | **40.56** | 0.07 | PAM → γ lobu |
| Amygdala (duygu) | 62.05 | 62.39 | Her iki DA → MB |
| PFC (planlama) | 20.94 | 1.12 | MBON → Descending |
| Görsel Korteks | 0.15 | 0.03 | — |

**En çarpıcı paralel:** Ödül dopamini (VTA/PAM) en güçlü olarak yeni bellek
oluşturma kapısına gidiyor (DG/γ lobu). "Bu olay önemliydi, hatırla" sinyali
her iki türde de aynı mantıkla çalışıyor.

### Dopaminin inhibitör nöronlar üzerindeki etkisi

| İnhibitör Tip | Dopamin Etkisi | Sonuç | Öğrenme İçin Anlamı |
|--------------|---------------|-------|-------------------|
| PV+ Basket | Hızlandırır | Gamma osilasyon ↑ | Dikkat keskinleşir, alakasız sinyaller bastırılır |
| SST+ O-LM | Yavaşlatır | Dendritik kapı açılır | Daha fazla bilgi piramidal hücrelere ulaşır |
| VIP+ (disinhibisyon) | Güçlendirir | İnhibisyon geçici kalkar | Öğrenme kapısı açılır, yeni bellek yazılabilir |
| CCK+ Basket | Endokannabinoid modülasyonu | Esneklik | Bellek güncelleme kolaylaşır |

**Bu tablo sinekte tek bir satır olurdu:** GABA — dopamin modüle eder — karar keskinleşir.
Fare ve insanda bu tek mekanizma 4+ özelleşmiş alt tipe ayrılmış. Evrim aynı prensibi
korurken **çözünürlüğü artırmış**.

---

## 4. Veri Kaynaklarının Güçlü ve Zayıf Yanları

| Veri Kaynağı | Kapsam | Çözünürlük | NT Bilgisi | Dopamin | Güçlü Yan |
|-------------|--------|-----------|-----------|---------|-----------|
| **FlyWire** | Tüm beyin | Her sinaps | 6 NT skoru | Doğrudan (PAM/PPL1) | Tam beyin, tam NT |
| **MICrONS** | 1mm³ korteks | Her sinaps | Hücre tipinden çıkarım | Yok (korteks dışı) | Sinaps boyutu, inh alt tipler |
| **Allen Atlas** | Tüm beyin | Bölge seviyesi | Yok | Projeksiyon haritası | Makro bağlantılar, VTA/SNc |
| **Hippocampome** | Hipokampüs | Hücre tipi | Literatürden | D1/D2 reseptör bilgisi | 122 nöron tipi, sinaptik fizyoloji |

**Eksik olan:** Fare hipokampüsünün veya striatum'un sinaps-seviye connectome'u.
MouseConnects projesi (NIH, $40M, 2023-2028) bunu hedefliyor ama veri henüz yok.
Bu veri çıktığında, FlyWire MB ↔ fare hipokampüs doğrudan sinaps-sinaps
karşılaştırması mümkün olacak.

---

## 5. Sonuç: 600 Milyon Yıllık Korunmuş Mimari

Böcekler ve memeliler ~600 milyon yıl önce ayrıldı. Sinek beyni 139K nöron,
insan beyni 86 milyar. Ama karar verme devreleri altı ortak prensip üzerine kurulu:

```
1. Önce güvenlik kontrolü    → Tehdit sinyali her zaman ödülden önce işleniyor
2. İki hızlı sistem          → Hızlı doğuştan + yavaş öğrenilmiş paralel yollar
3. Karşıt süreç              → Ödül ve ceza birbirini modüle eden denge sistemi
4. İnhibitör filtreleme      → GABA olmadan karar sinyali gürültüde kaybolur
5. İyimserlik bias'ı         → Geri bildirim ödül merkezine daha güçlü bağlı
6. Yaklaşma baskınlığı       → Kaçınma, yaklaşmayı override eden fren mekanizması
```

Bu prensiplerin korunmuş olması tesadüf değil — bunlar **etkili karar vermenin
matematiksel gereklilikleri:**
- Asimetrik maliyet → negativity bias (yanlış alarm ucuz, kaçırma pahalı)
- Hız-doğruluk dengeleme → iki hızlı sistem
- Homeostaz → opponent processing
- Sinyal/gürültü → inhibitör filtreleme
- Keşif teşviki → iyimserlik bias'ı (exploration-exploitation)
- Enerji verimliliği → yaklaşma varsayılanı (aktif inhibisyon pahalı)

---

*Veri: FlyWire v783 (Dorkenwald et al., 2024), MICrONS (MICrONS Consortium, 2025),
Allen Mouse Brain Connectivity Atlas, Hippocampome.org v2.0.
Destekleyen literatür: Aso et al. 2014, Bennett et al. 2022, Berry et al. 2024,
Sharot et al. 2012, Solomon & Corbit 1974, Kahneman 2011.
Code: `25_cross_species_decision.py`*
