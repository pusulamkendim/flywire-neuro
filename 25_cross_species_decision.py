"""
FlyWire Analysis 25 — Cross-Species Decision Circuit Comparison
Sinek ↔ Fare ↔ İnsan: Karar Verme Mimarisinin Evrimsel Korunmuşluğu

4 veri kaynağını birleştirerek türler arası karşılaştırma:
  1. FlyWire v783 (sinek, tüm beyin, sinaps seviyesi)
  2. MICrONS (fare görsel korteks, sinaps seviyesi)
  3. Allen Brain Connectivity Atlas (fare tüm beyin, bölge seviyesi)
  4. Hippocampome.org + literatür (fare hipokampüs, hücre tipi seviyesi)

Soru: Karar verme devrelerinin temel prensipleri 600 milyon yıllık
evrim boyunca korunmuş mu?
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 80)
print("ANALYSIS 25: CROSS-SPECIES DECISION CIRCUIT COMPARISON")
print("Sinek ↔ Fare ↔ İnsan")
print("=" * 80)

# =====================================================================
# VERİ: Önceki analizlerden ve API sorgularından derlenen sonuçlar
# =====================================================================

# --- FlyWire (Analysis 19'dan) ---
fly = {
    'total_neurons': 139255,
    'total_synapses': 50_000_000,
    'inhibitory_pct': 20.0,  # GABA
    'da_reward': 307,        # PAM
    'da_punishment': 16,     # PPL1
    'da_ratio': 19.2,        # PAM/PPL1
    'mbon_approach': 52,     # ACh MBON
    'mbon_avoidance': 25,    # Glut MBON
    'mbon_suppress': 19,     # GABA MBON
    'mbon_to_pam': 5480,     # sinaps
    'mbon_to_ppl1': 3195,    # sinaps
    'feedback_bias': 1.7,    # PAM lehine
    'avoidance_mbon_to_pam': 2789,
    'avoidance_mbon_to_ppl1': 870,
    'avoidance_bias': 3.2,   # PAM lehine
    'self_inhibition': 19.7, # GABA→GABA %
    'gaba_no_gaba_motor': 4850, # motor nöron artışı %
    'lh_activation_step': 2,
    'mb_activation_step': 3,
    'desc_from_both': 34,    # entegrasyon nöronları
    'silence_approach_effect': -22, # %
    'silence_avoidance_effect': -5, # %
}

# --- MICrONS (CAVEclient sorgularından) ---
microns = {
    'total_neurons': 54887,
    'total_synapses': 523_000_000,
    'exc_neurons': 49082,
    'inh_neurons': 5806,
    'inhibitory_pct': 10.6,
    'ii_pct': 9.5,            # I→I
    'ie_pct': 90.5,           # I→E
    'ei_pct': 21.4,           # E→I
    'ee_pct': 78.6,           # E→E
    'bc_ii': 8.7,             # Basket cell I→I %
    'mc_ii': 9.4,             # Martinotti I→I %
    'bpc_ii': 33.7,           # Bipolar cell I→I %
    'ngc_ii': 13.3,           # Neurogliaform I→I %
    'syn_size_ee': 7559,      # E→E sinaps boyutu
    'syn_size_ie': 4348,      # I→E sinaps boyutu
    'syn_size_ratio': 1.74,   # E/I sinaps boyutu oranı
}

# --- Allen Brain Connectivity (API sorgularından) ---
allen = {
    'vta_to_acb': 231.12,     # VTA → Nucleus Accumbens
    'vta_to_cp': 115.13,      # VTA → Caudoputamen
    'vta_to_dg': 40.56,       # VTA → Dentate Gyrus
    'vta_to_ca1': 6.45,       # VTA → CA1
    'vta_to_pfc': 20.94,      # VTA → PFC toplam
    'vta_to_amyg': 62.05,     # VTA → Amygdala toplam
    'vta_to_visp': 0.15,      # VTA → Visual Cortex
    'snc_to_cp': 100.33,      # SNc → Caudoputamen
    'snc_to_vm': 135.23,      # SNc → Ventromedial thalamus
    'snc_to_amyg': 62.39,     # SNc → Amygdala toplam
    'snc_to_pfc': 1.12,       # SNc → PFC toplam
    'pfc_to_vta': 23.39,      # PL → VTA
    'pfc_to_snc': 13.17,      # PL → SNc
    'pfc_feedback_bias': 1.8, # VTA lehine
}

# --- Hippocampome / Literatür ---
hippo = {
    'n_types_total': 122,     # tüm hipokampüs nöron tipleri
    'ca1_exc_pct': 80,        # piramidal
    'ca1_inh_pct': 15,        # internöron
    'ca1_inh_types': 8,       # farklı inhibitör alt tip
    'd1_effect': 'LTP',       # güçlendirme
    'd2_effect': 'LTD',       # zayıflatma
    'da_on_pv': 'hızlandırır',
    'da_on_sst': 'yavaşlatır',
    'da_on_vip': 'güçlendirir',
}

# =====================================================================
# RAPOR: Markdown
# =====================================================================

report = """# Analysis 25: Türler Arası Karar Devresi Karşılaştırması
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
"""

# Raporu kaydet
report_path = os.path.join(RESULTS_DIR, "25_cross_species_decision.md")
with open(report_path, 'w') as f:
    f.write(report)
print(f"\nRapor kaydedildi: {report_path}")

# =====================================================================
# GÖRSELLEŞTIRME
# =====================================================================
print("\nGörselleştirme hazırlanıyor...")

fig = plt.figure(figsize=(22, 28))
fig.suptitle('Analysis 25: Cross-Species Decision Circuit Comparison\nSinek ↔ Fare ↔ İnsan: 600 Milyon Yıllık Korunmuş Mimari',
             fontsize=16, fontweight='bold', y=0.98)

# --- Panel 1: İnhibitör Oranlar ---
ax1 = fig.add_subplot(4, 3, 1)
species = ['Sinek\n(FlyWire)', 'Fare Korteks\n(MICrONS)', 'Fare Hipokampüs\n(Literatür)', 'İnsan\n(Literatür)']
inh_pcts = [20.0, 10.6, 15.0, 20.0]
colors_inh = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
bars = ax1.bar(species, inh_pcts, color=colors_inh, edgecolor='black', linewidth=0.5)
ax1.set_ylabel('İnhibitör Nöron Oranı (%)')
ax1.set_title('İnhibitör Nöron Oranı\nTürler Arası', fontsize=10, fontweight='bold')
ax1.set_ylim(0, 30)
for bar, val in zip(bars, inh_pcts):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
             f'%{val}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax1.axhline(y=15, color='gray', linestyle='--', alpha=0.5, label='~%15 ortalama')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.2, axis='y')

# --- Panel 2: Dopamin Hedef Karşılaştırma ---
ax2 = fig.add_subplot(4, 3, 2)
targets = ['Motivasyon\n(ACB/γlob)', 'Hareket\n(CP/αβlob)', 'Bellek\n(DG/γlob)', 'Duygu\n(Amyg)', 'Planlama\n(PFC)', 'Görme\n(VISp)']
vta_vals = [231.12, 115.13, 40.56, 62.05, 20.94, 0.15]
snc_vals = [15.69, 100.33, 0.07, 62.39, 1.12, 0.03]
x = np.arange(len(targets))
w = 0.35
bars1 = ax2.bar(x - w/2, vta_vals, w, label='VTA (ödül ≈ PAM)', color='#2ecc71', edgecolor='black', linewidth=0.5)
bars2 = ax2.bar(x + w/2, snc_vals, w, label='SNc (ceza ≈ PPL1)', color='#e74c3c', edgecolor='black', linewidth=0.5)
ax2.set_xticks(x)
ax2.set_xticklabels(targets, fontsize=7)
ax2.set_ylabel('Projection Energy')
ax2.set_title('Dopamin Hedef Bölgeleri\nVTA (ödül) vs SNc (ceza)', fontsize=10, fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.2, axis='y')

# --- Panel 3: İyimserlik Bias ---
ax3 = fig.add_subplot(4, 3, 3)
categories = ['Sinek\nMBON→PAM\nvs PPL1', 'Fare\nPFC→VTA\nvs SNc', 'İnsan\nL-DOPA\niyimserlik']
bias_vals = [1.7, 1.8, 1.0]  # insan için placeholder
bias_colors = ['#e74c3c', '#3498db', '#9b59b6']
bars = ax3.bar(categories, bias_vals, color=bias_colors, edgecolor='black', linewidth=0.5)
ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Eşit (1.0x)')
ax3.set_ylabel('Ödül / Ceza Feedback Oranı')
ax3.set_title('İyimserlik Bias\n(ödül merkezine feedback)', fontsize=10, fontweight='bold')
for bar, val in zip(bars, bias_vals):
    label = f'{val}x' if val > 1 else 'var*'
    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.03,
             label, ha='center', va='bottom', fontsize=10, fontweight='bold')
ax3.set_ylim(0, 2.5)
ax3.legend(fontsize=8)
ax3.text(0.95, 0.95, '*İnsan: Sharot 2012\nL-DOPA kötü haber\ngüncellemeyi azaltır',
         transform=ax3.transAxes, fontsize=7, va='top', ha='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax3.grid(True, alpha=0.2, axis='y')

# --- Panel 4: Self-inhibisyon karşılaştırma ---
ax4 = fig.add_subplot(4, 3, 4)
inh_types = ['Sinek\nGABA', 'Fare BC\n(Basket)', 'Fare MC\n(Martinotti)', 'Fare BPC\n(Bipolar)', 'Fare NGC\n(Neurogliaform)']
ii_vals = [19.7, 8.7, 9.4, 33.7, 13.3]
ii_colors = ['#e74c3c', '#3498db', '#3498db', '#e67e22', '#3498db']
bars = ax4.bar(inh_types, ii_vals, color=ii_colors, edgecolor='black', linewidth=0.5)
ax4.set_ylabel('I→I Oranı (%)')
ax4.set_title('Self-inhibisyon Oranları\n(inhibitör→inhibitör)', fontsize=10, fontweight='bold')
for bar, val in zip(bars, ii_vals):
    ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
             f'%{val}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax4.grid(True, alpha=0.2, axis='y')
# BPC'yi vurgula
ax4.annotate('DİSİNHİBİSYON\n(inhibitörleri\ninhibe eder)', xy=(3, 33.7), xytext=(3.5, 40),
            fontsize=8, fontweight='bold', color='#e67e22',
            arrowprops=dict(arrowstyle='->', color='#e67e22'))

# --- Panel 5: Bağlantı Matrisi (MICrONS) ---
ax5 = fig.add_subplot(4, 3, 5)
conn_matrix = np.array([
    [78.6, 21.4],   # E→E, E→I
    [90.5, 9.5],    # I→E, I→I
])
im = ax5.imshow(conn_matrix, cmap='YlOrRd', aspect='auto')
ax5.set_xticks([0, 1])
ax5.set_xticklabels(['→ Exc', '→ Inh'], fontsize=10)
ax5.set_yticks([0, 1])
ax5.set_yticklabels(['Exc →', 'Inh →'], fontsize=10)
ax5.set_title('MICrONS Bağlantı Matrisi (%)\n(fare görsel korteks)', fontsize=10, fontweight='bold')
for i in range(2):
    for j in range(2):
        ax5.text(j, i, f'%{conn_matrix[i,j]:.1f}', ha='center', va='center',
                fontsize=14, fontweight='bold', color='black' if conn_matrix[i,j] > 50 else 'white')
plt.colorbar(im, ax=ax5, shrink=0.8)

# --- Panel 6: Sinaps Boyutu ---
ax6 = fig.add_subplot(4, 3, 6)
conn_types = ['E→E', 'E→I', 'I→E', 'I→I']
syn_sizes = [7559, 5758, 4348, 4665]
syn_colors = ['#2ecc71', '#f39c12', '#e74c3c', '#8e44ad']
bars = ax6.bar(conn_types, syn_sizes, color=syn_colors, edgecolor='black', linewidth=0.5)
ax6.set_ylabel('Ortalama Sinaps Boyutu')
ax6.set_title('MICrONS Sinaps Boyutları\n(bağlantı tipine göre)', fontsize=10, fontweight='bold')
for bar, val in zip(bars, syn_sizes):
    ax6.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 100,
             str(val), ha='center', va='bottom', fontsize=9, fontweight='bold')
ax6.grid(True, alpha=0.2, axis='y')
ax6.annotate('Excitatory sinapslar\n1.74x daha büyük', xy=(0, 7559), xytext=(1.5, 8500),
            fontsize=8, fontweight='bold', color='#27ae60',
            arrowprops=dict(arrowstyle='->', color='#27ae60'))

# --- Panel 7: VTA Projeksiyon Haritası ---
ax7 = fig.add_subplot(4, 3, 7)
vta_targets_sorted = sorted(zip(
    ['ACB', 'LHA', 'OT', 'CP', 'FS', 'DR', 'DG', 'CEA', 'BLA', 'CA3', 'MD', 'ILA', 'PL'],
    [231.12, 147.62, 129.88, 115.13, 91.07, 62.23, 40.56, 30.78, 19.83, 16.33, 14.98, 10.54, 4.79]
), key=lambda x: x[1])
names, vals = zip(*vta_targets_sorted)
colors_vta = []
for n in names:
    if n in ['ACB', 'CP', 'OT', 'FS']:
        colors_vta.append('#2ecc71')  # striatum
    elif n in ['DG', 'CA3']:
        colors_vta.append('#3498db')  # hippocampus
    elif n in ['BLA', 'CEA']:
        colors_vta.append('#e74c3c')  # amygdala
    elif n in ['PL', 'ILA']:
        colors_vta.append('#9b59b6')  # PFC
    else:
        colors_vta.append('#95a5a6')  # diğer
ax7.barh(names, vals, color=colors_vta, edgecolor='black', linewidth=0.5)
ax7.set_xlabel('Projection Energy')
ax7.set_title('VTA (Ödül Dopamini)\nHedef Bölgeleri', fontsize=10, fontweight='bold')
ax7.grid(True, alpha=0.2, axis='x')
# Legend
patches = [mpatches.Patch(color='#2ecc71', label='Striatum'),
           mpatches.Patch(color='#3498db', label='Hipokampüs'),
           mpatches.Patch(color='#e74c3c', label='Amygdala'),
           mpatches.Patch(color='#9b59b6', label='PFC')]
ax7.legend(handles=patches, fontsize=7, loc='lower right')

# --- Panel 8: İki Hızlı Sistem Şeması ---
ax8 = fig.add_subplot(4, 3, 8)
ax8.set_xlim(0, 10)
ax8.set_ylim(0, 10)
ax8.axis('off')
ax8.set_title('İki Hızlı Sistem\n(korunan mimari)', fontsize=10, fontweight='bold')

# Sinek tarafı
ax8.text(2.5, 9.5, 'SİNEK', ha='center', fontsize=11, fontweight='bold', color='#e74c3c')
boxes_fly = [
    (2.5, 8.2, 'Koku (ORN)', '#ecf0f1'),
    (1.0, 6.5, 'Lateral Horn\n(hızlı, t+2)', '#e74c3c'),
    (4.0, 6.5, 'Mushroom Body\n(yavaş, t+3)', '#3498db'),
    (2.5, 4.8, '34 inen nöron\n(entegrasyon)', '#f39c12'),
    (2.5, 3.2, 'Motor çıktı', '#2ecc71'),
]
for x, y, txt, col in boxes_fly:
    ax8.add_patch(plt.Rectangle((x-1.1, y-0.5), 2.2, 1.0,
                  facecolor=col, edgecolor='black', linewidth=1, alpha=0.7, transform=ax8.transData))
    ax8.text(x, y, txt, ha='center', va='center', fontsize=7, fontweight='bold')

# Oklar
for (x1,y1), (x2,y2) in [((2.5,7.7),(1.0,7.0)), ((2.5,7.7),(4.0,7.0)),
                           ((1.0,6.0),(2.5,5.3)), ((4.0,6.0),(2.5,5.3)),
                           ((2.5,4.3),(2.5,3.7))]:
    ax8.annotate('', xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# Fare tarafı
ax8.text(7.5, 9.5, 'FARE', ha='center', fontsize=11, fontweight='bold', color='#3498db')
boxes_mouse = [
    (7.5, 8.2, 'Duyu girdisi', '#ecf0f1'),
    (6.0, 6.5, 'Amygdala\n(hızlı)', '#e74c3c'),
    (9.0, 6.5, 'Hipokampüs\n+ PFC (yavaş)', '#3498db'),
    (7.5, 4.8, 'Striatum\n(entegrasyon)', '#f39c12'),
    (7.5, 3.2, 'Motor çıktı', '#2ecc71'),
]
for x, y, txt, col in boxes_mouse:
    ax8.add_patch(plt.Rectangle((x-1.1, y-0.5), 2.2, 1.0,
                  facecolor=col, edgecolor='black', linewidth=1, alpha=0.7, transform=ax8.transData))
    ax8.text(x, y, txt, ha='center', va='center', fontsize=7, fontweight='bold')

for (x1,y1), (x2,y2) in [((7.5,7.7),(6.0,7.0)), ((7.5,7.7),(9.0,7.0)),
                           ((6.0,6.0),(7.5,5.3)), ((9.0,6.0),(7.5,5.3)),
                           ((7.5,4.3),(7.5,3.7))]:
    ax8.annotate('', xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# Eşleştirme çizgileri
for y in [6.5, 4.8]:
    ax8.plot([4.5, 5.5], [y, y], 'k--', alpha=0.4, lw=1)
    ax8.text(5.0, y+0.15, '≈', ha='center', fontsize=12, color='gray')

# --- Panel 9: Korunan 6 Prensip Özet ---
ax9 = fig.add_subplot(4, 3, 9)
ax9.axis('off')
ax9.set_title('6 Korunan Prensip\n(özet)', fontsize=10, fontweight='bold')

principles = [
    ('1. Önce güvenlik', 'Tehdit → ödülden önce', '#e74c3c'),
    ('2. İki hızlı sistem', 'Doğuştan + öğrenilmiş', '#3498db'),
    ('3. Karşıt süreç', 'Ödül ↔ ceza dengesi', '#f39c12'),
    ('4. GABA filtreleme', 'Sinyal/gürültü ayrımı', '#2ecc71'),
    ('5. İyimserlik bias', 'Ödüle güçlü feedback', '#9b59b6'),
    ('6. Yaklaşma baskın', 'Kaçınma = fren', '#e67e22'),
]

for i, (title, desc, color) in enumerate(principles):
    y = 0.85 - i * 0.15
    ax9.add_patch(plt.Rectangle((0.05, y-0.04), 0.08, 0.08,
                  facecolor=color, edgecolor='black', linewidth=1,
                  transform=ax9.transAxes))
    ax9.text(0.09, y, '✓', ha='center', va='center', fontsize=12,
            fontweight='bold', color='white', transform=ax9.transAxes)
    ax9.text(0.18, y+0.02, title, fontsize=10, fontweight='bold',
            va='center', transform=ax9.transAxes)
    ax9.text(0.18, y-0.04, desc, fontsize=8, color='gray',
            va='center', transform=ax9.transAxes)

    # Tür ikonları
    for j, (icon, c) in enumerate([('🪰', '#e74c3c'), ('🐭', '#3498db'), ('🧠', '#9b59b6')]):
        ax9.text(0.75 + j*0.1, y, '●', ha='center', va='center',
                fontsize=10, color=c, transform=ax9.transAxes)

ax9.text(0.75, 0.95, 'Sinek', ha='center', fontsize=8, color='#e74c3c',
        fontweight='bold', transform=ax9.transAxes)
ax9.text(0.85, 0.95, 'Fare', ha='center', fontsize=8, color='#3498db',
        fontweight='bold', transform=ax9.transAxes)
ax9.text(0.95, 0.95, 'İnsan', ha='center', fontsize=8, color='#9b59b6',
        fontweight='bold', transform=ax9.transAxes)

# --- Panel 10: Dopamin hipokampüs etkisi ---
ax10 = fig.add_subplot(4, 3, 10)
da_hipp = {
    'DG\n(yeni bellek kapısı)': 40.56,
    'CA3\n(pattern completion)': 16.33,
    'SUB\n(hipokampüs çıkışı)': 9.36,
    'CA1\n(bellek çıktısı)': 6.45,
}
names = list(da_hipp.keys())
vals = list(da_hipp.values())
colors_hipp = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
bars = ax10.bar(names, vals, color=colors_hipp, edgecolor='black', linewidth=0.5)
ax10.set_ylabel('VTA → Projection Energy')
ax10.set_title('VTA Dopamin → Hipokampüs\n(ödül → bellek bağlantısı)', fontsize=10, fontweight='bold')
ax10.grid(True, alpha=0.2, axis='y')
ax10.annotate('Sinekte PAM → γ lobu\naynı prensip', xy=(0, 40.56), xytext=(1.5, 50),
            fontsize=8, fontweight='bold', color='#27ae60',
            arrowprops=dict(arrowstyle='->', color='#27ae60'),
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# --- Panel 11: Feedback döngü karşılaştırma ---
ax11 = fig.add_subplot(4, 3, 11)
fb_labels = ['Sinek\nMBON→PAM', 'Sinek\nMBON→PPL1', 'Fare\nPFC→VTA', 'Fare\nPFC→SNc']
fb_vals = [5480, 3195, 23.39, 13.17]
fb_vals_norm = [5480/3195, 1.0, 23.39/13.17, 1.0]  # normalize
fb_colors = ['#2ecc71', '#e74c3c', '#2ecc71', '#e74c3c']
bars = ax11.bar(fb_labels, fb_vals_norm, color=fb_colors, edgecolor='black', linewidth=0.5)
ax11.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
ax11.set_ylabel('Ödül / Ceza Oranı (normalize)')
ax11.set_title('Feedback Asimetrisi\nÖdül merkezine güçlü geri bildirim', fontsize=10, fontweight='bold')
ax11.text(0, 1.75, '1.7x', ha='center', fontsize=12, fontweight='bold', color='#27ae60')
ax11.text(2, 1.85, '1.8x', ha='center', fontsize=12, fontweight='bold', color='#2980b9')
ax11.grid(True, alpha=0.2, axis='y')

# --- Panel 12: Evrimsel zaman çizelgesi ---
ax12 = fig.add_subplot(4, 3, 12)
ax12.set_xlim(-650, 50)
ax12.set_ylim(0, 10)
ax12.axis('off')
ax12.set_title('Evrimsel Zaman Çizelgesi\n(milyon yıl önce)', fontsize=10, fontweight='bold')

# Zaman çizgisi
ax12.plot([-600, 0], [5, 5], 'k-', lw=2)
events = [
    (-600, 'Ortak ata\n(basit sinir sistemi)', '#95a5a6'),
    (-500, 'Dopamin sistemi\northaya çıkıyor', '#f39c12'),
    (-350, 'Böcekler\nMushroom Body', '#e74c3c'),
    (-200, 'Memeliler\nHipokampüs + Korteks', '#3498db'),
    (-2, 'İnsan\nPrefrontal Korteks', '#9b59b6'),
]
for x, label, color in events:
    ax12.plot([x, x], [4.5, 5.5], color=color, lw=3)
    ax12.plot(x, 5, 'o', color=color, markersize=10, zorder=5)
    y_offset = 7 if events.index((x, label, color)) % 2 == 0 else 2.5
    ax12.text(x, y_offset, label, ha='center', va='center', fontsize=7,
             fontweight='bold', color=color,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.8))
    ax12.plot([x, x], [5.5 if y_offset > 5 else 4.5, y_offset - (0.5 if y_offset > 5 else -0.5)],
             '--', color=color, alpha=0.5, lw=1)

ax12.text(-300, 0.5, '6 karar prensibi bu süreçte korunmuş →', fontsize=9,
         fontstyle='italic', color='gray', ha='center')

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig_path = os.path.join(RESULTS_DIR, "25_cross_species_decision.png")
plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Figür kaydedildi: {fig_path}")

print(f"\n{'='*80}")
print("ANALYSIS 25 COMPLETE")
print(f"  Rapor: {report_path}")
print(f"  Figür: {fig_path}")
print(f"{'='*80}")
