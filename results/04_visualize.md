# Analiz 04 - Nörotransmitter Görselleştirmeleri

Script: `04_visualize.py`

---

## Üretilen Grafikler

### 01_pie_overall.png — Genel Nörotransmitter Dağılımı
54.5 milyon sinaps üzerinden ağırlıklı pasta grafik.
- Asetilkolin %48.4 ile açık ara baskın
- GABA %23.2 ile ikinci (fren sistemi)
- Glutamat %19.3
- Dopamin, Serotonin, Oktopamin toplamda ~%9

### 02_bar_regions.png — Bölge Bazında Dağılım (Yığılmış Bar)
En büyük 15 beyin bölgesinin NT karşılaştırması. Üstteki sayılar sinaps miktarını gösteriyor.
- Tüm bölgelerde Asetilkolin (yeşil) baskın
- **FB (Fan-shaped Body)** en düşük Asetilkolin oranına sahip (%36.7), Glutamat ve Dopamin daha belirgin
- **AL (Antennal Lobe)** bölgesinde Serotonin (mor) diğer bölgelere göre gözle görülür şekilde yüksek
- **LOP** bölgelerinde Oktopamin (turuncu) diğerlerine göre daha belirgin

### 03_da_vs_ser_targets.png — Dopamin vs Serotonin Hedef Bölgeleri
Yan yana yatay bar chart. Her bölgede dopamin (turuncu) vs serotonin (mor) sinaps sayısı.
- **AL (Antennal Lobe)** — Serotonin açık ara baskın (~50K sinaps vs ~7K dopamin)
- **MB_ML (Mushroom Body)** — Dopamin açık ara baskın (~35K vs ~800 serotonin)
- **GNG** — Serotonin baskın ama dopamin de güçlü
- **FB** — İkisi birbirine yakın

### 04_da_ser_neuron_types.png — Nöron Süper Sınıf Karşılaştırması
Yan yana pasta grafik.
- **Dopamin**: %96.5 merkezi beyin (central) — tek renk neredeyse
- **Serotonin**: %68.1 duyusal (sensory) — çok daha dağınık bir profil
- Bu iki sistem temelden farklı: biri "içeride karar veriyor", diğeri "dışarıdan gelen bilgiyi düzenliyor"

### 05_da_ser_cell_classes.png — Hücre Sınıfları Karşılaştırması
Yan yana yatay bar chart.
- **Dopamin**: Kenyon_Cell (5,172) diğer tüm sınıfları eziyor. DAN (330) ikinci.
- **Serotonin**: Daha dengeli dağılmış — olfactory (650), visual (638), AN (171), CX (147)...
- Dopamin tek bir merkeze odaklı, serotonin geniş bir duyusal ağa yayılmış

### 06_da_ser_dominance.png — Bölge Baskınlık Haritası
Tüm beyin bölgelerinde dopamin/serotonin oranı (renk gradyanı ile).
- **Kırmızı/koyu** = Dopamin baskın bölgeler (MB, LO, CRE, AOTU...)
- **Yeşil** = Serotonin baskın bölgeler (AL, AMMC, PRW, FLA, GNG...)
- **Sarı** = Dengeli bölgeler (FB, SMP, ICL, SLP...)

Önemli gözlemler:
- Mushroom Body (MB) lobları en uçta dopamin-baskın (%97-98)
- Antennal Lobe (AL) en uçta serotonin-baskın (%87-89)
- OCG bölgesinde serotonin neredeyse %100 baskın
- Bölgelerin çoğunluğu dopamin-baskın (çizginin sağında)

---

## Özet Yorum

Bu grafikler birlikte şu hikayeyi anlatıyor:

1. **Beynin ana dili Asetilkolin** — neredeyse her bölgede yarıdan fazla
2. **Dopamin ve Serotonin nadir ama uzmanlaşmış sistemler**
3. **Dopamin = İç dünya**: Hafıza merkezine (Mushroom Body) odaklı, Kenyon hücrelerinden çıkıyor
4. **Serotonin = Dış dünya**: Koku ve tat merkezlerine (AL, GNG) odaklı, duyusal nöronlardan çıkıyor
5. **İki sistem neredeyse hiç örtüşmüyor** — farklı bölgelerde baskınlar, farklı hücre tiplerinden geliyorlar
