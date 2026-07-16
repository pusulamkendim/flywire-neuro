# FlyWire Neuro — Proje Kapanisi

## Proje Ozeti

Meyve sinegi (Drosophila melanogaster) beyninin tam connectome'u uzerinde norotransmitter analizi, sinyal yayilma simulasyonu ve embodied beyin emulasyonu calismasi. 2024 Ekim — 2026 Mart.

---

## Yapilan Calismalar (Analizler)

### Faz 1: Norotransmitter Haritalama (Analiz 01–11)

| # | Konu | Bulgu |
|---|------|-------|
| 02 | NT dagilimi | ACh %47.9, GABA %21.4, Glut %20.1, DA %5.3, Oct %3.0, Ser %2.3 |
| 03 | Dopamin vs Serotonin | DA: MB odulunde (PAM), SMP motivasyonda. SER: AL koku kazancinda |
| 05 | DA-SER etkilesimi | 8 kopru noron iki sistemi baglıyor, SMP/SIP/SLP'de bulusuyorlar |
| 06 | Odul vs Ceza | PAM (307 noron) → odul, PPL1 (34 noron) → ceza. MB'nin farkli loblarini hedefliyor |
| 07 | Serotonin | Koku isleme kazanc kontrolu — AL'de modulasyon |
| 08 | Oktopamin | Gorme sisteminde arousal modulatoru — optik lobda yogun |
| 09 | GABA | Beyindeki fren sistemi — karar gecikmeleri, lateral inhibisyon |
| 10 | Asetilkolin | Ana uyarici — tum duyusal ve motor devrelerde |
| 11 | Glutamat | Cift ajan — motor planlama + inhibitor islevler |

### Faz 2: Sinyal Yayilma Simulasyonlari (Analiz 12–16)

| # | Konu | Bulgu |
|---|------|-------|
| 12 | Koku sinyal yayilimi | ORN → AL → MB/LH → SMP → motor: 5 katman, ~50ms |
| 13 | Tat sinyal yayilimi | Sugar GRN → SEZ → MN9: 2-3 sinaps, ~30ms |
| 14 | Ogrenme simulasyonu | PAM odul / PPL1 ceza → MB Kenyon hucreleri → MBON cikis |
| 15 | PPL1 temporal oncelik | Ceza sinyali odulden ~2ms once ulasıyor — yapısal avantaj |
| 16 | Parametre duyarlilik | PPL1 onceligi parametreye degil yapiya bagli — robust |

### Faz 3: Embodied Beyin Simulasyonu (Analiz 22–24)

| # | Konu | Bulgu |
|---|------|-------|
| 22 | Olfactory embodied | 138K noron LIF + NeuroMechFly. Scipy sparse 38x hizlanma |
| 23 | Sugar navigation | Koku takibi + goz temizleme + beslenme. Tam davranis zinciri |
| 24 | Autonomous fly | Hunger drive, PAM dopamin monitoring, doyum. **PAM calismiyor** |

---

## Temel Bulgular

### 1. Connectome yapisal olarak tam, fonksiyonel olarak eksik

Sugar GRN → PAM dopamin yolu connectome'da **topolojik olarak var** (2 sinaps, 7 ara noron, 307 PAM'in tamami ulasılabilir). Ancak LIF simulasyonunda **fonksiyonel olarak sessiz** — ne 150 step (22.5ms) ne 500 step (75ms) ne de 1000x agirlik boost'u ile PAM aktive edilemedi.

**Neden:** Noromodulasyon (serotonin, oktopamin) olmadan sinaptik agirliklar ara noronlari atesletmeye yetmiyor. Connectome kablolamayi veriyor ama kablolarin "voltaj ayarini" yapan kimyasal sistemi vermiyor.

### 2. "A connectome is not enough" — literaturle uyumlu

Shiu (2024, Nature) ayni modelde JO-F → aBN1 yolunun 78 sinapsa ragmen fonksiyonel olarak sessiz oldugunu gosterdi. Bizim sugar → PAM bulgumuz bunu tamamliyor ve genisletiyor.

Eksik parcalar (literaturden):
- Noromodulasyon (dopamin, serotonin dinamik kazanc kontrolu)
- Noropeptidler (100+ tip, yavas/uzun menzilli sinyal)
- Gap junction'lar (elektriksel sinapslar, connectome'da yok)
- Reseptor cesitliligi (ayni NT farkli etkiler)
- Internal state (aclik, uyku, cinsel motivasyon)

### 3. Hicbir proje tam otonom davranis uretemedi

| Proje | Otonom mu? | Gercek |
|-------|-----------|--------|
| Shiu 2024 | Hayir | Tek uyaran → tek motor cikti |
| Eon/Rojas | Hayir | Hand-mapped motor komutlar, gorme "dekoratif" |
| OpenWorm | Hayir | Sinaptik agirliklar bilinmiyor, ML ile egitilmis |
| NeuroSimWorm | Kismen | Fitness function ile optimize edilmis |
| **Bizim proje** | Kismen | Sugar→MN9 beyin-kaynakli, geri kalan programci-kaynakli |

### 4. Scipy sparse optimizasyonu ozgun katki

Torch CSR'dan scipy sparse'a gecisle **38x hizlanma** saglandi. Bu, GPU olmadan laptop'ta 138K noron simulasyonunu mumkun kildi. Eon GPU gerektiriyordu.

---

## Kullanilan Araclar ve Repolar

### Veri Kaynaklari

| Repo/Kaynak | Ekip | Katki |
|-------------|------|-------|
| [FlyWire v783](https://flywire.ai/) | Dorkenwald et al. | 139,255 noron, 50M+ sinaps connectome |
| [flywire_annotations](https://github.com/flyconnectome/flywire_annotations) | Schlegel, Cambridge | 8,453 hucre tipi, NT etiketleri, soy hatti |
| [synister](https://github.com/funkelab/synister) | Funke Lab, Janelia | EM goruntuden NT tahmini CNN (%87 sinaps, %94 noron) |
| [drosophila_neurotransmitters](https://github.com/funkelab/drosophila_neurotransmitters) | Bates, Funke | 900+ hucre tipi icin deneysel NT ground truth |

### Ag Analizi

| Repo | Ekip | Katki |
|------|------|-------|
| [flywire-network-analysis](https://github.com/murthylab/flywire-network-analysis) | Princeton (Lin, Yang) | Derece dagilimi, rich-club, motif, projectome |
| [flybrain-clustering](https://josiclab.github.io/flybrain-clustering/) | Josic Lab | Hiyerarsik moduler yapi, stochastic blockmodel |
| [ol_annotations](https://github.com/flyconnectome/ol_annotations) | Cambridge | Optik lob disi-erkek karsılasmasi |
| [male-visual-connectome](https://github.com/reiserlab/male-drosophila-visual-system-connectome-code) | Reiser Lab, Janelia | Erkek gorme sistemi envantari (Nature 2025) |

### Simulasyon

| Repo | Ekip | Katki |
|------|------|-------|
| [Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model) | Phil Shiu | LIF model, Brian 2. Sugar→MN9, JO→aDN1. 122 star (en populer) |
| [fly-brain](https://github.com/erojasoficial-byte/fly-brain) | Rojas (Eon) | Embodied simulasyon, Hebbian plastisite, "bilinc metrikleri" |
| [larval-connectome](https://github.com/brain-networks/larval-drosophila-connectome) | Betzel et al. | Larva beyni (3,016 noron) — kucuk olcekli test icin |

### Biomekanik

| Arac | Kullanim |
|------|----------|
| [NeuroMechFly v2](https://neuromechfly.org/) | 6 bacakli sinek vucudu, MuJoCo fizik motoru |
| [flygym](https://github.com/NeLy-EPFL/flygym) | CPG lokomotor, OdorArena, kamera sistemi |

---

## Proje Ciktilari

### Kod
- `01_explore.py` — `16_sensitivity_analysis.py`: 16 analiz scripti
- `22_olfactory_embodied.py`: Embodied koku simulasyonu
- `23_sugar_navigation.py`: Seker navigasyonu + grooming + beslenme
- `24_autonomous_fly.py`: Otonom sinek (hunger drive + PAM monitoring)
- `fly-brain-embodied/`: Moduler sensorimotorsistem (olfactory.py, gustatory.py, brain_body_bridge.py)

### Gorseller ve Videolar
- `results/01_pie_overall.png` — `results/11_reward_punishment_diagram.png`: NT analiz grafikleri
- `results/22_olfactory_embodied.mp4/.png`: Embodied koku simulasyonu
- `results/23_sugar_navigation.mp4/.png`: Seker navigasyonu
- `results/24_autonomous_fly.mp4/.png`: Otonom sinek
- `results/24_dopamine_levels.md`: PAM dopamin seviye analizi

### Dokumantasyon
- `BENIOKU.md`: Proje yapisi, veri kaynaklari, repolar
- `results/*.md`: Her analiz icin detayli bulgular (02–16)

---

## Acik Sorular ve Gelecek Calismalar

1. **Noromodulasyon entegrasyonu**: Sugar → PAM yolunu calistirmak icin serotonin/oktopamin dinamik kazanc kontrolu eklenmeli. Hicbir mevcut proje bunu basarmis degil.

2. **Effectome yaklasimi**: Bhatt et al. (2024, Nature) optogenetik perturbasyon ile gercek kausal etkileri olcmeyi oneriyor. Connectome'u prior olarak kullanip agirliklari veriden ogrenmek.

3. **Connectome-constrained RNN**: Turner et al. (2024, Nature) connectome yapisini sinirlandirma olarak kullanip agirliklari egiterek fonksiyonel model olusturma.

4. **Noropeptid connectome**: Ripoll-Sanchez et al. (2023, Neuron) C. elegans'ta yapti. Drosophila icin henuz yok — internal state modellemesi icin kritik.

5. **Gap junction haritalama**: Drosophila beyni icin elektriksel sinaps haritasi henuz cikarilmamis. "The missing piece of the connectome" (Current Biology, 2023).

---

## Sonuc

Bu proje Drosophila connectome'unu sadece yapisal olarak degil, fonksiyonel olarak anlamaya calisti. 6 norotransmitter sistemini haritaladik, sinyal yayilimini simule ettik, 138K noronlu beyni fiziksel bir vucutla birlestirdik.

En onemli bulgu: **connectome gerekli ama yeterli degil**. Kablolama dogru, yollar var, ama bu yollari acip kapatan kimyasal sistem (noromodulasyon) olmadan tam otonom davranis mumkun degil. Bu, sadece bizim degil, alanin tamaminin karsilastigi temel sinir.

Tam otonom beyin simulasyonu hala acik bir problem. Cozumu muhtemelen connectome + noromodulasyon + ogrenme kombinasyonu olacak. Bu proje, o yolda nerede oldugumuzun net bir haritasini cikartti.
