# FlyWire Neuro - Meyve Sineği Beyin Connectome Analizi

Meyve sineği (*Drosophila melanogaster*) beyninin tam sinaptik bağlantı haritası (connectome) üzerinde nörotransmitter analizi projesi.

## Proje Hakkında

FlyWire projesi, Ekim 2024'te Nature'da yayınlanan, bir yetişkin dişi meyve sineğinin beyninin tamamının sinaps düzeyinde haritalanmasıdır. 139.255 nöron ve 50 milyondan fazla sinaptik bağlantı içerir.

---

## Veri Kaynakları

### Zenodo - Ham Veri Dosyaları

**FlyWire Connectome v783**: https://zenodo.org/records/10676866

| Dosya | Boyut | Format | Açıklama |
|-------|-------|--------|----------|
| `flywire_synapses_783.feather` | 8.8 GB | Feather | ~130 milyon sinaps: 3D koordinatları (nanometre), nörotransmitter tahminleri, pre/post-sinaptik nöron ID'leri |
| `proofread_connections_783.feather` | 812.6 MB | Feather | Nöron-nöron bağlantı çiftleri, sinaps sayıları, beyin bölgesine göre nörotransmitter olasılıkları |
| `per_neuron_neuropil_count_post_783.feather` | 223 MB | Feather | Post-sinaptik sayımlar (nöron + beyin bölgesi bazında) |
| `per_neuron_neuropil_count_pre_783.feather` | 16.1 MB | Feather | Pre-sinaptik sayımlar (nöron + beyin bölgesi bazında) |
| `proofread_root_ids_783.npy` | 1.1 MB | NumPy | Doğrulanmış tüm nöron ID'lerinin listesi |

**Network Analysis Data**: https://zenodo.org/records/12572930
- Ağ istatistikleri veri dışa aktarımları ve analiz scriptleri
- Nörotransmitter sınıflandırmasına göre gruplandırılmış analizler

### İndirilen Veri

#### `data/proofread_connections_783.feather` — 16,847,997 satır, 10 sütun

| Sütun | Tip | Açıklama |
|-------|-----|----------|
| `pre_pt_root_id` | int64 | Sinyal gönderen (pre-sinaptik) nöron ID |
| `post_pt_root_id` | int64 | Sinyal alan (post-sinaptik) nöron ID |
| `neuropil` | string | Beyin bölgesi (ör. AVLP_R, SLP_R, SMP_R) |
| `syn_count` | int64 | Sinaps sayısı (bağlantı gücü) |
| `gaba_avg` | float64 | GABA olasılığı (inhibitör nörotransmitter) |
| `ach_avg` | float64 | Asetilkolin olasılığı (uyarıcı nörotransmitter) |
| `glut_avg` | float64 | Glutamat olasılığı |
| `oct_avg` | float64 | Oktopamin olasılığı |
| `ser_avg` | float64 | Serotonin olasılığı |
| `da_avg` | float64 | Dopamin olasılığı |

#### `data/neuron_annotations.tsv` — 139,244 satır, 31 sütun

| Sütun | Benzersiz Değer | Açıklama |
|-------|----------------|----------|
| `root_id` | 139,244 | Nöron ID (bağlantı verisiyle eşleştirme anahtarı) |
| `pos_x/y/z` | — | Nöron pozisyonu (3D koordinat) |
| `soma_x/y/z` | — | Soma (hücre gövdesi) pozisyonu |
| `flow` | 3 | intrinsic / sensory / motor |
| `super_class` | 10 | optic, central, sensory, visual_projection, ascending, descending, motor, endocrine... |
| `cell_class` | 49 | ME>LO, Kenyon_Cell, CX, olfactory, mechanosensory... |
| `cell_sub_class` | 100 | Alt sınıf detayı |
| `cell_type` | 8,806 | Detaylı hücre tipi |
| `top_nt` | 6 | Tahmin edilen nörotransmitter (acetylcholine, GABA, glutamate, dopamine, serotonin, octopamine) |
| `top_nt_conf` | — | Nörotransmitter tahmini güven skoru |
| `known_nt` | 156 | Deneysel olarak doğrulanmış nörotransmitter |
| `known_nt_source` | 155 | Doğrulama kaynağı (ör. Zhao et al., 2023 FISH) |
| `side` | 4 | left / right / center |
| `hemibrain_type` | 4,189 | Hemibrain karşılığı hücre tipi |
| `ito_lee_hemilineage` | 214 | Soy hattı (Ito-Lee sınıflandırması) |
| `dimorphism` | 3 | Cinsiyet dimorfizmi (isomorphic, dimorphic...) |

#### `data/hemibrain_meta.csv` — 2.5 MB
Hemibrain connectome metadata (neuPrint'ten).

### FlyWire Veri Setleri (Codex)

| Dataset | Açıklama | Nöron | Bağlantı |
|---------|----------|-------|----------|
| FAFB (v783) | Dişi yetişkin sinek beyni | 139,255 | 3,732,460 |
| BANC (v626) | Dişi beyin + sinir kordonu | 115,151 | 2,676,592 |
| MANC (v1.2.1) | Erkek sinir kordonu | 23,665 | 5,305,638 |
| MAOL (v1.1) | Erkek sağ optik lob (görme merkezi) | 52,445 | 6,484,936 |
| MCNS (v0.9) | Erkek merkezi sinir sistemi | 166,694 | 6,239,112 |

Tüm datasetler meyve sineği (*Drosophila melanogaster*) üzerinedir.

---

## GitHub Repoları

### Analiz ve Araştırma

**[murthylab/flywire-network-analysis](https://github.com/murthylab/flywire-network-analysis)** — `flywire-network-analysis/` olarak klonlandı
Princeton ekibi, Nature 2024. Jupyter notebook + MATLAB scriptleri:
- Derece dağılımları, hub nöronlar (integrator vs broadcaster)
- Rich-club analizi, küçük dünya (small-world) özelliği
- Nörotransmitter bazlı projectome — her beyin bölgesinde hangi transmitter baskın
- Motif analizi, spektral analiz, bölgeler arası bağlantı matrisleri

**[flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations)** — anotasyon verileri `data/` altında
Cambridge ekibi (Schlegel et al., Nature 2024). 8,453 hücre tipi, soy hattı, nörotransmitter etiketleri.

**[flyconnectome/ol_annotations](https://github.com/flyconnectome/ol_annotations)**
Optik lob (görme merkezi) nöron anotasyonları. Dişi vs Erkek karşılaştırması.

**[josiclab/flybrain-clustering](https://josiclab.github.io/flybrain-clustering/)**
Hiyerarşik modüler yapı analizi. Stochastic blockmodel ile topluluk tespiti. Topluluklar nörotransmitter tipiyle korelasyon gösteriyor.

**[reiserlab/male-drosophila-visual-system-connectome-code](https://github.com/reiserlab/male-drosophila-visual-system-connectome-code)**
Erkek görme sistemi connectome analizi (Nern et al. 2025, Nature). Görme nöronlarının nörotransmitter kimlikleri.

**[brain-networks/larval-drosophila-connectome](https://github.com/brain-networks/larval-drosophila-connectome)**
Larva Drosophila connectome (3,016 nöron, 548K sinaps). Daha küçük, tam anotasyonlu — analiz yöntemlerini test etmek için ideal.

### Nörotransmitter Tahmin Araçları

**[funkelab/synister](https://github.com/funkelab/synister)**
Elektron mikroskobu görüntülerinden nörotransmitter tahmini yapan CNN modeli. 6 tip: ACh, GABA, Glut, Ser, Dop, Oct. Sinaps başına %87, nöron başına %94 doğruluk. Bizim verideki tahminlerin kaynağı bu model (Eckstein, Bates et al., Cell 2024).

**[funkelab/drosophila_neurotransmitters](https://github.com/funkelab/drosophila_neurotransmitters)**
Deneysel olarak doğrulanmış nörotransmitter ground truth verisi (CSV). Tahminleri doğrulamak için referans.

### Simülasyon

**[erojasoficial-byte/fly-brain](https://github.com/erojasoficial-byte/fly-brain)**
138,639 nöronun tamamını spiking model olarak simüle. FlyWire v783 + NeuroMechFly v2 (MuJoCo). Nörotransmitter-spesifik sinaptik entegrasyon.

**[philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model)**
Leaky integrate-and-fire hesaplamalı model. Brian 2 simülatörü ile laptop üzerinde çalıştırılabilir.

### Python Kütüphaneleri

**[navis-org/navis](https://github.com/navis-org/navis)**
Nöroanatomik veri analizi ve görselleştirme. Morfoloji ölçümleri, akson/dendrit ayırma, NBLAST kümeleme, 2D/3D görselleştirme (matplotlib, plotly, Blender). `pip install navis`

**[navis-org/fafbseg-py](https://github.com/navis-org/fafbseg-py)**
FlyWire connectome ile çalışmak için araçlar. navis ile tam uyumlu. İskelet ve mesh çekme, morfoloji analizi. `pip install fafbseg`

**[seung-lab/CAVEclient](https://github.com/seung-lab/CAVEclient)**
CAVE API Python istemcisi. Gerçek zamanlı veritabanı erişimi. `pip install caveclient`

**[seung-lab/FlyConnectome](https://github.com/seung-lab/FlyConnectome)**
Programatik erişim tutorial'ları.

### R Paketleri

**[natverse/coconatfly](https://github.com/natverse/coconatfly)** — Karşılaştırmalı connectomics (hemibrain vs FlyWire)
**[natverse/hemibrainr](https://github.com/natverse/hemibrainr)** — Hemibrain veri analizi
**[natverse/fafbseg](https://github.com/natverse/fafbseg)** — FAFB-FlyWire R arayüzü

---

## Online Erişim Araçları

| Araç | URL | Açıklama |
|------|-----|----------|
| Codex | https://codex.flywire.ai/ | Ana keşif aracı, arama, 3D görselleştirme, CSV/JSON export |
| Codex Sample Queries | https://codex.flywire.ai/sample_queries | Örnek sorgular (nt_type == GABA, vs.) |
| Codex API | https://codex.flywire.ai/api/download | Programatik veri indirme |
| FlyWire Editor | https://edit.flywire.ai/ | 3D nöron inceleme |
| NeuroNLP | https://flywire.neuronlp.fruitflybrain.org/ | Doğal dille sorgulama |
| CATMAID | https://fafb-flywire.catmaid.org/ | İnteraktif connectome verisi |
| Virtual Fly Brain | https://www.virtualflybrain.org/ | 200K+ kayıtlı görüntü, gen ekspresyon entegrasyonu |
| neuPrint | https://neuprint.janelia.org | Hemibrain/MANC Neo4j sorguları |
| Nature Interaktif | https://www.nature.com/immersive/d42859-024-00053-4/index.html | Görsel sunum |

---

## Nörotransmitter Referansı

| Kısaltma | Nörotransmitter | Verideki Ort. | Rol |
|----------|-----------------|---------------|-----|
| `ach` | Asetilkolin | 0.479 | Uyarıcı (excitatory) — en yaygın |
| `gaba` | GABA | 0.214 | İnhibitör (inhibitory) |
| `glut` | Glutamat | 0.201 | Uyarıcı/İnhibitör |
| `da` | Dopamin | 0.053 | Ödül, motivasyon, öğrenme |
| `oct` | Oktopamin | 0.030 | Savaş-kaç tepkisi (böceklerde adrenalin benzeri) |
| `ser` | Serotonin | 0.023 | Ruh hali, uyku, iştah düzenleme |

Tahminler **Synister** CNN modeli ile sinaps EM görüntülerinden yapılmıştır. Sinaps başına doğruluk: %87, nöron başına: %94 (Eckstein, Bates et al., Cell 2024).

---

## Süper Sınıf Dağılımı (neuron_annotations.tsv)

| Süper Sınıf | Nöron Sayısı | Açıklama |
|-------------|-------------|----------|
| optic | 77,539 | Görme sistemi nöronları |
| central | 32,384 | Merkezi beyin nöronları |
| sensory | 16,904 | Duyusal nöronlar |
| visual_projection | 8,038 | Görsel projeksiyon |
| ascending | 1,750 | Yükselen nöronlar |
| descending | 1,303 | İnen nöronlar |
| sensory_ascending | 612 | Duyusal yükselen |
| visual_centrifugal | 524 | Görsel santrifügal |
| motor | 110 | Motor nöronlar |
| endocrine | 80 | Hormonal nöronlar |

---

## Kurulum

```bash
cd ~/projects/flywire-neuro
python3 -m venv venv
source venv/bin/activate
pip install caveclient pandas pyarrow navis fafbseg
```

## Proje Yapısı

```
flywire-neuro/
├── data/
│   ├── proofread_connections_783.feather   (813 MB - 16.8M bağlantı)
│   ├── neuron_annotations.tsv             (31 MB - 139K nöron etiketi)
│   └── hemibrain_meta.csv                 (2.5 MB)
├── results/
│   ├── 01_pie_overall.png                 (Genel NT dağılımı pasta grafik)
│   ├── 02_bar_regions.png                 (Bölge bazında NT yığılmış bar)
│   ├── 03_da_vs_ser_targets.png           (DA vs SER hedef bölgeleri)
│   ├── 04_da_ser_neuron_types.png         (Nöron süper sınıf karşılaştırması)
│   ├── 05_da_ser_cell_classes.png         (Hücre sınıfları karşılaştırması)
│   ├── 06_da_ser_dominance.png            (Bölge baskınlık haritası)
│   ├── 07_da_ser_interaction.png          (DA↔SER sinaps matrisi + buluşma bölgeleri)
│   ├── 08_bridge_neurons.png              (Köprü nöronların profili)
│   ├── 09_interaction_diagram.png         (Etkileşim özet şeması)
│   ├── 10_reward_vs_punishment.png        (PAM vs PPL1 MB bölgeleri)
│   ├── 11_reward_punishment_diagram.png   (Ödül-ceza devresi şeması)
│   ├── 02_nt_distribution.md              (Nörotransmitter dağılım sonuçları)
│   ├── 03_dopamine_serotonin.md           (Dopamin/Serotonin analiz sonuçları)
│   ├── 04_visualize.md                    (Görselleştirme açıklamaları)
│   ├── 05_da_ser_interaction.md           (DA↔SER etkileşim analiz sonuçları)
│   ├── 06_reward_vs_punishment.md        (PAM ödül vs PPL1 ceza analiz sonuçları)
│   ├── 07_serotonin_role.md              (Serotoninin koku işlemedeki rolü)
│   ├── 08_octopamine.md                 (Oktopamin — görme sistemi adrenalini)
│   ├── 09_gaba_inhibition.md            (GABA inhibisyon — beyindeki fren sistemi)
│   ├── 10_acetylcholine.md              (Asetilkolin — beyindeki elektrik şebekesi)
│   ├── 11_glutamate.md                  (Glutamat — motor planlamacı ve çift ajan)
│   ├── 12_signal_propagation.md         (Sinyal yayılma simülasyonu — koku geldi ne oldu?)
│   ├── 13_taste_propagation.md          (Tat sinyal yayılma — sinek bir şey tattı)
│   ├── 14_learning_simulation.md        (Ödül/ceza öğrenme simülasyonu)
│   ├── 15_ppl1_priority.md              (PPL1 temporal öncelik — ceza neden önce?)
│   └── 16_sensitivity_analysis.md       (Parametre duyarlılık — PPL1 yapısal mı?)
├── flywire-network-analysis/              (murthylab repo - hazır analizler)
│   ├── python_scripts/                    (Jupyter notebook'lar)
│   ├── matlab_scripts/                    (MATLAB analizleri)
│   ├── v630_data/                         (v630 snapshot verisi)
│   └── data_products/
├── venv/
├── 00_setup_token.py                      (CAVEclient token kurulumu)
├── 01_explore.py                          (Datastack keşfi - CAVEclient erişim izni gerekli)
├── 02_nt_distribution.py                  (Beyin bölgelerinde nörotransmitter dağılımı)
├── 03_dopamine_serotonin.py               (Dopamin/Serotonin odaklı analiz)
├── 04_visualize.py                        (6 grafik üreten görselleştirme scripti)
├── 05_da_ser_interaction.py               (DA↔SER etkileşim + köprü nöron analizi)
├── 06_reward_vs_punishment.py             (PAM ödül vs PPL1 ceza devresi analizi)
├── 07_serotonin_role.py                   (Serotoninin koku işlemedeki rolü)
├── 08_octopamine.py                       (Oktopamin analizi — görme modülatörü)
├── 09_gaba_inhibition.py                  (GABA inhibisyon analizi — fren sistemi)
├── 10_acetylcholine.py                    (Asetilkolin analizi — ana uyarıcı)
├── 11_glutamate.py                        (Glutamat analizi — çift rolü)
├── 12_signal_propagation.py               (Sinyal yayılma simülasyonu — koku)
├── 13_taste_propagation.py                (Sinyal yayılma simülasyonu — tat)
├── 14_learning_simulation.py              (Ödül/ceza öğrenme simülasyonu)
├── 15_ppl1_priority.py                    (PPL1 temporal öncelik analizi)
├── 16_sensitivity_analysis.py             (Parametre duyarlılık analizi)
└── README.md
```

## İlgili Yayınlar

- Dorkenwald et al. (2024) "Neuronal wiring diagram of an adult brain" — Nature
- Schlegel et al. (2024) "Whole-brain annotation and multi-connectome cell typing" — Nature
- Lin, Yang et al. (2024) "Network Statistics of the Whole-Brain Connectome" — Nature
- Eckstein, Bates et al. (2024) "Neurotransmitter classification from EM images" — Cell
- Winding et al. (2023) "The connectome of an insect brain" — Science
- Nern et al. (2025) "Connectome-driven neural inventory of a complete visual system" — Nature


 FlyWire / Drosophila Connectome -- Repository Research Report

       ---
       1. murthylab/flywire-network-analysis

       - Team: Princeton (Albert Lin, Runzhe Yang, Sven Dorkenwald)
       - What they built: Jupyter/MATLAB notebooks for whole-brain network analysis -- degree distributions, motif frequencies, rich-club organization, spectral
       properties, random walks, reciprocal connectivity, neuropil projectome.
       - Key findings: Identified fundamental organizational principles of a complete brain connectome including rich-club topology, characteristic motif
       distributions, and reciprocal wiring patterns. Uses FlyWire v630 snapshot.
       - Paper: Lin, Yang et al. 2023 + Dorkenwald et al. 2023 (Nature 2024)
       - Status: Last commit Feb 2024. 7 stars. MIT license. Stable/complete.
       - Limitations: Analysis only, no simulation. Tied to v630 (not latest v783). Small community engagement (7 stars, 1 fork).

       ---
       2. flyconnectome/flywire_annotations

       - Team: Cambridge (Philipp Schlegel, Gregory Jefferis, Yijie Yin, Marta Costa)
       - What they built: Systematic neuron annotations for the FlyWire female adult brain (release 783) -- cell types, morphology, neurotransmitter labels, synapse
       tables, NBLAST similarity scores, cross-connectome mapping to hemibrain.
       - Key findings: Whole-brain annotation and multi-connectome cell typing enabling cross-dataset comparison. Most comprehensive annotation resource for the adult
        fly brain.
       - Paper: Schlegel et al. 2024 (Nature); also referenced by Berg et al. 2025, Matsliah et al. 2024, Dorkenwald et al. 2024.
       - Status: Actively maintained. Last update Oct 2025 (v3.0.0). 40 stars.
       - Limitations: Annotation-only (no dynamics/simulation). Primarily female brain; male annotations handled elsewhere.

       ---
       3. flyconnectome/ol_annotations

       - Team: Cambridge / flyconnectome (linked to Nern et al.)
       - What they built: Matching tables that map optic lobe cell types between FAFB-FlyWire (female) and Male CNS datasets, using Nern et al. cell types as
       reference.
       - Key findings: Great majority of cross-sex cell type matches are 1-to-1, but some are more complex, enabling comparative sexual dimorphism analysis in the
       visual system.
       - Paper: Nern et al. 2024 (bioRxiv preprint 2024.04.16.589741)
       - Status: Last commit May 2024. 2 stars. R-based. Versioned releases.
       - Limitations: Narrow scope (optic lobe only). Very small community. R-only codebase.

       ---
       4. josiclab/flybrain-clustering

       - Team: Josic Lab (Alex Kunin et al.)
       - What they built: Hierarchical clustering analysis of the hemibrain and larval mushroom body connectomes using generalized modularity density maximization.
       - Key findings: Demonstrated modular organization in fly brain networks with distinct clustering patterns across brain regions (mushroom body, fan-shaped
       body). Cell type-specific wiring follows modular structure.
       - Paper: "Clustering analysis of the Fly Hemibrain data set" (bioRxiv 2022)
       - Status: Last commit May 2023. 2 stars, 3 forks. Stable/complete.
       - Limitations: Based on hemibrain (partial brain), not the full FlyWire connectome. Preprint only (not peer-reviewed as of last check). Stale -- no updates in
       ~3 years.

       ---
       5. reiserlab/male-drosophila-visual-system-connectome-code

       - Team: Reiser Lab, Janelia (Nern et al., 14 contributors)
       - What they built: Analysis code for the male optic lobe connectome from Janelia's FlyEM/neuPrint database. Jupyter notebooks, Python scripts, Blender
       visualizations, Snakemake pipelines.
       - Key findings: Connectome-driven neural inventory of a complete visual system. Companion "Cell type explorer" repo provides interactive exploration.
       - Paper: Nern et al. 2025 (Nature, doi:10.1038/s41586-025-08746-0)
       - Status: Active. v1.0 released March 2025. 9 stars, 7 forks.
       - Limitations: Male visual system only (not whole brain). Analysis code, not simulation.

       ---
       6. brain-networks/larval-drosophila-connectome

       - Team: Betzel, Puxeddu, Seguin (brain-networks org)
       - What they built: Supplementary connectome data from the larval Drosophila brain (originally from Winding et al. 2023, Science) with hierarchical community
       analysis.
       - Key findings: Identified hierarchical communities in the larval connectome and linked them to cellular annotations and network topology.
       - Paper: Betzel et al. 2023 (bioRxiv preprint); data from Winding et al. 2023 (Science).
       - Status: Minimal. 0 stars, 0 forks, only 3 commits. Data archive only.
       - Limitations: Larval brain (very different from adult). Essentially a data deposit, not an analysis toolkit. Zero community adoption. Preprint only.

       ---
       7. funkelab/synister

       - Team: Funke Lab, Janelia Research Campus
       - What they built: Deep learning (CNN) system for predicting neurotransmitter types (GABA, ACh, glutamate, serotonin, octopamine, dopamine) from EM image
       patches at synaptic sites. Predictions available for both FAFB/FlyWire and HemiBrain volumes.
       - Key findings: Demonstrated that neurotransmitter identity can be classified directly from EM imagery at synaptic sites, enabling brain-wide neurotransmitter
       mapping without wet-lab experiments.
       - Paper: "Neurotransmitter Classification from Electron Microscopy Images at Synaptic Sites in Drosophila" (bioRxiv 2020)
       - Status: Active. Last commit Jan 2025. 16 stars, 6 forks. Python.
       - Limitations: Predictions are probabilistic (not ground truth). Limited to 6 NT types. Requires EM imagery as input. Accuracy depends on synapse quality.

       ---
       8. funkelab/drosophila_neurotransmitters

       - Team: Alexander Bates (Harvard), Diane Adjavon, Jan Funke (Janelia); Mark Eddison (EASI-FISH)
       - What they built: Curated ground-truth CSV mapping neurotransmitters to 900+ cell types from 71 published studies. Covers FAFB-FlyWire, HemiBrain, FANC, MANC,
        and larval L1 connectomes. Confidence-scored (0-5 scale).
       - Key findings: For FAFB-783 alone: 27.9M ACh synapses across 442 ACh cell types. Provides the empirical validation layer that synister and other NT prediction
        tools are benchmarked against.
       - Paper: Bates et al. 2019; Eckstein & Bates et al. 2024.
       - Status: Active. Last commit April 2024. 2 stars. Welcomes community PRs.
       - Limitations: Manual curation -- coverage gaps for rare NT types. Community-dependent for updates. Small engagement (2 stars).

       ---
       9. erojasoficial-byte/fly-brain

       - Team: Enrique Manuel Rojas Aliaga (independent researcher, Lima, Peru)
       - What they built: Full embodied simulation: 138,639 spiking neurons + 15.1M synapses (FlyWire v783) coupled to NeuroMechFly v2 biomechanical body (87 joints,
       6 legs, 2 wings) in MuJoCo. Includes vision (721 ommatidia/eye), olfaction, gustation, somatosensory, Hebbian plasticity, and "consciousness metrics" (Phi,
       Broadcast, Self-Model, Complexity).
       - Key findings: Two identical flies diverged after 24h of embodied experience: escape behavior 81% vs 47%, neural integration CI=0.221 vs 0.198, 76k divergent
       synapses. Claims "computational individuality" emerges from connectome + experience + plasticity.
       - Paper: Rojas 2026 (bioRxiv preprint, English + Spanish).
       - Status: Brand new (March 2026). 2 stars.
       - Limitations: Single-author, independent work -- no institutional backing or peer review. Extraordinary claims (consciousness metrics) without established
       validation. Reproducibility unverified by community. Very early stage.

       ---
       10. philshiu/Drosophila_brain_model

       - Team: Phil Shiu (independent / academic)
       - What they built: Leaky integrate-and-fire (LIF) model of the entire adult Drosophila brain using Brian 2 simulator. Supports FlyWire v630 and v783. Users can
        activate neurons (optogenetic-style Poisson spiking) or silence them (zero synaptic weights) by FlyWire ID. Outputs spike times and firing rates.
       - Key findings: Demonstrated that a simple LIF model over the full connectome can reveal insights into sensorimotor processing pathways.
       - Paper: "A leaky integrate-and-fire computational model based on the connectome of the entire adult Drosophila brain reveals insights into sensorimotor
       processing" (bioRxiv 2023).
       - Status: 122 stars, 29 forks (by far the most popular). MIT license. Last significant activity ~2023, appears stale.
       - Limitations: LIF is the simplest neuron model -- no dendritic computation, no plasticity, no neuromodulation. Preprint only. Development appears stalled
       despite high community interest.

       ---
       Cross-Cutting Observations

       ┌────────────────────┬─────────────────────────────────────────────────────────────┬─────────────────────────────────────────────────────────────────────┐
       │     Dimension      │                          Coverage                           │                                Gaps                                 │
       ├────────────────────┼─────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
       │ Connectome data    │ Female whole-brain (v783), male visual system, larval brain │ No unified male whole-brain repo yet                                │
       ├────────────────────┼─────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
       │ Annotations        │ Cell types, NT labels, cross-sex matching                   │ NT ground truth still manually curated                              │
       ├────────────────────┼─────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
       │ Network analysis   │ Motifs, modularity, rich-club, hierarchy                    │ Most use hemibrain or v630, not latest v783                         │
       ├────────────────────┼─────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
       │ NT prediction      │ synister CNN + ground truth CSV                             │ Only 6 NT types; neuropeptides missing                              │
       ├────────────────────┼─────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
       │ Simulation         │ LIF whole-brain (Shiu), embodied spiking (Rojas)            │ No biophysically detailed (Hodgkin-Huxley) whole-brain model exists │
       ├────────────────────┼─────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
       │ Community traction │ Shiu's LIF model dominates (122 stars)                      │ Most repos have under 10 stars                                      │
       └────────────────────┴─────────────────────────────────────────────────────────────┴─────────────────────────────────────────────────────────────────────┘
  ⎿  Done (11 tool uses · 17.5k tokens · 3m 17s)
