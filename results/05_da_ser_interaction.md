# Analiz 05 - Dopamin ↔ Serotonin Etkileşim Analizi

Script: `05_da_ser_interaction.py`

---

## 1. Doğrudan Bağlantılar

| Yön | Bağlantı | Sinaps | Açıklama |
|-----|----------|--------|----------|
| DA → SER | 2,430 | 13,755 | Dopamin nöronları serotonine sinyal gönderiyor |
| SER → DA | 2,456 | 15,233 | Serotonin nöronları dopamine sinyal gönderiyor |
| DA → DA | 495,444 | 681,341 | Dopamin kendi içinde konuşuyor (50x daha fazla!) |
| SER → SER | 13,961 | 57,971 | Serotonin kendi içinde konuşuyor |

### Kritik Bulgu: Beklenenden 5x DAHA AZ etkileşim

- Rastgele beklenen DA↔SER bağlantı: ~11,717
- Gerçek: ~2,430-2,456
- **Oran: 0.21x** — İki sistem birbirinden aktif olarak kaçınıyor!

Bu "rastgele olsa bile bu kadar bağlantı olurdu" hesabından 5 kat daha az bağlantı var. Yani bu iki sistem sadece farklı bölgelerde değil, **kasıtlı olarak birbirinden izole**.

### Dopamin kendi içinde çok konuşuyor

DA→DA: 681,341 sinaps — bu devasa bir iç iletişim ağı. Kenyon hücreleri birbirleriyle sürekli haberleşiyor. Serotonin ise daha az iç iletişim yapıyor (57,971 sinaps).

## 2. Buluşma Noktaları

İki sistem doğrudan bağlantı kurduğunda, neredeyse tamamı **3 bölgede** gerçekleşiyor:

| Bölge | DA→SER | SER→DA | Toplam Oran | Açıklama |
|-------|--------|--------|-------------|----------|
| **AL_L** (Antennal Lobe sol) | 6,159 | 6,378 | ~%43 | Koku merkezi |
| **AL_R** (Antennal Lobe sağ) | 3,360 | 3,440 | ~%23 | Koku merkezi |
| **FB** (Fan-shaped Body) | 2,396 | 2,303 | ~%17 | Navigasyon merkezi |

**Toplam: Bu 3 bölge etkileşimin %83'ünü oluşturuyor.**

### Buluşmanın Anahtar Nöronu: lLN1_bc

En güçlü DA↔SER bağlantılarının **tamamı** aynı hücre tipinden: **lLN1_bc** (Local Lateral Neuron, Antennal Lobe Local Neuron sınıfı).

- Bunlar koku merkezindeki yerel ara nöronlar
- Hem dopaminerjik hem serotonerjik olarak etiketlenmiş olanları var
- Bağlantı başına 120-147 sinaps — bu çok güçlü bir bağlantı
- Koku işleme sırasında iki sistemi koordine eden özel hücreler

## 3. Köprü Nöronlar (Hem DA hem SER'den sinyal alanlar)

| Metrik | Değer |
|--------|-------|
| DA'dan sinyal alan | 50,761 nöron |
| SER'den sinyal alan | 29,069 nöron |
| **Her ikisinden de alan (köprü)** | **19,235 nöron** |

19,235 nöron her iki sistemden de bilgi alıyor — bu beyindeki toplam nöronların **%14'ü**.

### Köprü Nöronların Profili

**NT Tipi:**
- Asetilkolin: 11,048 (%57) — uyarıcı nöronlar baskın
- Glutamat: 4,143 (%22)
- GABA: 2,942 (%15) — inhibitör nöronlar da önemli bir kısım
- Serotonin: 545 (%3)
- Dopamin: 505 (%3)

**Süper Sınıf:**
- Central: 13,371 (%70) — merkezi beyin
- Optic: 2,247 (%12) — görme
- Sensory: 1,091 (%6) — duyusal

**Hücre Sınıfı (öne çıkanlar):**
- CX (Central Complex): 1,855 — navigasyon/karar merkezi
- ALPN (Antennal Lobe Projection Neurons): 594 — koku bilgisini ileten nöronlar
- ALLN (Antennal Lobe Local Neurons): 348 — koku yerel işleme

## 4. Üretilen Grafikler

- `07_da_ser_interaction.png` — Sinaps matrisi + buluşma bölgeleri
- `08_bridge_neurons.png` — Köprü nöronların NT ve süper sınıf dağılımı
- `09_interaction_diagram.png` — Özet etkileşim şeması

---

## Özet Hikaye

```
ALGILAMA (Serotonin)          KÖPRÜ              DEĞERLENDİRME (Dopamin)

Koku geldi                  lLN1_bc              Kenyon Cell
    ↓                    (Antennal Lobe           ↓
Antennal Lobe             yerel nöronları)    Mushroom Body
Hassasiyeti ayarla        İki sistemi          "Bu koku iyi miydi?"
    ↓                    koordine et              ↓
    └──────────────────────→↓←────────────────────┘
                       CX (Central Complex)
                       "Nereye gideyim?"
                       Navigasyon kararı
```

1. **İki sistem kasıtlı olarak ayrı** (beklenenden 5x az bağlantı)
2. **Buluşma noktası: Koku merkezi (AL)** — etkileşimin %66'sı burada
3. **Özel köprü hücreler var: lLN1_bc** — koku işlemede iki sistemi koordine eden nöronlar
4. **19,235 köprü nöron** (%14) her iki sistemden bilgi alarak entegrasyon yapıyor
5. **CX (Central Complex)** en büyük köprü — navigasyon kararlarında iki sistem buluşuyor
