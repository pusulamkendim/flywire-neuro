# FlyWire Neuro — Improvement Points

Bu belge, web arayüzündeki **Brain Activity (Dorsal View)** panelini gerçek LIF
nöron spike'larıyla beslemek için gerekli geliştirmeleri ve tahmini eforları
kaydeder.

## Varsayımlar

- Eforlar mevcut kod tabanını bilen tek bir geliştirici içindir.
- Bir iş günü yaklaşık 6 saat net geliştirme süresidir.
- İlk hedef lokal çalışan bir MVP'dir; internet üzerinde statik kayıt oynatma
  desteği ayrıca belirtilmiştir.
- Mevcut FlyWire v783 ve LIF tensor indeks sırası korunacaktır.

## Mevcut durum

- Anatomik arka plan gerçek FlyWire soma koordinatlarından örneklenmiş 10.000
  nokta kullanıyor.
- Frontend tekil aktif nöronları almıyor; yalnızca popülasyon ve DN toplamlarını
  alıyor.
- Frontend `pam`, `ppl1`, `mbon` ve `kc` alanlarını beklerken backend
  `mbon_approach`, `mbon_avoidance` ve `mbon_suppress` alanlarını gönderiyor.
- Son gözlenen simülasyonda ACh, GABA ve glutamat spike'ları bulunmasına rağmen
  Anatomy View bu toplu değerleri aktif görsel katmanda kullanmıyor.

## Geliştirme maddeleri ve eforlar

| # | Geliştirme | Çıktı | Tahmini efor | Bağımlılık |
|---|-------------|-------|---------------|------------|
| 1 | Backend'den sparse aktif nöron indekslerini gönder | Her yayın aralığında yalnızca spike üreten tensor indeksleri ve spike sayıları WebSocket karesine eklenir | 0,5–1 gün | Yok |
| 2 | Simülasyon indekslerini FlyWire soma koordinatlarıyla eşleştir | Tensor indeksi → `root_id` → normalize soma koordinatı → NT tipi eşlemesini içeren sürümlü harita varlığı üretilir | 1–2 gün | #1 ile paralel ilerleyebilir |
| 3 | Frontend'de gerçek aktif nöronları çiz | Aktif indeksler koordinat haritasından bulunur ve sadece gerçekten ateşleyen nöronlar dinamik katmanda gösterilir | 1–1,5 gün | #1, #2 |
| 4 | Spike şiddeti ve temporal sönümlenme uygula | Nokta boyutu/parlaklığı spike sayısına bağlanır; hızlı yükselme ve kontrollü sönümlenme eklenir | 0,5–1 gün | #3 |
| 5 | MBON ve diğer popülasyon alanlarını düzelt | `mbon_approach + mbon_avoidance + mbon_suppress` toplamı veya ayrı katmanları kullanılır; eksik KC alanı backend'e eklenir | 0,5 gün | Yok |
| 6 | Bulk NT aktivitesini doğru bağla | ACh, GABA, glutamat, serotonin, oktopamin ve dopamin toplamları legend/gösterge katmanına bağlanır | 0,5 gün | #3 önerilir |
| 7 | Performans ve yük testi | 10K/50K/139K anatomik nokta, farklı spike yoğunlukları ve 30 FPS hedefi ölçülür | 0,5–1 gün | #3, #4 |
| 8 | Otomatik testler | İndeks eşleme, frame şeması, boş spike karesi, yüksek aktivite ve frontend veri dönüşümü test edilir | 0,5–1 gün | #1–#5 |

## Uygulama ayrıntıları

### 1. Sparse spike frame şeması

Önerilen kare alanı:

```json
{
  "event": "brain_frame",
  "t_ms": 120.0,
  "active_neurons": [
    [1842, 3],
    [5811, 1],
    [7420, 5]
  ]
}
```

- İlk değer LIF tensor indeksidir.
- İkinci değer yayın aralığındaki spike sayısıdır.
- Sıfır aktiviteye sahip nöronlar gönderilmez.
- Frame başına maksimum nöron sayısı için güvenlik sınırı konulmalıdır.

### 2. Anatomi eşleme varlığı

İlk MVP için okunabilir JSON kullanılabilir:

```json
{
  "version": "flywire-v783-lif-v1",
  "neurons": [
    [1842, 0.581, 0.805, 0]
  ]
}
```

Alanlar sırasıyla `tensor_index`, normalize `x`, normalize `y` ve NT indeksidir.
Üretim sürümünde dosya boyutunu azaltmak için koordinatlar `uint16`, NT tipi
`uint8` ve tensor indeksi `uint32` olarak binary saklanabilir.

### 3. Frontend çizim yaklaşımı

- Statik soma bulutu mevcut önbelleklenmiş katmanda kalır.
- Her frame'de yalnızca `active_neurons` dinamik katmanı güncellenir.
- Nöron rengi gerçek NT tipinden alınır.
- Parlaklık `log1p(spike_count)` ile normalize edilir.
- Son aktivite değeri nöron başına kısa bir decay buffer içinde tutulur.
- Çok yoğun karelerde en yüksek spike değerli nöronlara öncelik verilir.

### 4. Popülasyon şeması düzeltmesi

Kısa vadede frontend şu alanları doğrudan kullanmalıdır:

- `mbon_approach`
- `mbon_avoidance`
- `mbon_suppress`
- `gaba`
- `ach`
- `glut`
- `serotonin`
- `octopamine`
- `pam`
- `ppl1`

KC aktivitesi isteniyorsa backend'deki `pop_tensors` sözlüğüne gerçek KC
indeksleri ayrıca eklenmelidir.

## Kabul kriterleri

- Simülasyon boşken dinamik katmanda aktif nöron görünmez.
- P9, LC4 veya başka bir uyaran açıldığında gerçekten spike üreten nöronlar
  kendi soma koordinatlarında görünür.
- Ekrandaki aktif nöron sayısı backend frame verisiyle doğrulanabilir.
- NT rengi, nöronun gerçek anotasyonuyla eşleşir.
- Aktivite yeni frame gelmediğinde kontrollü biçimde söner.
- 10.000 statik nokta ve tipik sparse spike yükünde en az 30 görsel FPS korunur.
- Boş, eksik veya aşırı büyük spike frame'i arayüzü çökertmez.

## Tahmini toplam

### Lokal MVP

- Gerçek spike indeksleri
- Soma eşleme dosyası
- Canvas üzerinde gerçek aktif nöron çizimi
- Şema düzeltmeleri ve temel testler

**Tahmin: 4–6 geliştirici günü.**

### Dışarıya açılmaya hazır sürüm

Lokal MVP'ye ek olarak:

- Binary/sıkıştırılmış veri formatı
- WebGL renderer veya yüksek yoğunluk optimizasyonu
- Kayıt dosyası oynatma desteği
- Hata toleransı, performans testi ve tarayıcı uyumluluğu

**Tahmin: toplam 7–10 geliştirici günü.**

## Önerilen sıra

1. #5 — mevcut popülasyon şemasını düzelt.
2. #1 — backend sparse spike frame'ini üret.
3. #2 — tensor indeksi ile soma koordinatını eşleştir.
4. #3 — gerçek aktif nöronları çiz.
5. #4 ve #6 — görsel şiddet, decay ve NT katmanlarını tamamla.
6. #7 ve #8 — performans ve doğruluk testlerini çalıştır.

## İleri aşama: çevrimdışı kayıt oynatma

Dış web sunucusunda 139K LIF modelini çalıştırmamak için aynı
`active_neurons` frame şeması lokal simülasyondan kayıt dosyasına yazılabilir.
Tarayıcı bu kaydı statik bir varlık olarak indirip Anatomy View üzerinde oynatır.
Bu çalışma için ek tahmin:

- Kayıt yazıcı ve metadata: 0,5–1 gün
- Tarayıcı oynatıcı, seek ve hız kontrolü: 1–1,5 gün
- Sıkıştırma ve büyük dosya testi: 0,5–1 gün

**Ek toplam: 2–3 geliştirici günü.**
