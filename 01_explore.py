"""
FlyWire CAVEclient - Keşif Scripti
Mevcut tabloları ve veri yapısını gösterir.
"""
from caveclient import CAVEclient

# FlyWire public datastack'e bağlan
client = CAVEclient("flywire_fafb_production")

# 1. Datastack bilgisi
print("=" * 60)
print("DATASTACK BİLGİSİ")
print("=" * 60)
info = client.info.get_datastack_info()
for k, v in info.items():
    print(f"  {k}: {v}")

# 2. Mevcut tablolar
print("\n" + "=" * 60)
print("MEVCUT TABLOLAR")
print("=" * 60)
tables = client.materialize.get_tables()
for i, t in enumerate(tables, 1):
    print(f"  {i:3d}. {t}")
print(f"\n  TOPLAM: {len(tables)} tablo")

# 3. Her tablonun kısa açıklaması
print("\n" + "=" * 60)
print("TABLO DETAYLARI")
print("=" * 60)
for t in tables:
    try:
        meta = client.materialize.get_table_metadata(t)
        desc = meta.get("description", "Açıklama yok")
        print(f"\n  [{t}]")
        print(f"    Açıklama: {desc}")
    except Exception as e:
        print(f"\n  [{t}]")
        print(f"    Hata: {e}")
