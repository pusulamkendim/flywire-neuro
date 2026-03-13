"""
FlyWire CAVEclient - Token Kurulumu
Bu scripti çalıştırarak token alıp kaydet.
"""
from caveclient import CAVEclient

# 1. Global client oluştur
client = CAVEclient(server_address="https://global.daf-apis.com")

# 2. Token alma talimatlarını göster
print("=" * 60)
print("FLYWIRE API TOKEN KURULUMU")
print("=" * 60)
print()
print("Adım 1: Aşağıdaki linki tarayıcıda aç:")
print("  https://global.daf-apis.com/auth/api/v1/create_token")
print()
print("Adım 2: Google hesabınla giriş yap")
print()
print("Adım 3: Sana verilen token'ı kopyala")
print()
print("Adım 4: Aşağıdaki komutu terminalde çalıştır:")
print('  python -c "from caveclient import CAVEclient; c = CAVEclient(server_address=\\"https://global.daf-apis.com\\"); c.auth.save_token(\\"BURAYA_TOKEN_YAPISTIR\\")"')
print()
print("Alternatif: Token'ı bu scripte yapıştırıp çalıştır:")
print("  python 00_setup_token.py TOKEN_BURAYA")
print()

import sys
if len(sys.argv) > 1:
    token = sys.argv[1]
    client.auth.save_token(token)
    print(f"Token kaydedildi!")
    print(f"Kayıt yeri: {client.auth.token_file}")
