"""
FlyWire Analiz 14 - Ödül/Ceza Öğrenme Simülasyonu
"Sinek koku + şeker ile yaklaşmayı, koku + şok ile kaçmayı öğreniyor mu?"

Model: Gerçek connectome verisine dayalı Hebbian öğrenme
- Gerçek KC→MBON bağlantı ağırlıkları (başlangıç)
- Gerçek PAM→KC ve PPL1→KC bağlantıları (dopamin sinyali)
- MBON tiplerine göre davranış çıkışı (yaklaş / kaç / bastır)
- Hebbian kural: birlikte aktif → güçlen, ayrı aktif → zayıfla
"""
import pandas as pd
import numpy as np
from collections import defaultdict

print("Veri yükleniyor...")
conn = pd.read_feather("data/proofread_connections_783.feather")
ann = pd.read_csv("data/neuron_annotations.tsv", sep="\t", low_memory=False)

# === Devre elemanlarını hazırla ===
print("Devre elemanları hazırlanıyor...")

# Kenyon Cell'ler
kc_neurons = ann[ann['cell_class'] == 'Kenyon_Cell']
kc_ids = list(kc_neurons['root_id'].values)
kc_set = set(kc_ids)
print(f"  Kenyon Cell: {len(kc_ids)}")

# MBON'lar — NT tipine göre rollerini belirle
mbon_neurons = ann[ann['cell_class'] == 'MBON'].copy()
mbon_ids = list(mbon_neurons['root_id'].values)
mbon_set = set(mbon_ids)

# MBON rolleri: ACh → yaklaş (+1), Glut → kaç (-1), GABA → bastır (-0.5)
mbon_role = {}
for _, row in mbon_neurons.iterrows():
    rid = row['root_id']
    nt = row['top_nt']
    if nt == 'acetylcholine':
        mbon_role[rid] = ('yaklaş', +1.0)
    elif nt == 'glutamate':
        mbon_role[rid] = ('kaç', -1.0)
    elif nt == 'gaba':
        mbon_role[rid] = ('bastır', -0.3)
    else:
        mbon_role[rid] = ('nötr', 0.0)

ach_mbon = sum(1 for r in mbon_role.values() if r[0] == 'yaklaş')
glut_mbon = sum(1 for r in mbon_role.values() if r[0] == 'kaç')
gaba_mbon = sum(1 for r in mbon_role.values() if r[0] == 'bastır')
print(f"  MBON: {len(mbon_ids)} (yaklaş:{ach_mbon}, kaç:{glut_mbon}, bastır:{gaba_mbon})")

# PAM ve PPL1
pam_neurons = ann[ann['cell_type'].str.startswith('PAM', na=False)]
ppl1_neurons = ann[ann['cell_type'].str.startswith('PPL1', na=False)]
pam_ids = set(pam_neurons['root_id'])
ppl1_ids = set(ppl1_neurons['root_id'])
print(f"  PAM (ödül): {len(pam_ids)}")
print(f"  PPL1 (ceza): {len(ppl1_ids)}")

# === Gerçek bağlantı ağırlıklarını çıkar ===
print("\nBağlantılar çıkarılıyor...")

# KC → MBON bağlantıları (öğrenilebilir sinapslar)
kc_mbon_conn = conn[
    (conn['pre_pt_root_id'].isin(kc_set)) &
    (conn['post_pt_root_id'].isin(mbon_set))
]
print(f"  KC → MBON: {len(kc_mbon_conn):,} bağlantı, {kc_mbon_conn['syn_count'].sum():,} sinaps")

# KC → MBON ağırlık matrisi (sparse — dict of dicts)
kc_to_mbon_weights = defaultdict(lambda: defaultdict(float))
for _, row in kc_mbon_conn.iterrows():
    kc = row['pre_pt_root_id']
    mbon = row['post_pt_root_id']
    kc_to_mbon_weights[kc][mbon] = float(row['syn_count'])

# PAM → KC bağlantıları (ödül sinyali)
pam_kc_conn = conn[
    (conn['pre_pt_root_id'].isin(pam_ids)) &
    (conn['post_pt_root_id'].isin(kc_set))
]
pam_target_kcs = set(pam_kc_conn['post_pt_root_id'])
print(f"  PAM → KC: {len(pam_kc_conn):,} bağlantı → {len(pam_target_kcs):,} KC'ye ulaşıyor")

# PAM'dan gelen sinyal gücü (KC başına)
pam_signal = defaultdict(float)
for _, row in pam_kc_conn.iterrows():
    pam_signal[row['post_pt_root_id']] += row['syn_count']
# Normalize
max_pam = max(pam_signal.values()) if pam_signal else 1
for k in pam_signal:
    pam_signal[k] /= max_pam

# PPL1 → KC bağlantıları (ceza sinyali)
ppl1_kc_conn = conn[
    (conn['pre_pt_root_id'].isin(ppl1_ids)) &
    (conn['post_pt_root_id'].isin(kc_set))
]
ppl1_target_kcs = set(ppl1_kc_conn['post_pt_root_id'])
print(f"  PPL1 → KC: {len(ppl1_kc_conn):,} bağlantı → {len(ppl1_target_kcs):,} KC'ye ulaşıyor")

# PPL1'den gelen sinyal gücü
ppl1_signal = defaultdict(float)
for _, row in ppl1_kc_conn.iterrows():
    ppl1_signal[row['post_pt_root_id']] += row['syn_count']
max_ppl1 = max(ppl1_signal.values()) if ppl1_signal else 1
for k in ppl1_signal:
    ppl1_signal[k] /= max_ppl1

# Koku→KC bağlantıları (hangi KC'ler koku alıyor)
orn_ids = set(ann[ann['cell_class'] == 'olfactory']['root_id'])
alpn_ids = set(ann[ann['cell_class'] == 'ALPN']['root_id'])

# ALPN → KC (projeksiyon nöronlarından hafıza hücrelerine)
alpn_kc_conn = conn[
    (conn['pre_pt_root_id'].isin(alpn_ids)) &
    (conn['post_pt_root_id'].isin(kc_set))
]
alpn_to_kc = defaultdict(float)
for _, row in alpn_kc_conn.iterrows():
    alpn_to_kc[row['post_pt_root_id']] += row['syn_count']

kc_with_odor_input = set(alpn_to_kc.keys())
print(f"  ALPN → KC: {len(kc_with_odor_input):,} KC koku bilgisi alıyor")

# Normalize KC başlangıç ağırlıklarını
all_weights = []
for kc in kc_to_mbon_weights:
    for mbon in kc_to_mbon_weights[kc]:
        all_weights.append(kc_to_mbon_weights[kc][mbon])
mean_weight = np.mean(all_weights) if all_weights else 1.0
for kc in kc_to_mbon_weights:
    for mbon in kc_to_mbon_weights[kc]:
        kc_to_mbon_weights[kc][mbon] /= mean_weight

# Başlangıç ağırlıklarının kopyasını sakla
original_weights = {}
for kc in kc_to_mbon_weights:
    original_weights[kc] = dict(kc_to_mbon_weights[kc])

# === Koku Temsili ===
# Her koku rastgele bir KC alt kümesini aktive eder (sparse coding — gerçekçi)
np.random.seed(42)
kc_list = list(kc_with_odor_input)

# Koku A: KC'lerin %10'u (sparse coding — gerçek sinekte %5-15)
odor_a_kcs = set(np.random.choice(kc_list, size=int(len(kc_list) * 0.10), replace=False))
# Koku B: farklı %10 alt kümesi
remaining = list(set(kc_list) - odor_a_kcs)
odor_b_kcs = set(np.random.choice(remaining, size=int(len(kc_list) * 0.10), replace=False))
# Koku C: test için — hiç eğitim görmemiş
remaining2 = list(set(kc_list) - odor_a_kcs - odor_b_kcs)
odor_c_kcs = set(np.random.choice(remaining2, size=int(len(kc_list) * 0.10), replace=False))

print(f"\n  Koku A: {len(odor_a_kcs)} KC aktif (ödülle eşleşecek)")
print(f"  Koku B: {len(odor_b_kcs)} KC aktif (cezayla eşleşecek)")
print(f"  Koku C: {len(odor_c_kcs)} KC aktif (kontrol — eğitimsiz)")
print(f"  Örtüşme A∩B: {len(odor_a_kcs & odor_b_kcs)} KC")

# === Davranış Hesaplama Fonksiyonu ===
def compute_behavior(active_kcs, weights):
    """Aktif KC'lerden MBON çıkışını hesapla → davranış skoru"""
    mbon_activation = defaultdict(float)

    for kc in active_kcs:
        if kc in weights:
            for mbon, w in weights[kc].items():
                mbon_activation[mbon] += w

    # Davranış skoru: MBON aktivasyonu × rol çarpanı
    approach_score = 0.0  # yaklaş
    avoid_score = 0.0     # kaç
    suppress_score = 0.0  # bastır

    for mbon, act in mbon_activation.items():
        if mbon in mbon_role:
            role_name, role_sign = mbon_role[mbon]
            if role_sign > 0:
                approach_score += act * role_sign
            elif role_sign < -0.5:
                avoid_score += act * abs(role_sign)
            else:
                suppress_score += act * abs(role_sign)

    # Normalize
    total = approach_score + avoid_score + suppress_score
    if total == 0:
        return 0.0, 0.0, 0.0, 0.0

    # Net davranış: pozitif = yaklaş, negatif = kaç
    net_behavior = (approach_score - avoid_score - suppress_score)

    return net_behavior, approach_score, avoid_score, suppress_score

# === ÖĞRENME PARAMETRELERİ ===
LEARNING_RATE_REWARD = 0.15   # ödül öğrenme hızı
LEARNING_RATE_PUNISH = 0.25   # ceza öğrenme hızı (PPL1 daha güçlü)
WEIGHT_MIN = 0.1              # minimum sinaps ağırlığı
WEIGHT_MAX = 5.0              # maksimum sinaps ağırlığı
NUM_TRAINING = 10             # eğitim tekrar sayısı
NUM_EXTINCTION = 15           # söndürme tekrar sayısı

# === SİMÜLASYON ===
print("\n" + "=" * 80)
print("ÖDÜL/CEZA ÖĞRENME SİMÜLASYONU")
print("=" * 80)

# --- Faz 0: Naif sinek ---
print("\n" + "-" * 60)
print("FAZ 0: NAİF SİNEK (öğrenme öncesi)")
print("-" * 60)

for odor_name, odor_kcs in [("Koku A", odor_a_kcs), ("Koku B", odor_b_kcs), ("Koku C", odor_c_kcs)]:
    net, approach, avoid, suppress = compute_behavior(odor_kcs, kc_to_mbon_weights)
    decision = "→ YAKLAŞ" if net > 0 else "→ KAÇ" if net < 0 else "→ NÖTR"
    print(f"\n  {odor_name}: net={net:>8.1f}  yaklaş={approach:>8.1f}  kaç={avoid:>8.1f}  bastır={suppress:>6.1f}  {decision}")

# --- Faz 1: Eğitim ---
print("\n" + "-" * 60)
print(f"FAZ 1: EĞİTİM ({NUM_TRAINING} tekrar)")
print("-" * 60)

print(f"\n  Koku A + ŞEKER (ödül) → PAM aktif → KC→MBON güçlendir")
print(f"  Koku B + ŞOK (ceza)  → PPL1 aktif → KC→MBON zayıflat")

print(f"\n{'Tekrar':<8} {'Koku A net':>12} {'Karar A':>10} {'Koku B net':>12} {'Karar B':>10} {'Koku C net':>12} {'Karar C':>10}")
print("-" * 80)

training_history = []

for trial in range(NUM_TRAINING):
    # --- Koku A + Ödül ---
    for kc in odor_a_kcs:
        if kc in kc_to_mbon_weights:
            reward_strength = pam_signal.get(kc, 0.0)
            if reward_strength > 0:
                for mbon in kc_to_mbon_weights[kc]:
                    role_name, role_sign = mbon_role.get(mbon, ('nötr', 0))
                    # Ödül: yaklaş MBON'ları güçlendir, kaç MBON'ları zayıflat
                    if role_sign > 0:  # yaklaş
                        delta = LEARNING_RATE_REWARD * reward_strength
                        kc_to_mbon_weights[kc][mbon] = min(WEIGHT_MAX,
                            kc_to_mbon_weights[kc][mbon] + delta)
                    elif role_sign < -0.5:  # kaç
                        delta = LEARNING_RATE_REWARD * reward_strength * 0.5
                        kc_to_mbon_weights[kc][mbon] = max(WEIGHT_MIN,
                            kc_to_mbon_weights[kc][mbon] - delta)

    # --- Koku B + Ceza ---
    for kc in odor_b_kcs:
        if kc in kc_to_mbon_weights:
            punish_strength = ppl1_signal.get(kc, 0.0)
            if punish_strength > 0:
                for mbon in kc_to_mbon_weights[kc]:
                    role_name, role_sign = mbon_role.get(mbon, ('nötr', 0))
                    # Ceza: kaç MBON'ları güçlendir, yaklaş MBON'ları zayıflat
                    if role_sign < -0.5:  # kaç
                        delta = LEARNING_RATE_PUNISH * punish_strength
                        kc_to_mbon_weights[kc][mbon] = min(WEIGHT_MAX,
                            kc_to_mbon_weights[kc][mbon] + delta)
                    elif role_sign > 0:  # yaklaş
                        delta = LEARNING_RATE_PUNISH * punish_strength * 0.5
                        kc_to_mbon_weights[kc][mbon] = max(WEIGHT_MIN,
                            kc_to_mbon_weights[kc][mbon] - delta)

    # Ölç
    net_a, ap_a, av_a, su_a = compute_behavior(odor_a_kcs, kc_to_mbon_weights)
    net_b, ap_b, av_b, su_b = compute_behavior(odor_b_kcs, kc_to_mbon_weights)
    net_c, ap_c, av_c, su_c = compute_behavior(odor_c_kcs, kc_to_mbon_weights)

    dec_a = "YAKLAŞ ✓" if net_a > 0 else "KAÇ ✗"
    dec_b = "KAÇ ✓" if net_b < 0 else "YAKLAŞ ✗"
    dec_c = "NÖTR" if abs(net_c) < abs(net_a) * 0.3 else ("YAKLAŞ" if net_c > 0 else "KAÇ")

    training_history.append({
        'trial': trial + 1, 'phase': 'train',
        'net_a': net_a, 'net_b': net_b, 'net_c': net_c,
    })

    print(f"  {trial+1:<6} {net_a:>12.1f} {dec_a:>10} {net_b:>12.1f} {dec_b:>10} {net_c:>12.1f} {dec_c:>10}")

# --- Faz 2: Test ---
print("\n" + "-" * 60)
print("FAZ 2: TEST (eğitim sonrası)")
print("-" * 60)

for odor_name, odor_kcs in [("Koku A (ödüllü)", odor_a_kcs), ("Koku B (cezalı)", odor_b_kcs), ("Koku C (kontrol)", odor_c_kcs)]:
    net, approach, avoid, suppress = compute_behavior(odor_kcs, kc_to_mbon_weights)
    decision = "→ YAKLAŞ" if net > 0 else "→ KAÇ"

    pct_approach = approach / (approach + avoid + suppress) * 100 if (approach + avoid + suppress) > 0 else 0
    pct_avoid = avoid / (approach + avoid + suppress) * 100 if (approach + avoid + suppress) > 0 else 0

    print(f"\n  {odor_name}:")
    print(f"    Net skor: {net:>8.1f}  {decision}")
    print(f"    Yaklaş: {approach:>8.1f} ({pct_approach:.0f}%)  |  Kaç: {avoid:>8.1f} ({pct_avoid:.0f}%)")

# --- Faz 3: Söndürme (Extinction) ---
print("\n" + "-" * 60)
print(f"FAZ 3: SÖNDÜRME — Koku A ödülsüz sunuluyor ({NUM_EXTINCTION} tekrar)")
print("-" * 60)
print("  (Koku A tekrar tekrar sunuluyor ama şeker verilmiyor)")

# Naif duruma dönüş hızı
EXTINCTION_RATE = 0.08  # ödülsüz sunumda sinapslar yavaşça orijinale döner

print(f"\n{'Tekrar':<8} {'Koku A net':>12} {'Karar':>10} {'Koku B net':>12} {'B Karar':>10}")
print("-" * 60)

extinction_history = []

for trial in range(NUM_EXTINCTION):
    # Koku A ödülsüz — ağırlıklar yavaşça orijinale döner
    for kc in odor_a_kcs:
        if kc in kc_to_mbon_weights and kc in original_weights:
            for mbon in kc_to_mbon_weights[kc]:
                if mbon in original_weights.get(kc, {}):
                    orig = original_weights[kc][mbon] / mean_weight
                    current = kc_to_mbon_weights[kc][mbon]
                    # Orijinale doğru çek
                    kc_to_mbon_weights[kc][mbon] = current + EXTINCTION_RATE * (orig - current)

    net_a, _, _, _ = compute_behavior(odor_a_kcs, kc_to_mbon_weights)
    net_b, _, _, _ = compute_behavior(odor_b_kcs, kc_to_mbon_weights)

    dec_a = "YAKLAŞ" if net_a > 0 else "KAÇ" if net_a < 0 else "NÖTR"
    dec_b = "KAÇ" if net_b < 0 else "YAKLAŞ"

    extinction_history.append({
        'trial': trial + 1, 'net_a': net_a, 'net_b': net_b,
    })

    print(f"  {trial+1:<6} {net_a:>12.1f} {dec_a:>10} {net_b:>12.1f} {dec_b:>10}")

# === SONUÇ ANALİZİ ===
print("\n" + "=" * 80)
print("SONUÇ ANALİZİ")
print("=" * 80)

# Öğrenme eğrisi
print("\nÖğrenme Eğrisi (Koku A — ödül):")
initial_a = training_history[0]['net_a']
final_a = training_history[-1]['net_a']
change_a = final_a - initial_a
print(f"  Başlangıç: {initial_a:>8.1f}")
print(f"  Son:       {final_a:>8.1f}")
print(f"  Değişim:   {change_a:>+8.1f} ({change_a/abs(initial_a)*100:+.0f}%)" if initial_a != 0 else f"  Değişim:   {change_a:>+8.1f}")

print("\nÖğrenme Eğrisi (Koku B — ceza):")
initial_b = training_history[0]['net_b']
final_b = training_history[-1]['net_b']
change_b = final_b - initial_b
print(f"  Başlangıç: {initial_b:>8.1f}")
print(f"  Son:       {final_b:>8.1f}")
print(f"  Değişim:   {change_b:>+8.1f} ({change_b/abs(initial_b)*100:+.0f}%)" if initial_b != 0 else f"  Değişim:   {change_b:>+8.1f}")

# Ceza vs Ödül hızı
print(f"\nCeza vs Ödül Öğrenme Hızı:")
print(f"  Ödül (Koku A) değişim:  {abs(change_a):>8.1f}")
print(f"  Ceza (Koku B) değişim:  {abs(change_b):>8.1f}")
if abs(change_a) > 0:
    ratio = abs(change_b) / abs(change_a)
    print(f"  Ceza/Ödül oranı:        {ratio:.1f}x")
    if ratio > 1:
        print(f"  → Ceza {ratio:.1f}x daha hızlı öğreniliyor!")
    else:
        print(f"  → Ödül {1/ratio:.1f}x daha hızlı öğreniliyor!")

# Kontrol kokusu
print(f"\nKontrol (Koku C — eğitimsiz):")
net_c_final, _, _, _ = compute_behavior(odor_c_kcs, kc_to_mbon_weights)
print(f"  Başlangıç: {training_history[0]['net_c']:>8.1f}")
print(f"  Son:       {net_c_final:>8.1f}")
print(f"  → Kontrol kokusu {'değişmedi ✓' if abs(net_c_final - training_history[0]['net_c']) < abs(change_a) * 0.2 else 'değişti ✗ (genelleme!)'}")

# Söndürme
print(f"\nSöndürme (Koku A ödülsüz):")
ext_start = extinction_history[0]['net_a']
ext_end = extinction_history[-1]['net_a']
print(f"  Eğitim sonu:    {ext_start:>8.1f}")
print(f"  Söndürme sonu:  {ext_end:>8.1f}")
recovery = abs(ext_end - initial_a) / abs(final_a - initial_a) * 100 if abs(final_a - initial_a) > 0 else 0
print(f"  Orijinale dönüş: %{100 - recovery:.0f}")
print(f"  → Koku B (ceza) söndürme sırasında: {extinction_history[-1]['net_b']:.1f} (sabit mi?)")

# Dopamin kapsama
print(f"\n" + "=" * 80)
print("DOPAMİN KAPSAMA ANALİZİ")
print("=" * 80)

both = pam_target_kcs & ppl1_target_kcs
only_pam = pam_target_kcs - ppl1_target_kcs
only_ppl1 = ppl1_target_kcs - pam_target_kcs
neither = kc_set - pam_target_kcs - ppl1_target_kcs

print(f"\n  Toplam KC: {len(kc_set):,}")
print(f"  PAM alan (ödül): {len(pam_target_kcs):,} ({len(pam_target_kcs)/len(kc_set)*100:.1f}%)")
print(f"  PPL1 alan (ceza): {len(ppl1_target_kcs):,} ({len(ppl1_target_kcs)/len(kc_set)*100:.1f}%)")
print(f"  İkisini de alan: {len(both):,} ({len(both)/len(kc_set)*100:.1f}%)")
print(f"  Sadece PAM: {len(only_pam):,} ({len(only_pam)/len(kc_set)*100:.1f}%)")
print(f"  Sadece PPL1: {len(only_ppl1):,} ({len(only_ppl1)/len(kc_set)*100:.1f}%)")
print(f"  Hiçbirini almayan: {len(neither):,} ({len(neither)/len(kc_set)*100:.1f}%)")

# Koku A ve B'nin dopamin kapsamı
odor_a_pam = len(odor_a_kcs & pam_target_kcs)
odor_a_ppl1 = len(odor_a_kcs & ppl1_target_kcs)
odor_b_pam = len(odor_b_kcs & pam_target_kcs)
odor_b_ppl1 = len(odor_b_kcs & ppl1_target_kcs)

print(f"\n  Koku A KC'leri:")
print(f"    PAM alan: {odor_a_pam} / {len(odor_a_kcs)} ({odor_a_pam/len(odor_a_kcs)*100:.0f}%) — ödül öğrenebilir")
print(f"    PPL1 alan: {odor_a_ppl1} / {len(odor_a_kcs)} ({odor_a_ppl1/len(odor_a_kcs)*100:.0f}%)")

print(f"\n  Koku B KC'leri:")
print(f"    PAM alan: {odor_b_pam} / {len(odor_b_kcs)} ({odor_b_pam/len(odor_b_kcs)*100:.0f}%)")
print(f"    PPL1 alan: {odor_b_ppl1} / {len(odor_b_kcs)} ({odor_b_ppl1/len(odor_b_kcs)*100:.0f}%) — ceza öğrenebilir")

# === MBON ağırlık değişimi detayı ===
print(f"\n" + "=" * 80)
print("MBON AĞIRLIK DEĞİŞİMİ (eğitim öncesi → sonrası)")
print("=" * 80)

# Her MBON'un ortalama ağırlık değişimini hesapla
mbon_weight_change = defaultdict(lambda: {'before': [], 'after': []})

for kc in odor_a_kcs:
    if kc in original_weights:
        for mbon in original_weights[kc]:
            orig_w = original_weights[kc][mbon] / mean_weight
            curr_w = kc_to_mbon_weights[kc].get(mbon, orig_w)
            mbon_weight_change[mbon]['before'].append(orig_w)
            mbon_weight_change[mbon]['after'].append(curr_w)

print(f"\n  Koku A (ödüllü) — MBON ağırlık değişimleri:")
print(f"  {'MBON Tipi':<15} {'Rol':<10} {'Önce':>8} {'Sonra':>8} {'Değişim':>10}")
print(f"  {'-'*55}")

changes_by_role = defaultdict(list)
for mbon in sorted(mbon_weight_change.keys(), key=lambda x: mbon_role.get(x, ('?',0))[0]):
    if len(mbon_weight_change[mbon]['before']) > 0:
        before = np.mean(mbon_weight_change[mbon]['before'])
        after = np.mean(mbon_weight_change[mbon]['after'])
        change = after - before
        role_name = mbon_role.get(mbon, ('?', 0))[0]
        changes_by_role[role_name].append(change)

for role in ['yaklaş', 'kaç', 'bastır']:
    if changes_by_role[role]:
        avg_change = np.mean(changes_by_role[role])
        print(f"  {role:<15} {'':<10} {'':<8} {'':<8} {avg_change:>+10.3f}  {'↑ güçlendi' if avg_change > 0 else '↓ zayıfladı'}")

print(f"\n  Beklenen: yaklaş ↑ güçlenir, kaç ↓ zayıflar (ödüllü koku)")
