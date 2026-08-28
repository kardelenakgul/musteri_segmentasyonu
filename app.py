import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

np.random.seed(42)

# --- 1. 500 KİŞİLİK MÜŞTERİ VERİSİ SİMÜLASYONU ---
recency_vip = np.random.randint(1, 20, 100)
freq_vip = np.random.randint(15, 40, 100)
monetary_vip = np.random.randint(3000, 8000, 100)

recency_churn = np.random.randint(60, 180, 150)
freq_churn = np.random.randint(1, 4, 150)
monetary_churn = np.random.randint(100, 600, 150)

recency_std = np.random.randint(15, 60, 250)
freq_std = np.random.randint(4, 15, 250)
monetary_std = np.random.randint(600, 2500, 250)

df = pd.DataFrame({
    'Musteri_ID': range(1, 501),
    'Recency': np.concatenate([recency_vip, recency_churn, recency_std]),
    'Frequency': np.concatenate([freq_vip, freq_churn, freq_std]),
    'Monetary': np.concatenate([monetary_vip, monetary_churn, monetary_std])
})

# --- 2. ÖLÇEKLEME ---
ozellikler = ['Recency', 'Frequency', 'Monetary']
scaler = StandardScaler()
olcekli_veri = scaler.fit_transform(df[ozellikler])

# --- 3. DIRSEK (ELBOW) ANALİZİ ---
hata_paylari = []
K_araligi = range(1, 11)

for k in K_araligi:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(olcekli_veri)
    hata_paylari.append(km.inertia_)

# --- 4. MODELİ K=3 İLE ÇALIŞTIRMA ---
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Kume'] = kmeans.fit_predict(olcekli_veri)

# --- 5. İKİ GRAFİĞİ YAN YANA ÇİZDİRME ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Sol Grafik: Dirsek Yöntemi
ax1.plot(K_araligi, hata_paylari, 'bx-', linewidth=2)
ax1.set_title('1. Neden K=3? (Dirsek Yöntemi)', fontsize=14)
ax1.set_xlabel('Küme Sayısı (K)', fontsize=12)
ax1.set_ylabel('Hata Payı (Inertia)', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.5)

# Sağ Grafik: 500 Müşterinin Küme Dağılımı
scatter = ax2.scatter(
    df['Monetary'],
    df['Frequency'],
    c=df['Kume'],
    cmap='viridis',
    alpha=0.7,
    edgecolors='k'
)
ax2.set_title('2. Müşteriler Nasıl Gruplandı? (K=3 Dağılımı)', fontsize=14)
ax2.set_xlabel('Toplam Harcama (Monetary - TL)', fontsize=12)
ax2.set_ylabel('Sipariş Sıklığı (Frequency - Adet)', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.5)
fig.colorbar(scatter, ax=ax2, label='Küme Numarası')

# Grafikleri Ekrana Basma
plt.tight_layout()
plt.savefig('segmentasyon_sonucu.png', dpi=300)
plt.show()

# --- 6. KONSOL ÖZET RAPORU ---
print("\n" + "="*50)
print("--- KÜMELERİN ORTALAMA DAVRANIŞ PROFİLİ ---")
print("="*50)
ozet_tablo = df.groupby('Kume')[['Recency', 'Frequency', 'Monetary']].mean()
ozet_tablo['Musteri_Sayisi'] = df['Kume'].value_counts()
print(ozet_tablo.round(2))