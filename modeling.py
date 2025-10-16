# =========================================================
# LEMBAR KERJA PERTEMUAN 4 - DATA PREPARATION
# =========================================================

# Langkah 1 & 2: Collection
# Import Pustaka
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt # Diperlukan untuk menampilkan plot
from sklearn.model_selection import train_test_split

# Catatan: Pastikan file kelulusan_mahasiswa.csv sudah dibuat
# Ganti nama file sesuai dengan instruksi tugas (kelulusan_mahasiswa.csv)
try:
    df = pd.read_csv(r"C:\.venv\Machine-learning\dataset.csv")
except FileNotFoundError:
    print("ERROR: File 'kelulusan_mahasiswa.csv' tidak ditemukan.")
    print("Pastikan Anda sudah membuat file CSV dan namanya sudah benar.")
    exit()

print("==========================================")
print("LANGKAH 2: Collection & Info")
print("==========================================")
print(df.info())
print("\nData Head:")
print(df.head())


# Langkah 3: Cleaning
print("\n==========================================")
print("LANGKAH 3: Cleaning")
print("==========================================")

# 3a. Periksa Missing Value (Handling jika ada)
print("\nCek Missing Values:")
print(df.isnull().sum())
# Karena dataset kecil, kita asumsikan tidak ada missing value,
# tetapi jika ada, Anda akan menanganinya di sini (misalnya: df['Kolom'].fillna(df['Kolom'].median(), inplace=True))

# 3b. Hapus data duplikat
print(f"\nJumlah baris sebelum drop duplikat: {len(df)}")
df = df.drop_duplicates()
print(f"Jumlah baris setelah drop duplikat: {len(df)}")

# 3c. Identifikasi outlier dengan boxplot
plt.figure(figsize=(6, 4))
sns.boxplot(x=df['IPK'])
plt.title('Boxplot IPK untuk Identifikasi Outlier')
plt.show() # Tampilkan plot


# Langkah 4: Exploratory Data Analysis (EDA)
print("\n==========================================")
print("LANGKAH 4: Exploratory Data Analysis (EDA)")
print("==========================================")

# 4a. Hitung statistik deskriptif
print("\nStatistik Deskriptif:")
print(df.describe())

# 4b. Buat histogram distribusi IPK
plt.figure(figsize=(7, 5))
sns.histplot(df['IPK'], bins=5, kde=True) # bins diperkecil karena data sedikit
plt.title('Distribusi IPK')
plt.show()

# 4c. Visualisasi scatterplot (IPK vs Waktu Belajar)
plt.figure(figsize=(7, 5))
sns.scatterplot(x='IPK', y='Waktu_Belajar_Jam', data=df, hue='Lulus')
plt.title('IPK vs Waktu Belajar (Dikelompokkan Berdasarkan Kelulusan)')
plt.show()

# 4d. Tampilkan heatmap korelasi
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title('Heatmap Korelasi')
plt.show()


# Langkah 5: Feature Engineering
print("\n==========================================")
print("LANGKAH 5: Feature Engineering")
print("==========================================")

# 5a. Buat fitur turunan baru
df['Rasio_Absensi'] = df['Jumlah_Absensi'] / 14
df['IPK_x_Study'] = df['IPK'] * df['Waktu_Belajar_Jam']
print("\nData setelah Feature Engineering (5 Baris Terakhir):")
print(df.tail())

# 5b. Simpan file yang diproses
df.to_csv(r"C:\.venv\Machine-learning\dataset.csv", index=False)
print("\nData berhasil disimpan ke 'processed_kelulusan.csv'")


# Langkah 6: Splitting Dataset
print("\n==========================================")
print("LANGKAH 6: Splitting Dataset (Train, Val, Test)")
print("==========================================")

X = df.drop('Lulus', axis=1)
y = df['Lulus']

# Split 1: Pisahkan Training (70%) dan Temporary (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

# --- PENTING: Solusi untuk Error ValueError ---
# Karena dataset Anda sangat kecil (hanya 10 baris) dan y_temp hanya 3 baris (30%),
# stratifikasi pada split kedua hampir pasti akan gagal.
# Kita HARUS menghapus 'stratify' dari split kedua, atau membersihkan data y_temp.
# Solusi: Hapus 'stratify' dari split kedua (metode tercepat).

# Split 2: Pisahkan Temporary (30%) menjadi Validation (15%) dan Test (15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42) 
# Perhatikan: 'stratify=y_temp' sudah DIHAPUS untuk menghindari ValueError

# Cek ukuran akhir
print("\nUkuran Akhir Dataset:")
print(f"X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape} | y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape} | y_test shape: {y_test.shape}")