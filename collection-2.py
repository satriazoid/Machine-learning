import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split


df = pd.read_csv(r"C:\.venv\Machine-learning\dataset.csv")
print(df.info())
print(df.head())

# langkah 3 - Cleaing
print(df.isnull().sum())
df = df.drop_duplicates()

sns.boxplot(x=df['IPK'])

print(df.describe())
sns.histplot(df['IPK'], bins=10, kde=True)
sns.scatterplot(x='IPK', y='Waktu_Belajar_Jam', data=df, hue='Lulus')
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")

df['Rasio_Absensi'] = df['Jumlah_Absensi'] / 14
df['IPK_x_Study'] = df['IPK'] * df['Waktu_Belajar_Jam']
df.to_csv("processed_kelulusan.csv", index=False)


X = df.drop('Lulus', axis=1)
y = df['Lulus']

# =========================================================
# LOKASI BARU: Tambahkan Split Pertama di sini (WAJIB)
# Membagi data menjadi Training (70%) dan Temporary (30%)
# Temporary (X_temp, y_temp) akan dibagi lagi menjadi Validation dan Test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)
# =========================================================


# Sekarang X_temp dan y_temp sudah terdefinisi.
# Cetak Class Counts di y_temp (untuk melihat apakah ada kelas yang hanya berisi 1 anggota)
print("\nClass counts in y_temp before final split:")
print(y_temp.value_counts()) # Ini akan menunjukkan apakah split kedua akan error


# Lakukan Split Kedua: Membagi Temporary (30%) menjadi Validation (15%) dan Test (15%)
# Catatan: Perhatikan output dari y_temp.value_counts(). Jika ada nilai 1,
# Anda akan mendapatkan error ValueError (masalah yang kita bahas sebelumnya) di sini.
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42) # stratify=y_temp dihapus


# Mencetak Shape (Ukuran Data) setelah Split
print(f"\nUkuran Data Training (X_train): {X_train.shape}")
print(f"Ukuran Data Validation (X_val): {X_val.shape}")
print(f"Ukuran Data Testing (X_test): {X_test.shape}")