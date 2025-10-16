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

# Now run the split
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42) # stratify=y_temp dihapus

# Assuming y is a pandas Series
print("Class counts in y_temp before final split:")
print(y_temp.value_counts())