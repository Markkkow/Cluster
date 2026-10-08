# Text Clustering - K-Means + TF-IDF + Confusion Matrix
# Standalone version for running directly from GitHub.

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    confusion_matrix,
    accuracy_score,
    classification_report,
)

from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore")


# ============================================================
# 1. LOAD DATA
# ============================================================

# Ganti dengan lokasi CSV kamu.
# Contoh:
# DATA_PATH = "dummy_text_clustering.csv"
DATA_PATH = "dummy_text_clustering.csv"

df = pd.read_csv(DATA_PATH)

print("Ukuran data:", df.shape)
print(df.head())


# ============================================================
# 2. CEK KOLOM
# ============================================================

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

if TEXT_COLUMN not in df.columns:
    raise ValueError(
        f"Kolom '{TEXT_COLUMN}' tidak ditemukan. "
        f"Kolom tersedia: {list(df.columns)}"
    )

if LABEL_COLUMN not in df.columns:
    raise ValueError(
        f"Kolom '{LABEL_COLUMN}' tidak ditemukan. "
        f"Kolom tersedia: {list(df.columns)}"
    )


# ============================================================
# 3. TEXT PREPROCESSING
# ============================================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    text = re.sub(r"[^a-zA-ZÀ-ÿ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["clean_text"] = df[TEXT_COLUMN].fillna("").apply(clean_text)

print("\nContoh hasil preprocessing:")
print(df[[TEXT_COLUMN, "clean_text"]].head())


# ============================================================
# 4. TF-IDF
# ============================================================

# Stopword Bahasa Indonesia sederhana.
indonesian_stopwords = [
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk",
    "dengan", "pada", "adalah", "akan", "dalam", "atau", "juga",
    "karena", "oleh", "sebagai", "lebih", "sudah", "telah",
    "tidak", "ada", "bagi", "para", "agar", "dapat", "bisa",
    "saat", "sebuah", "seperti", "mereka", "kami", "kita",
]

vectorizer = TfidfVectorizer(
    stop_words=indonesian_stopwords,
    ngram_range=(1, 2),
    max_features=3000,
    sublinear_tf=True,
)

X = vectorizer.fit_transform(df["clean_text"])

print("\nUkuran matriks TF-IDF:", X.shape)


# ============================================================
# 5. MENENTUKAN JUMLAH CLUSTER - ELBOW
# ============================================================

k_values = range(2, min(10, len(df) - 1) + 1)
inertias = []

for k in k_values:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    model.fit(X)
    inertias.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(list(k_values), inertias, marker="o")
plt.xlabel("Jumlah Cluster (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.grid(True)
plt.show()


# ============================================================
# 6. SILHOUETTE SCORE
# ============================================================

silhouette_values = []

for k in k_values:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    labels_k = model.fit_predict(X)

    if len(set(labels_k)) > 1:
        score = silhouette_score(X, labels_k)
    else:
        score = np.nan

    silhouette_values.append(score)

plt.figure(figsize=(8, 5))
plt.plot(list(k_values), silhouette_values, marker="o")
plt.xlabel("Jumlah Cluster (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score")
plt.grid(True)
plt.show()

best_k = list(k_values)[int(np.nanargmax(silhouette_values))]

print("K terbaik berdasarkan Silhouette Score:", best_k)


# ============================================================
# 7. K-MEANS
# ============================================================

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans.fit_predict(X)

df["cluster"] = cluster_labels

print("\nDistribusi cluster:")
print(df["cluster"].value_counts().sort_index())


# ============================================================
# 8. PCA VISUALIZATION
# ============================================================

pca = PCA(n_components=2, random_state=42)

X_dense = X.toarray()
X_pca = pca.fit_transform(X_dense)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=cluster_labels,
    alpha=0.7
)

plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Visualisasi Cluster dengan PCA")
plt.colorbar(scatter, label="Cluster")
plt.grid(True)
plt.show()

print(
    "Explained variance PCA:",
    round(pca.explained_variance_ratio_.sum() * 100, 2),
    "%"
)


# ============================================================
# 9. TOP TERMS SETIAP CLUSTER
# ============================================================

terms = vectorizer.get_feature_names_out()

print("\nTop terms setiap cluster:")

for cluster_id in range(best_k):
    center = kmeans.cluster_centers_[cluster_id]
    top_indices = center.argsort()[::-1][:10]
    top_terms = terms[top_indices]

    print(f"\nCluster {cluster_id}:")
    print(", ".join(top_terms))


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

true_labels = df[LABEL_COLUMN].astype(str).values
pred_labels = df["cluster"].values

unique_true = np.unique(true_labels)

# Mapping cluster -> label menggunakan Hungarian Algorithm
cm_raw = confusion_matrix(true_labels, pred_labels)

row_ind, col_ind = linear_sum_assignment(-cm_raw)

cluster_to_label = {
    col: unique_true[row]
    for row, col in zip(row_ind, col_ind)
}

mapped_predictions = np.array([
    cluster_to_label.get(cluster, f"Cluster_{cluster}")
    for cluster in pred_labels
])

cm = confusion_matrix(
    true_labels,
    mapped_predictions,
    labels=unique_true
)

accuracy = accuracy_score(true_labels, mapped_predictions)

print("\nConfusion Matrix:")
print(cm)

print("\nAccuracy setelah optimal cluster-label mapping:")
print(round(accuracy * 100, 2), "%")


# ============================================================
# 11. VISUAL CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(7, 6))
plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.colorbar()

plt.xticks(
    range(len(unique_true)),
    unique_true,
    rotation=45,
    ha="right"
)
plt.yticks(
    range(len(unique_true)),
    unique_true
)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.show()


# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")
print(
    classification_report(
        true_labels,
        mapped_predictions,
        labels=unique_true,
        zero_division=0
    )
)


# ============================================================
# 13. HELPER INTERPRETASI OTOMATIS
# ============================================================

best_silhouette = np.nanmax(silhouette_values)

print("\n" + "=" * 60)
print("HELPER INTERPRETASI OTOMATIS")
print("=" * 60)

# Silhouette
if best_silhouette >= 0.50:
    silhouette_text = "Struktur cluster cukup baik dan pemisahan antar-cluster relatif jelas."
elif best_silhouette >= 0.25:
    silhouette_text = "Struktur cluster cukup, tetapi masih terdapat beberapa overlap antar-cluster."
else:
    silhouette_text = "Pemisahan cluster masih lemah sehingga hasil clustering perlu diperiksa kembali."

print("\n1. Silhouette Score")
print(f"   Nilai terbaik: {best_silhouette:.3f}")
print(f"   Interpretasi: {silhouette_text}")

# Distribusi
cluster_counts = df["cluster"].value_counts().sort_index()

print("\n2. Distribusi Cluster")
for cluster_id, count in cluster_counts.items():
    percentage = count / len(df) * 100
    print(f"   Cluster {cluster_id}: {count} data ({percentage:.1f}%)")

# Top terms
print("\n3. Karakteristik Cluster")
for cluster_id in range(best_k):
    center = kmeans.cluster_centers_[cluster_id]
    top_indices = center.argsort()[::-1][:5]
    top_terms = terms[top_indices]

    print(
        f"   Cluster {cluster_id} paling banyak ditandai oleh: "
        + ", ".join(top_terms)
    )

# Accuracy
print("\n4. Confusion Matrix")
print(f"   Accuracy: {accuracy * 100:.2f}%")

if accuracy >= 0.80:
    accuracy_text = "Hasil clustering memiliki kecocokan yang tinggi terhadap label aktual."
elif accuracy >= 0.60:
    accuracy_text = "Hasil clustering memiliki kecocokan sedang terhadap label aktual."
else:
    accuracy_text = "Hasil clustering memiliki kecocokan yang masih rendah terhadap label aktual."

print(f"   Interpretasi: {accuracy_text}")

# PCA
pca_variance = pca.explained_variance_ratio_.sum() * 100

print("\n5. PCA")
print(f"   Dua komponen PCA menjelaskan {pca_variance:.2f}% variasi data.")

if pca_variance >= 70:
    print("   Interpretasi: visualisasi 2D cukup representatif terhadap data.")
elif pca_variance >= 50:
    print("   Interpretasi: visualisasi 2D cukup membantu, tetapi belum mewakili seluruh variasi data.")
else:
    print("   Interpretasi: visualisasi 2D hanya mewakili sebagian kecil variasi data.")

# Kesimpulan
print("\n6. Kesimpulan")
if best_silhouette >= 0.50 and accuracy >= 0.80:
    print(
        "   Secara umum, hasil clustering cukup baik: "
        "cluster relatif terpisah dan memiliki kecocokan tinggi dengan label aktual."
    )
elif best_silhouette >= 0.25 and accuracy >= 0.60:
    print(
        "   Secara umum, hasil clustering cukup baik, "
        "tetapi masih terdapat overlap atau ketidaksesuaian antar-cluster."
    )
else:
    print(
        "   Secara umum, hasil clustering masih perlu diperbaiki, "
        "misalnya melalui preprocessing, representasi fitur, atau pemilihan jumlah cluster."
    )

print("\nSelesai.")
