import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from contextlib import redirect_stdout
from pathlib import Path
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    silhouette_score,
)

import matplotlib.pyplot as plt


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load the mobile price dataset from CSV.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found at {path.resolve()}")
    df = pd.read_csv(path)
    return df


def basic_eda(df: pd.DataFrame) -> None:
    """
    Print basic information about the dataset.
    """
    print("=== BASIC EDA ===")
    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nMissing values per column:\n", df.isnull().sum())
    print("\nClass distribution (price_range):\n", df["price_range"].value_counts())


def train_test_split_scaled(df: pd.DataFrame):
    """
    Split into train/test and apply StandardScaler on features.
    """
    X = df.drop("price_range", axis=1)
    y = df["price_range"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,  # keep class balance
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler


def train_evaluate_svm(X_train_scaled, X_test_scaled, y_train, y_test):
    """
    Train SVM classifier and print / save evaluation metrics.
    """
    svm_clf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
    svm_clf.fit(X_train_scaled, y_train)
    y_pred = svm_clf.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    print("\n=== SVM Results ===")
    print("Accuracy:", acc)
    print("\nClassification report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("SVM Confusion Matrix")
    plt.tight_layout()
    plt.savefig("results/svm_confusion_matrix.png", dpi=300)
    plt.close()

    return svm_clf, acc, cm


def train_evaluate_knn(X_train_scaled, X_test_scaled, y_train, y_test, k: int = 5):
    """
    Train KNN classifier and print / save evaluation metrics.
    """
    knn_clf = KNeighborsClassifier(n_neighbors=k)
    knn_clf.fit(X_train_scaled, y_train)
    y_pred = knn_clf.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== KNN Results (k={k}) ===")
    print("Accuracy:", acc)
    print("\nClassification report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"KNN Confusion Matrix (k={k})")
    plt.tight_layout()
    plt.savefig(f"results/knn_confusion_matrix_k{k}.png", dpi=300)
    plt.close()

    return knn_clf, acc, cm


def run_kmeans_clustering(df: pd.DataFrame, scaler: StandardScaler, n_clusters: int = 4):
    """
    Run K-Means clustering and relate clusters to price_range.
    """
    feature_cols = [col for col in df.columns if col != "price_range"]
    X = df[feature_cols].values

    # Scale features before clustering
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    df_clusters = df.copy()
    df_clusters["cluster"] = cluster_labels

    sil_score = silhouette_score(X_scaled, cluster_labels)
    print("\n=== K-Means Clustering Results ===")
    print("Silhouette score:", sil_score)

    print("\nCluster vs price_range contingency table:")
    print(pd.crosstab(df_clusters["cluster"], df_clusters["price_range"]))

    # 2D visualisation using PCA
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure()
    plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=cluster_labels,
        alpha=0.6,
        edgecolor="k",
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("K-Means Clusters (PCA 2D Projection)")
    plt.tight_layout()
    plt.savefig("results/kmeans_pca_clusters.png", dpi=300)
    plt.close()

    return kmeans, df_clusters, sil_score


def main():
    # Adjust this if you move the file (e.g., "data/train.csv")
    csv_path = "data/train.csv"

    # Ensure results folder exists
    Path("results").mkdir(exist_ok=True)

    # 1) Load and EDA
    df = load_dataset(csv_path)
    basic_eda(df)

    # 2) Train/test split and scaling
    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled,
        scaler,
    ) = train_test_split_scaled(df)

    # 3) Supervised models
    svm_clf, svm_acc, svm_cm = train_evaluate_svm(
        X_train_scaled, X_test_scaled, y_train, y_test
    )

    knn_clf, knn_acc, knn_cm = train_evaluate_knn(
        X_train_scaled, X_test_scaled, y_train, y_test, k=5
    )

    # 4) Unsupervised clustering
    kmeans, df_clusters, sil_score = run_kmeans_clustering(
        df, scaler, n_clusters=4
    )

    # 5) Save summary metrics for report use
    summary = {
        "model": ["SVM", "KNN", "KMeans"],
        "accuracy_or_silhouette": [svm_acc, knn_acc, sil_score],
    }
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv("results/model_summary.csv", index=False)
    print("\nSaved results/model_summary.csv")


if __name__ == "__main__":
    # Make sure results folder exists
    Path("results").mkdir(exist_ok=True)

    # File to store terminal output
    log_path = Path("results") / "viva_terminal_output.txt"

    # Save all prints + model output here
    with open(log_path, "w", encoding="utf-8") as f, redirect_stdout(f):
        main()

    print(f"Run complete. Terminal output saved to {log_path}")

    main()
