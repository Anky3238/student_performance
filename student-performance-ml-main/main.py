import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def main():
    # Load data
    data = pd.read_csv("student-mat.csv", sep=";")

    plt.hist(data['G3'], bins=20)
    plt.axvline(10, linestyle='--')  # pass/fail threshold
    plt.xlabel('Final Grade (G3)')
    plt.ylabel('Number of Students')
    plt.title('Distribution of Final Grades')
    plt.show()

    print("\n=== Basic Info ===")
    print(data.info())
    print("\n=== Numeric Summary (describe) ===")
    print(data.describe())

    data["pass"] = (data["G3"] >= 10).astype(int)

    print("\n=== Class Balance (pass/fail) ===")
    print(data["pass"].value_counts())

    # Supervised Learning: Classification
    features = [
        "studytime",
        "absences",
        "failures",
        "freetime",
        "goout",
        "Dalc",
        "Walc",
        "health",
        "Medu",
        "Fedu",
    ]

    X = data[features]
    y = data["pass"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("\n=== Supervised Model: Logistic Regression (class_weight='balanced') ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
          target_names=["Fail(0)", "Pass(1)"]))

    # Confusion matrix
    print("\nShowing Confusion Matrix window...")
    ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
    plt.title("Confusion Matrix - Logistic Regression")
    plt.show()

    coef_df = pd.DataFrame(
        {"feature": features, "coef": clf.coef_[0]}
    ).sort_values(by="coef", ascending=False)

    print("\n=== Logistic Regression Coefficients (Interpretation Aid) ===")
    print(coef_df)

    # Unsupervised Learning: KMeans Clustering
    cluster_features = [
        "studytime",
        "absences",
        "freetime",
        "goout",
        "Dalc",
        "Walc",
    ]

    X_cluster = data[cluster_features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    # KMeans (choose 3 clusters as a simple, explainable choice)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
    data["cluster"] = kmeans.fit_predict(X_scaled)

    print("\n=== Unsupervised Model: KMeans (k=3) ===")
    print("Cluster counts:")
    print(data["cluster"].value_counts().sort_index())

    print("\nCluster feature means (interpret clusters):")
    print(data.groupby("cluster")[cluster_features].mean())

    # Optional: see pass rate by cluster (NOT used for training, only interpretation)
    print("\nPass rate by cluster (interpretation only):")
    print(data.groupby("cluster")["pass"].mean())

    # PCA graph
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data["cluster"])
    plt.title("KMeans Clusters (PCA 2D Projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()


if __name__ == "__main__":
    main()
