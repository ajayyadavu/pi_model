from vectorizer import vectorize_text
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from train_model import train_and_evaluate  # ✅ new import

# === Load both datasets ===
X1, y1 = vectorize_text("data.csv")
X2, y2 = vectorize_text("data1.csv")
#x3, y3 = vectorize_text("data2.csv")

# === Train & Evaluate ===
model1, acc1 = train_and_evaluate(X1, y1)
model2, acc2 = train_and_evaluate(X2, y2)

print(f"Accuracy for data.csv: {acc1:.2f}")
print(f"Accuracy for data1.csv: {acc2:.2f}")

# === Function to plot PCA ===
def plot_pca(X, y, title):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.7)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(title)
    plt.legend(*scatter.legend_elements(), title="Label")
    plt.grid(True)
    plt.show()

# === Visualize PCA ===
plot_pca(X1, y1, "PCA - data.csv")
plot_pca(X2, y2, "PCA - data1.csv")
