from vectorize import vectorize_text
from reduce_dim import apply_pca
from train_model import train_and_evaluate
from visualize import plot_pca

X, y = vectorize_text("data.csv")
X_pca = apply_pca(X, n_components=2)
model, acc = train_and_evaluate(X_pca, y)

print(f"Model Accuracy: {acc}")
plot_pca(X_pca, y)
