from sklearn.decomposition import PCA

def apply_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    print("\nPCA Output Sample (first 5 rows):")
    print(X_pca[:5])
    return X_pca
