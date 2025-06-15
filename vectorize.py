import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import clean_text

def vectorize_text(data_path):
    df = pd.read_csv(data_path, quoting=1)
    df = df.dropna(subset=['text', 'label'])
    df['label'] = df['label'].astype(int)

    # Clean text
    df['clean_text'] = df['text'].apply(clean_text)

    print("Original Text Sample:")
    print(df['text'].head())

    print("\nCleaned Text Sample:")
    print(df['clean_text'].head())

    # TF-IDF vectorization
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df['clean_text']).toarray()

    print("\nTF-IDF Feature Shape:", X.shape)
    print("TF-IDF Feature Sample (first row):")
    print(X[0])

    y = df['label'].values
    return X, y
