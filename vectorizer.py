import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import clean_text

def vectorize_text(data_path):
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    # Rename columns if needed
    if 'response_text' in df.columns and 'class' in df.columns:
        df = df.rename(columns={'response_text': 'text', 'class': 'label'})

    # Drop missing values
    df = df.dropna(subset=['text', 'label'])

    # Normalize and map labels
    df['label'] = df['label'].astype(str).str.strip().str.lower()

    if set(df['label'].unique()).issubset({'0', '1', '0.0', '1.0'}):
        df['label'] = df['label'].astype(float).astype(int)
    else:
        df['label'] = df['label'].map({'not_flagged': 0, 'flagged': 1})
        df = df.dropna(subset=['label'])
        df['label'] = df['label'].astype(int)

    # Clean text
    df['clean_text'] = df['text'].apply(clean_text)
    df = df[df['clean_text'].str.strip() != '']

    if df.empty:
        raise ValueError(f"No valid text found after cleaning in {data_path}")

    # TF-IDF
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df['clean_text']).toarray()
    y = df['label'].values

    return X, y
