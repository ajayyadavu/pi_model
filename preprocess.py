import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def clean_text(text):
    text = re.sub(r'\W+', ' ', text.lower())
    return ' '.join([word for word in text.split() if word not in stopwords.words('english')])
