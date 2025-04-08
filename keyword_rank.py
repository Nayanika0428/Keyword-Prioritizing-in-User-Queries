import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import Parallel, delayed
import multiprocessing
import openai

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# OpenAI API key setup (Use environment variable or secure config in production)
api_key = 'OpenAI_api_key'
openai.api_key = api_key

# Filter POS tags to exclude irrelevant terms
IRRELEVANT_POS = ['ADP', 'AUX', 'SYM', 'PUNCT']

def custom_tokenizer(text):
    """
    Tokenizes and lemmatizes input using spaCy while removing stopwords
    and irrelevant parts of speech.
    """
    doc = nlp(text)
    tokens = [
        token.lemma_.lower() for token in doc
        if token.pos_ not in IRRELEVANT_POS and not token.is_stop and not token.is_space
    ]
    return tokens

# Sample corpus
corpus = [
    "This dataset explores gene expression in cancer tissues.",
    "Human genome mapping using RNA-seq methods."
]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    analyzer='word',
    tokenizer=custom_tokenizer,
    ngram_range=(1, 2),
    sublinear_tf=True,
    smooth_idf=True
)
tfidf_matrix = vectorizer.fit_transform(corpus)

# Extract keywords and their scores
feature_names = vectorizer.get_feature_names_out()
dense = tfidf_matrix.todense()

data = []
for doc in dense:
    scores = dict(zip(feature_names, doc.tolist()[0]))
    for term, score in scores.items():
        data.append({'Term': term, 'Score': score})

df1 = pd.DataFrame(data).sort_values(by='Score', ascending=False)

# GPT Filtering Section

def is_biological_batch(terms, api_key):
    """
    Use GPT to determine if a list of terms are biological.
    """
    openai.api_key = api_key
    terms = [term for term in terms if "http://www" not in term]

    prompt = "\n".join([
        f"Given a term, determine whether it is a biological term or not. If it's biological, respond 'yes'. If it's not biological, skip to the next term.\n\nTerm: {term}"
        for term in terms
    ])

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        n=len(terms),
        stop=None,
        temperature=0.0
    )

    return [choice['text'].strip().lower() == 'yes' for choice in response.choices]

def filter_biological_terms_batch(chunk, api_key):
    """
    Splits a chunk of terms into biological and non-biological terms using GPT.
    """
    terms = list(chunk['Term'])
    is_bio_flags = is_biological_batch(terms, api_key)

    biological = [{'Term': t, 'Score': s} for t, s, b in zip(chunk['Term'], chunk['Score'], is_bio_flags) if b]
    non_biological = [{'Term': t, 'Score': s} for t, s, b in zip(chunk['Term'], chunk['Score'], is_bio_flags) if not b]

    return biological, non_biological

# Parallel processing
num_cores = multiprocessing.cpu_count()
chunk_size = max(len(df1) // num_cores, 1)
chunks = [df1[i:i + chunk_size] for i in range(0, len(df1), chunk_size)]

results = Parallel(n_jobs=-1)(
    delayed(filter_biological_terms_batch)(chunk, api_key) for chunk in chunks
)

bio_terms, non_bio_terms = zip(*results)

biological_df = pd.DataFrame([term for sublist in bio_terms for term in sublist])
non_biological_df = pd.DataFrame([term for sublist in non_bio_terms for term in sublist])

print("Biological Terms:")
print(biological_df.head())

print("\nNon-Biological Terms:")
print(non_biological_df.head())
