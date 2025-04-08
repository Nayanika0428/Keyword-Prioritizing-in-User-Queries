# Keyword-Prioritizing-in-User-Queries using NLP+TF-IDF+GPT

## 🔍 Project Summary

This project aims to enhance search functionalities in biomedical platforms by **prioritizing biologically significant keywords**—even when they aren't tagged by Named Entity Recognition (NER) models.

For example:
In a query like _"differentially expressed miRNAs in lung cancer"_, existing systems may only recognize "lung cancer" and ignore "miRNAs"—a highly important biological keyword.

This project proposes a **TF-IDF based keyword ranking system** to identify and prioritize such terms.

</br>

## ✅ Key Features
1. **Corpus Construction & Preprocessing**:
* Metadata fields (abstract, description) are used as input corpus.
* Uses spaCy to tokenize and filter out irrelevant parts of speech.

2. **Keyword Scoring using TF-IDF**:
* TF-IDF is applied on clean tokens to generate importance scores.
* Generates a ranked list of domain-relevant keywords.

3. **GPT-based Filtering**: 
* GPT-3 (text-davinci-003) is used to semantically filter biological vs. non-biological terms.
* Uses batch processing and parallelization via joblib.

</br>

## 📦 Tech Stack
* Python, Pandas, spaCy

* scikit-learn (TF-IDF)

* OpenAI GPT (for refinement)

* Joblib + multiprocessing

</br>

## 📌 Why This Matters

This approach improves the **recall** and **relevance** of search results in bioinformatics platforms by:
* Understanding biological importance beyond simple entity recognition.
* Helping users find more accurate datasets by considering nuanced domain-specific terms.

</br>

## 🚀 Output
* biological_df: Refined, ranked list of biological keywords.
* non_biological_df: Filtered-out general/non-bio terms.

</br>

## 🚀 Future Work

- Integrate results with Elasticsearch.
- Expand to include more repositories like TCGA and HPA.
- Extend to other data types like proteomics and clinical data.
