from keybert import KeyBERT
from sentence_transformers import SentenceTransformer


def main(text):
    sent_trans = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    keyBERT_model = KeyBERT(model=sent_trans)

    def extract_terms(document, top_N=5, model=keyBERT_model,
                      diversity_threshold=0.7):
        keywords = model.extract_keywords(document, stop_words='english',
                                          keyphrase_ngram_range=(4, 4),
                                          use_mmr=True,
                                          diversity=diversity_threshold,
                                          top_n=top_N)
        return sorted(keywords, key=lambda tup: (-tup[1], tup[0]))

    best_terms = extract_terms(text)

    return best_terms

