"""
Retriever File contains two classes for document retriever. The code is adpated from
https://colab.research.google.com/github/UKPLab/sentence-transformers/blob/master/examples/applications/retrieve_rerank/retrieve_rerank_simple_wikipedia.ipynb#scrollTo=UlArb7kqN3Re
"""
from sentence_transformers import util
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction import _stop_words
from tqdm.autonotebook import tqdm
import string
import numpy as np


def _bm25_tokenizer(text):
    tokenized_doc = []
    for token in text.lower().split():
        token = token.strip(string.punctuation)

        if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
            tokenized_doc.append(token)
    return tokenized_doc


class LexicalSearch(object):
    def __init__(self):
        self.bm25 = None
        self.context = None

    def _init(self):
        pass

    def fit(self, context):
        self.context = list(set(context))
        tokenized_corpus = []
        for passage in tqdm(self.context):
            tokenized_corpus.append(_bm25_tokenizer(passage))

        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query, top_k=5):
        bm25_scores = self.bm25.get_scores(_bm25_tokenizer(query))
        top_n = np.argpartition(bm25_scores, -5)[-5:]
        bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
        bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)

        results = []
        for i in range(top_k):
            results.append(self.context[bm25_hits[i]['corpus_id']])

        return results


class PassageRanking(object):
    """
    PassageRanking Class
    """

    def __init__(self, bi_encoder, cross_encoder):
        """
        @param: model - model to extract sentence embeddings
        """
        self.context_embeddings = None
        self.context = None
        self.bi_encoder = bi_encoder
        self.cross_encoder = cross_encoder

    def fit(self, context):
        """
        Get all context embeddings
        @param: context - list context texts
        """
        self.context = list(set(context))
        # We encode all passages into our vector space.
        self.context_embeddings = self.bi_encoder.encode(self.context, convert_to_tensor=True, show_progress_bar=True)
        self.context_embeddings = self.context_embeddings.cuda()

    def search(self, query, top_k=5):
        # Semantic Search
        # Encode the query using the bi-encoder and find potentially relevant passages
        question_embedding = self.bi_encoder.encode(query, convert_to_tensor=True)
        question_embedding = question_embedding.cuda()
        hits = util.semantic_search(question_embedding, self.context_embeddings, top_k=top_k)
        hits = hits[0]  # Get the hits for the first query

        # Re-Ranking
        # Now, score all retrieved passages with the cross_encoder
        cross_inp = [[query, self.context[hit['corpus_id']]] for hit in hits]
        cross_scores = self.cross_encoder.predict(cross_inp)

        # Sort results by the cross-encoder scores
        for idx in range(len(cross_scores)):
            hits[idx]['cross-score'] = cross_scores[idx]

        hits = sorted(hits, key=lambda x: x['score'], reverse=True)

        results = []
        for i in range(top_k):
            results.append(self.context[hits[i]['corpus_id']])

        return results
