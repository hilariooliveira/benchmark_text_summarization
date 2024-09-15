import os
import src.corpora.corpora_utils as utils

from summarizer import Summarizer
from summarizer.sbert import SBertSummarizer
from src.summarizers.extractive_summarizers import summarize_bert


if __name__ == '__main__':

    model_name = 'ext_bert'
    # model_name = 'sbert'

    n_docs = -1

    min_len = 40
    max_len = 110

    corpus_path = f'../../data/corpus_cnn'

    summaries_dir = f'../../data/summaries/ext'

    os.makedirs(summaries_dir, exist_ok=True)

    print('\n  Reading corpus ...')

    corpus = utils.build_cnn_corpus(corpus_path)

    if n_docs != -1:
        documents = corpus.documents[0: n_docs]
    else:
        documents = corpus.documents

    print(f'\n  Total documents: {len(documents)}')

    model = None

    if model_name == 'ext_bert':
        model = Summarizer()
    elif model_name == 'sbert':
        model = SBertSummarizer('paraphrase-MiniLM-L6-v2')

    print(f'\nModel name: {model_name}\n')

    summarize_bert(documents, model, min_len, max_len, summaries_dir, model_name)
