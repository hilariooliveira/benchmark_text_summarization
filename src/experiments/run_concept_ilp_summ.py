import os
import sys

from src.corpora import corpora_utils as corpora
from src.analyzers.nlp_analyzer import NLPAnalyzer
from src.utils import utils
from src.summarizers import concept_ilp_summ as concept_ilp
from tqdm import tqdm


if __name__ == '__main__':

    n_docs = -1

    corpus_path = f'../../data/corpus_cnn'

    summaries_dir = f'../../data/summaries/ext'

    os.makedirs(summaries_dir, exist_ok=True)

    corpus = corpora.build_cnn_corpus(corpus_path)

    if n_docs != -1:
        documents = corpus.documents[0: n_docs]
    else:
        documents = corpus.documents

    print(f'\nTotal documents: {len(documents)}')

    min_ngram_size = 1
    max_ngram_size = 2

    is_use_filtering = True

    weighting_method = 'comb_mult'
    # weighting_method = 'sent_freq'
    # weighting_method = 'sent_pos'

    summary_name = f'ilp_{weighting_method}_{min_ngram_size}_{max_ngram_size}'

    nlp_analyzer = NLPAnalyzer()

    summary_size = 100

    print(f'\nSummarization method: {summary_name}\n')

    with tqdm(total=len(documents), file=sys.stdout, colour='red', desc='Summarizing') as pbar:

        for document in documents:

            nlp_analyzer.process_document(document)

            summary = concept_ilp.summarize_document(document, min_ngram_size, max_ngram_size,
                                                     is_use_filtering, weighting_method, summary_size)

            if summary is not None:
                document_dir = os.path.join(summaries_dir, document.name)
                if not os.path.exists(document_dir):
                    os.mkdir(document_dir)
                summary_file = os.path.join(document_dir, summary_name + '.txt')
                utils.save_file(summary_file, summary.text)

            pbar.update(1)
