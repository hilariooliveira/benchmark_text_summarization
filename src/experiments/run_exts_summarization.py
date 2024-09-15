import os
import src.corpora.corpora_utils as utils

from summarizer import Summarizer
from summarizer.sbert import SBertSummarizer
from src.summarizers.extractive_summarizers import summarize_bert, summarize_sentence_position


if __name__ == '__main__':

    # corpus_name = 'cnn_teste'
    # corpus_name = 'stanford_corpus'
    corpus_name = 'stanford_corpus_2'

    n_docs = -1

    model_name = 'sentence_position'
    # model_name = 'ext_bert'
    # model_name = 'sbert'

    min_len = 40
    max_len = 110

    corpus_path = f'/media/hilario/Novo volume/Hilario/Pesquisa/Recursos/Sumarização/Corpora/{corpus_name}'

    summaries_dir = f'/media/hilario/Novo volume/Hilario/Pesquisa/Experimentos/benchmark_abs_summarization' \
                    f'/summaries/{corpus_name}'

    os.makedirs(summaries_dir, exist_ok=True)

    print(f'\nCorpus: {corpus_name}')

    print('\n  Reading corpus ...')

    if corpus_name == 'cnn' or corpus_name == 'cnn_teste':
        corpus = utils.build_cnn_corpus(corpus_path)
    elif corpus_name == 'cnn_dailymail':
        corpus_sets = utils.read_cnn_dailymail(corpus_path)
        corpus = corpus_sets['test']
    elif corpus_name == 'stanford_corpus':
        corpus_path = f'{corpus_path}/writer_summaries.json'
        corpus = utils.read_stanford_corpus(corpus_path)
    elif corpus_name == 'stanford_corpus_2':
        corpus_path = f'{corpus_path}/pairwise_evaluation_results.json'
        corpus = utils.read_pairwise_corpus(corpus_path)
    else:
        print(f'\n\nCorpus Option {corpus_name} Invalid!')
        exit(-1)

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

    if model_name == 'sentence_position':
        summarize_sentence_position(documents, summaries_dir, model_name)
    else:
        summarize_bert(documents, model, min_len, max_len, summaries_dir, model_name)
