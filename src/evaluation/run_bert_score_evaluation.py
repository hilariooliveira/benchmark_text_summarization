import src.corpora.corpora_utils as corpora
import sys
import torch

from tqdm import tqdm
from bert_score import BERTScorer
from src.evaluation_measures.rouge_parser import generate_report


def evaluate_summaries(document_, bert_scorer_):
    for summary in document_.candidate_summaries:
        try:
            mean_p = 0.0
            mean_r = 0.0
            mean_f = 0.0
            for ref_summary in document_.references_summaries:
                precision, recall, f_measure = bert_scorer_.score([summary.text],
                                                                  [ref_summary])
                mean_p += float(precision.numpy()[0])
                mean_r += float(recall.numpy()[0])
                mean_f += float(f_measure.numpy()[0])
            mean_p /= len(document_.references_summaries)
            mean_r /= len(document_.references_summaries)
            mean_f /= len(document_.references_summaries)
            bert_score_measures = {
                'bert_score': {
                    'p': mean_p,
                    'r': mean_r,
                    'f': mean_f
                }
            }
        except Exception as e:
            print(e)
            bert_score_measures = {
                'bert_score': {
                    'p': 0.0,
                    'r': 0.0,
                    'f': 0.0
                }
            }
        summary.rouge_scores = bert_score_measures


if __name__ == '__main__':

    corpus_name = 'cnn_full'

    metric_name = 'bert_score'

    corpus_path = f'/mnt/Novo Volume/Hilario/Pesquisa/Recursos/Sumarização/Corpora/{corpus_name}'

    summaries_dir = f'/mnt/Novo Volume/Hilario/Pesquisa/Experimentos/teste/summaries/{corpus_name}'

    print(f'\nCorpus: {corpus_name}')

    print('\n  Reading corpus ...')

    if corpus_name == 'cnn_full':
        corpus = corpora.build_cnn_corpus(corpus_path)
    else:
        print(f'\n\nCorpus Option {corpus_name} Invalid!')
        exit(-1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bert_scorer = BERTScorer(lang='en', device=device)

    is_use_highlights = True

    if corpus_name == 'stanford_corpus':
        is_use_highlights = None

    prefix_file = ''

    if is_use_highlights is not None:
        if is_use_highlights:
            prefix_file = 'high'
        else:
            prefix_file = 'gold'

    print(f'\n  Total documents: {len(corpus.documents)}\n')

    print(f'\nDevice: {device}\n')

    corpora.read_summaries(corpus, summaries_dir)

    with tqdm(total=len(corpus.documents), file=sys.stdout, colour='green',
              desc='Evaluating') as pbar:
        for document in corpus.documents:
            if is_use_highlights is not None:
                document.references_summaries = [document.highlights] \
                    if is_use_highlights else [document.gold_standard]
            else:
                document.references_summaries = document.references_summaries
            evaluate_summaries(document, bert_scorer)
            pbar.update(1)

    generate_report(corpus, summaries_dir, prefix_file, metric_name)
