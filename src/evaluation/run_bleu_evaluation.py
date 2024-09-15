import src.corpora.corpora_utils as corpora
import sys

from tqdm import tqdm
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from src.evaluation_measures.rouge_parser import generate_report


def compute_bleu(references, candidate):
    smooth = SmoothingFunction()
    list_references = []
    for ref in references:
        list_references.append(ref.split())
    return sentence_bleu(list_references, candidate.split(), smoothing_function=smooth.method2)


def evaluate_summaries_bleu(document_):
    for summary in document_.candidate_summaries:
        bleu = compute_bleu(document_.references_summaries, summary.text)
        bleu_score_measures = {
            'bleu_score': {
                'p': bleu,
                'r': bleu,
                'f': bleu
            }
        }
        summary.rouge_scores = bleu_score_measures


if __name__ == '__main__':

    metric_name = 'bleu'

    corpus_path = f'../../data/corpus_cnn'

    summaries_dir = f'../../data/summaries/abs'

    print('\n  Reading corpus ...')

    corpus = corpora.build_cnn_corpus(corpus_path)

    is_use_highlights = False

    prefix_file = ''

    if is_use_highlights is not None:
        if is_use_highlights:
            prefix_file = 'high'
        else:
            prefix_file = 'gold'

    print(f'\n  Total documents: {len(corpus.documents)}\n')

    corpora.read_summaries(corpus, summaries_dir)

    with tqdm(total=len(corpus.documents), file=sys.stdout, colour='green',
              desc='  Evaluating') as pbar:
        for document in corpus.documents:
            if is_use_highlights is not None:
                document.references_summaries = [document.highlights] \
                    if is_use_highlights else [document.gold_standard]
            evaluate_summaries_bleu(document)
            pbar.update(1)

    generate_report(corpus, summaries_dir, prefix_file, metric_name)
