import src.corpora.corpora_utils as corpora
import sys

from src.evaluation_measures import rouge_parser
from tqdm import tqdm


if __name__ == '__main__':

    corpus_name = 'cnn_full'

    metric_name = 'rouge'

    corpus_path = f'/mnt/Novo Volume/Hilario/Pesquisa/Recursos/Sumarização/Corpora/{corpus_name}'

    summaries_dir = f'/mnt/Novo Volume/Hilario/Pesquisa/Experimentos/teste/summaries/{corpus_name}'

    print(f'\nCorpus: {corpus_name}')

    print('\n  Reading corpus ...')

    if corpus_name == 'cnn_full':
        corpus = corpora.build_cnn_corpus(corpus_path)
    else:
        print(f'\n\nCorpus Option {corpus_name} Invalid!')
        exit(-1)

    print(f'\n  Total documents: {len(corpus.documents)}')

    is_use_highlights = True

    if corpus_name == 'stanford_corpus':
        is_use_highlights = None

    is_use_stemming = False
    limit_words = 100

    prefix_file = ''

    if is_use_highlights is not None:
        if is_use_highlights:
            prefix_file = 'high'
        else:
            prefix_file = 'gold'

    print(f'\nUsing: {prefix_file}\n')

    corpora.read_summaries(corpus, summaries_dir)

    with tqdm(total=len(corpus.documents), file=sys.stdout, colour='green', desc='Evaluating') as pbar:
        for document in corpus.documents:
            if is_use_highlights is not None:
                document.references_summaries = [document.highlights] \
                    if is_use_highlights else [document.gold_standard]
            rouge_parser.evaluate_summaries(document, is_use_stemming, limit_words)
            pbar.update(1)

    rouge_parser.generate_report(corpus, summaries_dir, prefix_file, metric_name)
