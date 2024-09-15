import src.corpora.corpora_utils as utils
import sys
import re

from tqdm import tqdm


if __name__ == '__main__':

    corpus_name = 'cnn_teste'

    corpus_path = f'/media/hilario/Novo volume/Hilario/Pesquisa/Recursos/Sumarização/Corpora/{corpus_name}'

    summaries_source_dir = f'/media/hilario/Novo volume/Hilario/Pesquisa/Experimentos/' \
                           f'benchmark_abs_summarization/summaries/{corpus_name}'

    summaries_destiny_dir = f'/media/hilario/Novo volume/Hilario/Pesquisa/Experimentos/' \
                            f'benchmark_abs_summarization/summaries/cnn_full'

    print(f'\nCorpus: {corpus_name}')

    print('\n  Reading corpus ...')

    corpus = utils.build_cnn_corpus(corpus_path)

    print(f'\n  Total documents: {len(corpus.documents)}\n')

    utils.read_summaries(corpus, summaries_source_dir)

    with tqdm(total=len(corpus.documents), file=sys.stdout, colour='green', desc='Copying Summaries') as pbar:
        for document in corpus.documents:
            for summary in document.candidate_summaries:
                if 'davinci_003_' in summary.name:
                    new_summary_text = re.sub(r'\n*\d+\.', '\n', summary.text).strip()
                    document.summary = new_summary_text
                    model_name = summary.name
                    utils.save_summary(document, summaries_destiny_dir, model_name)
            pbar.update(1)
