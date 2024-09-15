import os
import openai
import src.corpora.corpora_utils as corpora
import sys
import re

from tqdm import tqdm


if __name__ == '__main__':

    corpus_name = 'cnn_full'
    # corpus_name = 'cnn_teste'
    # corpus_name = 'stanford_corpus'

    openai.api_key = ''

    n_docs = 3000

    model_engine = 'text-davinci-003'
    model_name = 'davinci_003'

    # temperature = 0.3
    # temperature = 0.6
    temperature = 1.0

    corpus_path = f'/media/hilario/Novo volume/Hilario/Pesquisa/Recursos/Sumarização/' \
                  f'Corpora/{corpus_name}'

    summaries_dir = f'/media/hilario/Novo volume/Hilario/Pesquisa/Experimentos/' \
                    f'benchmark_abs_summarization/summaries/{corpus_name}'

    os.makedirs(summaries_dir, exist_ok=True)

    print(f'\nCorpus: {corpus_name}')

    print('\n  Reading corpus ...')

    if corpus_name == 'cnn_full' or corpus_name == 'cnn_teste':
        corpus = corpora.build_cnn_corpus(corpus_path)
    elif corpus_name == 'stanford_corpus':
        corpus_path = f'{corpus_path}/writer_summaries.json'
        corpus = corpora.read_stanford_corpus(corpus_path)
    else:
        print(f'\n\nCorpus Option {corpus_name} Invalid!')
        exit(-1)

    if n_docs != -1:
        documents = corpus.documents[0: n_docs]
    else:
        documents = corpus.documents

    prompt_base = 'Article: {ARTICLE}. Summarize the article in {N_SENTENCES} sentences. Summary:'
    model_name = f'{model_name}_{temperature}'.replace('.', '')

    print(f'\n  Total documents: {len(documents)}')

    print(f'\nModel Engine: {model_engine} -- {model_name}\n')

    with tqdm(total=len(documents), file=sys.stdout, colour='red', desc='Summarizing') as pbar:

        for document in documents:

            summary_file = os.path.join(summaries_dir, document.name, model_name + '.txt')

            if os.path.exists(summary_file):
                pbar.update(1)
                continue

            if len(document.text.split()) > 2000:
                tokens = document.text.split()
                tokens = tokens[:2000]
                document.text = ' '.join(tokens).strip()

            if corpus_name == 'stanford_corpus':
                num_sentences = 3
            else:
                num_sentences = len(document.gold_standard.split('\n'))

            prompt = re.sub(r'{ARTICLE}', document.text, prompt_base)
            prompt = re.sub(r'{N_SENTENCES}', str(num_sentences), prompt)
            prompt = prompt.replace('\n', ' ').strip()

            response = openai.Completion.create(model=model_engine, prompt=prompt,
                                                temperature=temperature, max_tokens=250, top_p=1,
                                                frequency_penalty=0, presence_penalty=1)

            summary = response.choices[0].text

            document.summary = summary.strip()

            corpora.save_summary(document, summaries_dir, model_name)

            pbar.update(1)
