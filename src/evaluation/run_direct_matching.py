import src.corpora.corpora_utils as utils
import sys
import numpy as np
import os
import json

from src.evaluation_measures.direct_matching import compute_direct_matching
from tqdm import tqdm


if __name__ == '__main__':

    corpus_path = f'../../data/corpus_cnn'

    summaries_dir = f'../../data/summaries/abs'

    print('\n  Reading corpus ...')

    corpus = utils.build_cnn_corpus(corpus_path)

    print(f'\n  Total documents: {len(corpus.documents)}\n')

    is_use_highlights = True

    prefix_file = ''

    if is_use_highlights is not None:
        if is_use_highlights:
            prefix_file = 'high'
        else:
            prefix_file = 'gold'

    utils.read_summaries(corpus, summaries_dir)

    all_results = {}

    with tqdm(total=len(corpus.documents), file=sys.stdout, colour='green',
              desc='  Evaluating') as pbar:

        for document in corpus.documents:

            document.reference_summary = document.highlights if is_use_highlights \
                else document.gold_standard

            results = compute_direct_matching(document)

            for name, value in results.items():
                if name not in all_results:
                    all_results[name] = {'dm': [value]}
                else:
                    all_results[name]['dm'].append(value)

            pbar.update(1)

    csv_content = 'System;Mean Direct Matching;Std Direct Matching\n'

    for system_name, values_dict in all_results.items():
        values = values_dict['dm']
        mean_values = str(np.mean(values)).replace(".", ",")
        std_values = str(np.std(values)).replace(".", ",")
        csv_content += f'{system_name};{mean_values};{std_values}\n'

    csv_content = csv_content[:-1]

    results_file = os.path.join(summaries_dir, f'{prefix_file}_direct_matching.csv')

    with open(results_file, 'w') as file:
        file.write(csv_content)

    results_file = os.path.join(summaries_dir, f'{prefix_file}_direct_matching.json')

    with open(file=results_file, mode='w') as json_file:
        json.dump(all_results, json_file, indent=4)
