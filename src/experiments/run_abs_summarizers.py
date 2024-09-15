import os
import torch

from src.corpora import corpora_utils as corpora
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.summarizers import abstractive_summarizers as abs_summ


if __name__ == '__main__':

    n_docs = -1

    min_len = 40
    max_len = 100

    num_beams = 5

    summarization_name = 'pegasus'
    # summarization_name = 'brio'
    # summarization_name = 'flan_t5_large'
    # summarization_name = 'bart_large-cnn'

    corpus_path = f'../../data/corpus_cnn'

    summaries_dir = f'../../data/summaries/abs'

    print('\n  Reading corpus ...')

    corpus = corpora.build_cnn_corpus(corpus_path)

    tokenizer = None

    model_name = None
    model = None
    model_path = None

    summarize = None

    if summarization_name == 'bart_large-cnn':
        model_path = 'facebook/bart-large-cnn'
        model_name = 'Bart_Large'
        tokenizer = BartTokenizer.from_pretrained(model_path)
        model = BartForConditionalGeneration.from_pretrained(model_path)
        summarize = abs_summ.summarize_bart
    elif summarization_name == 'pegasus':
        model_path = 'google/pegasus-cnn_dailymail'
        tokenizer = PegasusTokenizer.from_pretrained(model_path)
        model = PegasusForConditionalGeneration.from_pretrained(model_path)
        summarize = abs_summ.summarize_pegasus
        model_name = 'Pegasus'
    elif summarization_name == 'flan_t5_large':
        model_path = 'spacemanidol/flan-t5-large-cnndm'
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        summarize = abs_summ.summarize_flan_t5
        model_name = summarization_name
    elif summarization_name == 'brio':
        model_path = 'Yale-LILY/brio-cnndm-uncased'
        tokenizer = BartTokenizer.from_pretrained(model_path)
        model = BartForConditionalGeneration.from_pretrained(model_path)
        summarize = abs_summ.summarize_brio
        model_name = summarization_name
    else:
        print('\nERROR.')
        exit(-1)

    if n_docs != -1:
        documents = corpus.documents[0: n_docs]
    else:
        documents = corpus.documents

    print(f'\n  Total documents: {len(documents)}')

    os.makedirs(summaries_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'\nDevice: {device}')

    model = model.to(device)

    print(f'\nSummary name: {summarization_name} - {model_name}')

    print('\nSummarizing ....\n')

    summarize(documents[:5], tokenizer, model, min_len, max_len, num_beams, summaries_dir,
              model_name, device)
