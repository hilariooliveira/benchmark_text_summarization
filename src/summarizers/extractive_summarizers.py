import sys
import src.corpora.corpora_utils as utils
import spacy

from src.basic_classes.classes import Document
from summarizer.summary_processor import SummaryProcessor
from tqdm import tqdm


def summarize_sentence_position(documents: list[Document], summaries_dir: str, model_name: str):
    nlp = spacy.load('en_core_web_sm')
    with tqdm(total=len(documents), file=sys.stdout, colour='red', desc='  Summarizing') as pbar:
        for document in documents:
            doc = nlp(document.text)
            if len(document.gold_standard) > 0:
                num_sentences = len(document.gold_standard.split('\n'))
            else:
                num_sentences = len(list(document.references_summaries)[0].split('\n'))
            sentences = [sentence.text for sentence in doc.sents]
            sentences_summary = sentences[:num_sentences]
            document.summary = ' '.join(sentences_summary).strip()
            utils.save_summary(document, summaries_dir, model_name)
            pbar.update(1)


"""
    https://github.com/dmmiller612/bert-extractive-summarizer
"""


def summarize_bert(documents: list[Document], model: SummaryProcessor, min_len: int, max_len: int,
                   summaries_dir: str, model_name: str):
    with tqdm(total=len(documents), file=sys.stdout, colour='red', desc='  Summarizing') as pbar:
        for document in documents:
            if len(document.gold_standard) > 0:
                num_sentences = len(document.gold_standard.split('\n'))
            else:
                num_sentences = len(list(document.references_summaries)[0].split('\n'))
            summary = model(document.text, min_length=min_len, max_length=max_len, num_sentences=num_sentences)
            document.summary = summary
            utils.save_summary(document, summaries_dir, model_name)
            pbar.update(1)
