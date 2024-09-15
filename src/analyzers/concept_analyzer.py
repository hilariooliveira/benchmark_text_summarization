from src.basic_classes.classes import Document
from src.utils.ngram_extractor import build_ngrams
from src.utils import utils


def extract_concepts(document: Document, min_ngram_size: int, max_ngram_size: int, is_use_filtering: bool):
    all_concepts = []
    for sentence in document.sentences:
        concepts = build_ngrams(sentence.tokens, min_ngram_size, max_ngram_size)
        if is_use_filtering:
            concepts = utils.filter_concepts(concepts)
        if concepts:
            id_concept = 1
            for concept in concepts:
                concept.id = id_concept
                id_concept += 1
            sentence.concepts = concepts
            all_concepts.extend(concepts)
    document.concepts = all_concepts


def measure_concepts_weight(document: Document):
    for concept in document.concepts:
        sent_freq = 0
        sent_pos = 0
        for sentence in document.sentences:
            if concept in sentence.concepts:
                sent_freq += 1
        for sentence in document.sentences:
            if concept in sentence.concepts:
                sent_pos = 1 - ((sentence.id - 1) / (1.0 * len(document.sentences)))
                break
        combined_weight_sum = (sent_freq + sent_pos) / 2
        combined_weight_mult = sent_freq * sent_pos
        concept.add_weight('sent_freq', sent_freq)
        concept.add_weight('sent_pos', sent_pos)
        concept.add_weight('comb_sum', combined_weight_sum)
        concept.add_weight('comb_mult', combined_weight_mult)


def set_concepts_weight(document: Document, method_label: str):
    highest_weight = 0
    for sentence in document.sentences:
        for concept in sentence.concepts:
            concept.weight = concept.weights[method_label]
            if concept.weight > highest_weight:
                highest_weight = concept.weight
    for sentence in document.sentences:
        for concept in sentence.concepts:
            concept.weight /= highest_weight
