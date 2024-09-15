from src.analyzers import concept_analyzer as concept_analyzer
from src.analyzers import ilp_analyzer as ilp_analyzer
from src.basic_classes.classes import Document, Summary


def summarize_document(document: Document, min_ngram_size: int, max_ngram_size: int, is_use_filtering: bool,
                       weighting_method: str, summary_size: int) -> Summary | None:

    concept_analyzer.extract_concepts(document, min_ngram_size, max_ngram_size, is_use_filtering)

    concept_analyzer.measure_concepts_weight(document)

    concept_analyzer.set_concepts_weight(document, weighting_method)

    filtered_sentences = []

    for sentence in document.sentences:
        if len(sentence.concepts) > 5:
            filtered_sentences.append(sentence)

    summary_sentences = ilp_analyzer.select_sentences(filtered_sentences, summary_size)

    if len(summary_sentences) > 0:
        summary_sentences.sort(key=lambda s: s.id, reverse=False)
        summary = Summary()
        text = ''
        concepts_summary = []
        for sentence in summary_sentences:
            text += sentence.text + '\n'
        text = text[:-1]
        summary.text = text
        summary.sentences = summary_sentences
        summary.concepts = concepts_summary
        return summary

    return None
