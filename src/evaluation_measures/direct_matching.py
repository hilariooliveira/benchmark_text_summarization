import nltk
import re

from src.basic_classes.classes import Document


def compute_direct_matching(document: Document) -> dict:

    reference_summary = document.reference_summary.lower().strip()

    sentences_reference_summary = nltk.sent_tokenize(reference_summary)

    sentences_reference_summary = [re.sub(r'\W+', '', s)
                                   for s in sentences_reference_summary]

    sentences_reference_summary = set(sentences_reference_summary)

    results = {}

    for summary in document.candidate_summaries:

        sentences_candidate_summary = nltk.sent_tokenize(summary.text.lower().strip())

        sentences_candidate_summary = [re.sub(r'\W+', '', s) for s in sentences_candidate_summary]

        sentences_candidate_summary = set(sentences_candidate_summary)

        intersection = sentences_reference_summary.intersection(sentences_candidate_summary)

        direct_match = len(intersection) / len(sentences_reference_summary)

        if summary.name not in results:
            results[summary.name] = direct_match
        else:
            results[summary.name] += direct_match

    return results
